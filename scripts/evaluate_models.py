"""Held-out evaluation for all 5 recommenders.

Splits ratings 80/20, retrains rating-sensitive models (collaborative SVD,
clustering, neural) on the train split only so metrics reflect genuine
generalization, then reports:
  - RMSE / MAE for the two rating-predicting models (collaborative, neural)
  - Precision@K / Recall@K for all five models' top-N lists, evaluated
    against each sampled test user's held-out "liked" movies (rating >= 4)

Results are written to models/metrics.json for the app's Model Comparison
page. Requires TensorFlow (see requirements-train.txt) - this script is a
dev-time tool, not part of the deployed app.

Run: python scripts/evaluate_models.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import data_utils, evaluate
from src.recommenders import clustering, collaborative, content_based, neural, popularity
from src.recommenders.base import low_signal_movie_ids
from train_models import build_and_train_neural

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
K = 10
N_EVAL_USERS = 150
MIN_TRAIN_RATINGS = 5
SEED = 42


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    movies = data_utils.load_movies()
    ratings = data_utils.load_ratings()
    tags = data_utils.load_tags()

    print("Splitting ratings 80/20...")
    shuffled = ratings.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = int(len(shuffled) * 0.8)
    train_ratings, test_ratings = shuffled.iloc[:split], shuffled.iloc[split:]

    train_by_user = {u: g.set_index("movieId")["rating"].to_dict() for u, g in train_ratings.groupby("userId")}
    test_by_user = {u: g.set_index("movieId")["rating"].to_dict() for u, g in test_ratings.groupby("userId")}

    print("Retraining collaborative SVD on train split...")
    collab_artifacts = collaborative.train_and_save(
        train_ratings, out_path=MODELS_DIR / "_eval_svd.joblib"
    )

    print("Retraining KMeans clustering on train split...")
    cluster_artifacts = clustering.train_and_save(
        movies, train_ratings, out_path=MODELS_DIR / "_eval_clustering.joblib"
    )

    print("Retraining neural model on train split (fewer epochs for speed)...")
    neural_weights = build_and_train_neural(movies, train_ratings, epochs=5)
    neural_model = neural.NeuralRecommender(neural_weights)

    print("Loading content-based + popularity artifacts (content is rating-independent)...")
    content_artifacts = content_based.load()
    pop_df = popularity.train_and_save(movies, train_ratings, out_path=MODELS_DIR / "_eval_popularity.csv")
    low_signal = low_signal_movie_ids(pop_df)

    # --- RMSE / MAE for rating-predicting models ---
    print("Scoring RMSE/MAE...")
    global_mean_train = float(train_ratings["rating"].mean())
    collab_preds, neural_preds, actuals = [], [], []
    for row in test_ratings.itertuples():
        train_dict = train_by_user.get(row.userId, {})
        collab_preds.append(collaborative.predict_rating(train_dict, row.movieId, collab_artifacts))
        idxs, preds = neural_model.predict_for_profile(train_dict, [row.movieId])
        neural_preds.append(float(preds[0]) if len(preds) else global_mean_train)
        actuals.append(row.rating)

    rating_metrics = {
        "collaborative": {"rmse": evaluate.rmse(collab_preds, actuals), "mae": evaluate.mae(collab_preds, actuals)},
        "neural": {"rmse": evaluate.rmse(neural_preds, actuals), "mae": evaluate.mae(neural_preds, actuals)},
    }

    # --- Precision@K / Recall@K for all 5 models ---
    print(f"Scoring Precision@{K}/Recall@{K} on up to {N_EVAL_USERS} sampled users...")
    eligible_users = [
        u for u in test_by_user
        if len(train_by_user.get(u, {})) >= MIN_TRAIN_RATINGS
        and any(r >= 4.0 for r in test_by_user[u].values())
    ]
    sample_users = rng.choice(eligible_users, size=min(N_EVAL_USERS, len(eligible_users)), replace=False)

    topn_scores = {m: {"precision": [], "recall": []} for m in
                   ["popularity", "content_based", "collaborative", "clustering", "neural"]}

    for u in sample_users:
        train_dict = train_by_user.get(int(u), {})
        watched = set(train_dict.keys())
        exclude = watched | low_signal  # candidates to filter out; NOT the same as "watched" for cluster identity
        relevant = {m for m, r in test_by_user[int(u)].items() if r >= 4.0}
        if not relevant:
            continue

        recs = {
            "popularity": popularity.because_you_watched(
                pop_df, seed_genres=_seed_genres(movies, train_dict), exclude_ids=exclude, n=K
            ),
            "content_based": content_based.recommend_for_profile(train_dict, content_artifacts, n=K, exclude_ids=exclude),
            "collaborative": collaborative.recommend_for_profile(train_dict, collab_artifacts, n=K, exclude_ids=exclude),
            "clustering": clustering.recommend_for_profile(watched, cluster_artifacts, pop_df, n=K, exclude_ids=exclude),
            "neural": neural_model.recommend_for_profile(train_dict, n=K, exclude_ids=exclude),
        }
        for model_name, rec_list in recs.items():
            rec_ids = [r.movie_id for r in rec_list]
            p, r = evaluate.precision_recall_at_k(rec_ids, relevant, K)
            topn_scores[model_name]["precision"].append(p)
            topn_scores[model_name]["recall"].append(r)

    topn_metrics = {
        m: {
            f"precision_at_{K}": float(np.mean(v["precision"])) if v["precision"] else 0.0,
            f"recall_at_{K}": float(np.mean(v["recall"])) if v["recall"] else 0.0,
        }
        for m, v in topn_scores.items()
    }

    metrics = {
        "k": K,
        "n_eval_users": int(len(sample_users)),
        "rating_prediction": rating_metrics,
        "top_n": topn_metrics,
        "notes": (
            "80/20 random split of ratings.csv. Collaborative/clustering/neural retrained "
            "on the train split only; content-based and popularity use rating-weighted "
            "profiles built from each test user's train-split ratings only. New/cold "
            "profiles fall back to popularity - not reflected in these per-user metrics."
        ),
    }

    out_path = MODELS_DIR / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")

    for p in ["_eval_svd.joblib", "_eval_clustering.joblib", "_eval_popularity.csv"]:
        fp = MODELS_DIR / p
        if fp.exists():
            fp.unlink()

    print(f"Evaluation finished in {time.time() - t0:.1f}s")
    print(json.dumps(metrics, indent=2))


def _seed_genres(movies, rated: dict) -> list:
    liked = [m for m, r in rated.items() if r >= 4.0]
    if not liked:
        return []
    sub = movies[movies["movieId"].isin(liked)]
    genres = []
    for gs in sub["genre_list"]:
        genres.extend(gs)
    return genres


if __name__ == "__main__":
    main()
