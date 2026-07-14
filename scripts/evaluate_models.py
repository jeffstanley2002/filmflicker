"""Temporal held-out evaluation for all 5 recommenders.

Holds out each user's latest interactions, retrains rating-sensitive models,
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
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import data_utils, evaluate
from src.recommenders import clustering, collaborative, content_based, ensemble, neural, popularity
from src.recommenders.base import low_signal_movie_ids, profile_baseline
from train_models import build_embedding_fallback_weights

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
K = 10
N_EVAL_USERS = 150
MIN_TRAIN_RATINGS = 5
SEED = 42
DEFAULT_MAX_RATINGS = 2_000_000
DEFAULT_MAX_RATING_PREDICTIONS = 200_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ratings", type=int, default=DEFAULT_MAX_RATINGS)
    parser.add_argument("--max-rating-predictions", type=int, default=DEFAULT_MAX_RATING_PREDICTIONS)
    parser.add_argument("--n-eval-users", type=int, default=N_EVAL_USERS)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "metrics.json")
    args = parser.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)

    movies = data_utils.load_movies()
    ratings = data_utils.load_ratings()
    tags = data_utils.load_tags()
    original_rating_count = len(ratings)
    if args.max_ratings and len(ratings) > args.max_ratings:
        ratings = _sample_complete_user_histories(ratings, args.max_ratings)
        print(f"Sampled complete histories: {len(ratings):,}/{original_rating_count:,} ratings.")

    print("Holding out each user's latest interactions...")
    train_ratings, test_ratings = evaluate.temporal_user_split(
        ratings, test_fraction=0.2, min_train=MIN_TRAIN_RATINGS, max_test=5
    )

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

    print("Building latent taste embedding artifact from train-split SVD factors...")
    neural_weights = build_embedding_fallback_weights(collab_artifacts, train_ratings)
    neural_model = neural.NeuralRecommender(neural_weights)

    print("Loading content-based + popularity artifacts (content is rating-independent)...")
    content_artifacts = content_based.load()
    pop_df = popularity.train_and_save(movies, train_ratings, out_path=MODELS_DIR / "_eval_popularity.csv")
    low_signal = low_signal_movie_ids(pop_df)

    # --- RMSE / MAE for rating-predicting models ---
    print("Scoring RMSE/MAE...")
    global_mean_train = float(train_ratings["rating"].mean())
    collab_preds, neural_preds, actuals = [], [], []
    rating_test = test_ratings
    if args.max_rating_predictions and len(rating_test) > args.max_rating_predictions:
        rating_test = rating_test.sample(n=args.max_rating_predictions, random_state=SEED).reset_index(drop=True)
    collab_movie_idx = collab_artifacts.get("movie_idx") or {
        int(m): i for i, m in enumerate(collab_artifacts["movie_ids"])
    }
    collab_user_vec_cache: dict[int, np.ndarray] = {}
    neural_user_vec_cache: dict[int, np.ndarray] = {}
    for row in rating_test.itertuples():
        uid = int(row.userId)
        train_dict = train_by_user.get(uid, {})
        if uid not in collab_user_vec_cache:
            collab_user_vec_cache[uid] = collaborative.fold_in_new_profile(train_dict, collab_artifacts)
        movie_idx = collab_movie_idx.get(int(row.movieId))
        if movie_idx is None:
            collab_preds.append(global_mean_train)
        else:
            movie_bias = collab_artifacts.get("movie_bias")
            bias = movie_bias[movie_idx] if movie_bias is not None else 0
            baseline = profile_baseline(train_dict, float(collab_artifacts["global_mean"]))
            score = baseline + collab_artifacts["item_factors"][movie_idx].dot(collab_user_vec_cache[uid]) + bias
            collab_preds.append(float(np.clip(score, 0.5, 5.0)))

        if uid not in neural_user_vec_cache:
            neural_user_vec_cache[uid] = neural_model.implied_user_vector(train_dict)
        neural_idx = neural_model.movie_idx.get(int(row.movieId))
        if neural_idx is None:
            neural_preds.append(global_mean_train)
        else:
            item_vec = neural_model._item_tower(np.array([neural_idx], dtype=np.int64))[0]
            baseline = profile_baseline(train_dict, neural_model.global_mean)
            score = baseline + item_vec.dot(neural_user_vec_cache[uid]) + neural_model.movie_bias[neural_idx]
            neural_preds.append(float(np.clip(score, 0.5, 5.0)))
        actuals.append(row.rating)

    rating_metrics = {
        "collaborative": {"rmse": evaluate.rmse(collab_preds, actuals), "mae": evaluate.mae(collab_preds, actuals)},
        "neural": {"rmse": evaluate.rmse(neural_preds, actuals), "mae": evaluate.mae(neural_preds, actuals)},
    }

    # --- Precision@K / Recall@K for all 5 models ---
    print(f"Scoring Precision@{args.k}/Recall@{args.k} on up to {args.n_eval_users} sampled users...")
    eligible_users = [
        u for u in test_by_user
        if len(train_by_user.get(u, {})) >= MIN_TRAIN_RATINGS
        and any(r >= 4.0 for r in test_by_user[u].values())
    ]
    sample_users = rng.choice(eligible_users, size=min(args.n_eval_users, len(eligible_users)), replace=False)

    topn_scores = {m: {"precision": [], "recall": [], "hit_rate": [], "ndcg": [], "mrr": [], "diversity": [], "novelty": []} for m in
                   ["popularity", "content_based", "collaborative", "clustering", "neural"]}
    coverage_ids = {m: set() for m in topn_scores}
    genres_by_movie = {
        int(row.movieId): set(row.genre_list) for row in movies[["movieId", "genre_list"]].itertuples(index=False)
    }
    popularity_counts = pop_df.set_index("movieId")["count"].to_dict()
    total_interactions = max(float(pop_df["count"].sum()), 1.0)
    indexed_movies = movies.set_index("movieId")
    ensemble_artifacts = {
        "content_based": content_artifacts,
        "collaborative": collab_artifacts,
        "clustering": cluster_artifacts,
        "neural": neural_model,
    }

    for u in sample_users:
        train_dict = train_by_user.get(int(u), {})
        watched = set(train_dict.keys())
        exclude = watched | low_signal  # candidates to filter out; NOT the same as "watched" for cluster identity
        relevant = {m for m, r in test_by_user[int(u)].items() if r >= 4.0}
        if not relevant:
            continue

        recs = {
            model_name: ensemble.recommend(
                model=model_name,
                n=args.k,
                rated=train_dict,
                watched_ids=watched,
                disliked_ids=set(),
                movies_df=indexed_movies,
                pop_df=pop_df,
                low_signal=low_signal,
                artifacts=ensemble_artifacts,
            )
            for model_name in topn_scores
        }
        for model_name, rec_list in recs.items():
            rec_ids = [r.movie_id for r in rec_list]
            coverage_ids[model_name].update(rec_ids)
            p, r = evaluate.precision_recall_at_k(rec_ids, relevant, args.k)
            topn_scores[model_name]["precision"].append(p)
            topn_scores[model_name]["recall"].append(r)
            topn_scores[model_name]["hit_rate"].append(evaluate.hit_rate_at_k(rec_ids, relevant, args.k))
            topn_scores[model_name]["ndcg"].append(evaluate.ndcg_at_k(rec_ids, relevant, args.k))
            topn_scores[model_name]["mrr"].append(evaluate.reciprocal_rank_at_k(rec_ids, relevant, args.k))
            topn_scores[model_name]["diversity"].append(evaluate.intra_list_diversity(rec_ids, genres_by_movie))
            novelty = [
                -np.log2((float(popularity_counts.get(mid, 0)) + 1.0) / (total_interactions + len(pop_df)))
                for mid in rec_ids
            ]
            topn_scores[model_name]["novelty"].append(float(np.mean(novelty)) if novelty else 0.0)

    topn_metrics = {
        m: {
            f"precision_at_{args.k}": float(np.mean(v["precision"])) if v["precision"] else 0.0,
            f"recall_at_{args.k}": float(np.mean(v["recall"])) if v["recall"] else 0.0,
            f"hit_rate_at_{args.k}": float(np.mean(v["hit_rate"])) if v["hit_rate"] else 0.0,
            f"ndcg_at_{args.k}": float(np.mean(v["ndcg"])) if v["ndcg"] else 0.0,
            f"mrr_at_{args.k}": float(np.mean(v["mrr"])) if v["mrr"] else 0.0,
            "intra_list_diversity": float(np.mean(v["diversity"])) if v["diversity"] else 0.0,
            "mean_novelty_bits": float(np.mean(v["novelty"])) if v["novelty"] else 0.0,
            "catalog_coverage": float(len(coverage_ids[m]) / max(len(movies), 1)),
        }
        for m, v in topn_scores.items()
    }

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_utils.DATA_DIR),
        "k": args.k,
        "n_eval_users": int(len(sample_users)),
        "original_rating_count": int(original_rating_count),
        "evaluated_rating_count": int(len(ratings)),
        "train_rating_count": int(len(train_ratings)),
        "test_rating_count": int(len(test_ratings)),
        "rating_prediction_sample_size": int(len(rating_test)),
        "rating_prediction": rating_metrics,
        "top_n": topn_metrics,
        "notes": (
            "Per-user temporal holdout of the latest interactions. Evaluation samples complete user histories. "
            "Collaborative, clustering, and latent taste embedding artifacts are retrained on the train split only; "
            "content-based and popularity use rating-weighted profiles built from each test user's "
            "train-split ratings only. New or cold profiles fall back to popularity."
        ),
    }

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")

    for p in ["_eval_svd.joblib", "_eval_clustering.joblib", "_eval_popularity.csv"]:
        fp = MODELS_DIR / p
        if fp.exists():
            fp.unlink()

    print(f"Evaluation finished in {time.time() - t0:.1f}s")
    print(json.dumps(metrics, indent=2))


def _sample_complete_user_histories(ratings: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Sample users, not rows, so evaluation retains each user's chronology."""
    sizes = ratings.groupby("userId", sort=False).size()
    shuffled_users = sizes.sample(frac=1.0, random_state=SEED)
    selected = shuffled_users.index[shuffled_users.cumsum() <= max_rows]
    if len(selected) == 0:
        selected = shuffled_users.index[:1]
    return ratings[ratings["userId"].isin(selected)].reset_index(drop=True)


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
