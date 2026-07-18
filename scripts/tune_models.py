"""Tune the existing collaborative hybrid on a chronological validation split.

The latest interactions are reserved as an untouched outer test split. This
script only scores the earlier validation split and writes the winning config;
scripts/evaluate_models.py remains the one-shot final test.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import data_utils, evaluate
from src.recommenders import collaborative, content_based, ensemble, popularity, ranking
from src.recommenders.base import low_signal_movie_ids

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SEED = 42
K = 10


def sample_complete_user_histories(ratings: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    sizes = ratings.groupby("userId", sort=False).size()
    shuffled = sizes.sample(frac=1.0, random_state=SEED)
    selected = shuffled.index[shuffled.cumsum() <= max_rows]
    if len(selected) == 0:
        selected = shuffled.index[:1]
    return ratings[ratings["userId"].isin(selected)].reset_index(drop=True)


def validation_users(fit_ratings, validation_ratings, limit: int) -> list[int]:
    fit_counts = fit_ratings.groupby("userId").size()
    liked = validation_ratings.loc[validation_ratings["rating"] >= 4.0, "userId"].unique()
    eligible = [int(user_id) for user_id in liked if fit_counts.get(user_id, 0) >= 5]
    rng = np.random.default_rng(SEED)
    return [int(user_id) for user_id in rng.choice(eligible, min(limit, len(eligible)), replace=False)]


def objective(metrics: dict) -> float:
    score = (
        0.40 * metrics["ndcg_at_10"]
        + 0.30 * metrics["hit_rate_at_10"]
        + 0.20 * metrics["recall_at_10"]
        + 0.10 * metrics["catalog_coverage"]
    )
    if metrics["intra_list_diversity"] < 0.75:
        score -= 0.05 * (0.75 - metrics["intra_list_diversity"])
    return float(score)


def score_config(
    users,
    fit_by_user,
    validation_by_user,
    movies_indexed,
    pop_df,
    low_signal,
    content_artifacts,
    collab_artifacts,
    ranking_config,
):
    values = {"precision": [], "recall": [], "hit": [], "ndcg": [], "mrr": [], "diversity": []}
    coverage = set()
    genres = {int(mid): set(row.genre_list) for mid, row in movies_indexed.iterrows()}
    artifacts = {"content_based": content_artifacts, "collaborative": collab_artifacts}
    for user_id in users:
        profile = fit_by_user[user_id]
        relevant = {mid for mid, rating in validation_by_user[user_id].items() if rating >= 4.0}
        recs = ensemble.recommend(
            model="collaborative",
            n=K,
            rated=profile,
            watched_ids=set(profile),
            disliked_ids=set(),
            movies_df=movies_indexed,
            pop_df=pop_df,
            low_signal=low_signal,
            artifacts=artifacts,
            ranking_config=ranking_config,
        )
        ids = [rec.movie_id for rec in recs]
        coverage.update(ids)
        precision, recall = evaluate.precision_recall_at_k(ids, relevant, K)
        values["precision"].append(precision)
        values["recall"].append(recall)
        values["hit"].append(evaluate.hit_rate_at_k(ids, relevant, K))
        values["ndcg"].append(evaluate.ndcg_at_k(ids, relevant, K))
        values["mrr"].append(evaluate.reciprocal_rank_at_k(ids, relevant, K))
        values["diversity"].append(evaluate.intra_list_diversity(ids, genres))

    metrics = {
        "precision_at_10": float(np.mean(values["precision"])),
        "recall_at_10": float(np.mean(values["recall"])),
        "hit_rate_at_10": float(np.mean(values["hit"])),
        "ndcg_at_10": float(np.mean(values["ndcg"])),
        "mrr_at_10": float(np.mean(values["mrr"])),
        "intra_list_diversity": float(np.mean(values["diversity"])),
        "catalog_coverage": float(len(coverage) / len(movies_indexed)),
    }
    metrics["objective"] = objective(metrics)
    return metrics


def evaluate_candidates(name, candidates, scorer):
    results = []
    for candidate in candidates:
        started = time.time()
        metrics = scorer(candidate)
        result = {"name": candidate["name"], "params": candidate["params"], "metrics": metrics}
        results.append(result)
        print(
            f"{name} {candidate['name']}: objective={metrics['objective']:.5f}, "
            f"NDCG={metrics['ndcg_at_10']:.4f}, hit={metrics['hit_rate_at_10']:.4f}, "
            f"diversity={metrics['intra_list_diversity']:.4f} ({time.time() - started:.1f}s)",
            flush=True,
        )
    return max(results, key=lambda result: result["metrics"]["objective"]), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ratings", type=int, default=1_000_000)
    parser.add_argument("--n-validation-users", type=int, default=150)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "tuning_results.json")
    args = parser.parse_args()
    started = time.time()

    movies = data_utils.load_movies()
    ratings = data_utils.load_ratings()
    if args.max_ratings and len(ratings) > args.max_ratings:
        ratings = sample_complete_user_histories(ratings, args.max_ratings)
    print(f"Using {len(ratings):,} ratings with complete user histories.")

    outer_train, outer_test = evaluate.temporal_user_split(ratings, test_fraction=0.2, min_train=8, max_test=5)
    fit_ratings, validation_ratings = evaluate.temporal_user_split(
        outer_train, test_fraction=0.15, min_train=5, max_test=3
    )
    print(
        f"Chronological split: {len(fit_ratings):,} fit, {len(validation_ratings):,} validation, "
        f"{len(outer_test):,} untouched test ratings."
    )

    fit_by_user = {
        int(user_id): group.set_index("movieId")["rating"].to_dict()
        for user_id, group in fit_ratings.groupby("userId")
    }
    validation_by_user = {
        int(user_id): group.set_index("movieId")["rating"].to_dict()
        for user_id, group in validation_ratings.groupby("userId")
    }
    users = validation_users(fit_ratings, validation_ratings, args.n_validation_users)
    print(f"Tuning on {len(users)} fixed validation users; outer test remains unscored.")

    movies_indexed = movies.set_index("movieId")
    content_artifacts = content_based.load()
    pop_path = MODELS_DIR / "_tune_popularity.csv"
    pop_df = popularity.train_and_save(movies, fit_ratings, out_path=pop_path)
    low_signal = low_signal_movie_ids(pop_df)
    temp_paths = []

    dimension_candidates = [
        {"name": f"rank_{rank}", "params": {"n_components": rank}}
        for rank in (16, 32, 64)
    ]
    dimension_artifacts = {}
    for candidate in dimension_candidates:
        rank = candidate["params"]["n_components"]
        path = MODELS_DIR / f"_tune_svd_{rank}.joblib"
        temp_paths.append(path)
        dimension_artifacts[rank] = collaborative.train_and_save(fit_ratings, out_path=path, n_components=rank)

    def score_dimension(candidate):
        return score_config(
            users, fit_by_user, validation_by_user, movies_indexed, pop_df, low_signal,
            content_artifacts, dimension_artifacts[candidate["params"]["n_components"]], ranking.DEFAULT_CONFIG,
        )

    best_dimension, dimension_results = evaluate_candidates("dimensions", dimension_candidates, score_dimension)
    selected_artifacts = dimension_artifacts[best_dimension["params"]["n_components"]]

    regularization_candidates = [
        {"name": "responsive", "params": {"fold_in_regularization": 0.15, "profile_baseline_strength": 3.0}},
        {"name": "current", "params": {"fold_in_regularization": 0.35, "profile_baseline_strength": 5.0}},
        {"name": "stable", "params": {"fold_in_regularization": 0.75, "profile_baseline_strength": 5.0}},
        {"name": "conservative", "params": {"fold_in_regularization": 0.50, "profile_baseline_strength": 10.0}},
    ]

    def artifacts_with(params):
        artifacts = copy.copy(selected_artifacts)
        artifacts.update(params)
        return artifacts

    def score_regularization(candidate):
        return score_config(
            users, fit_by_user, validation_by_user, movies_indexed, pop_df, low_signal,
            content_artifacts, artifacts_with(candidate["params"]), ranking.DEFAULT_CONFIG,
        )

    best_regularization, regularization_results = evaluate_candidates(
        "regularization", regularization_candidates, score_regularization
    )
    selected_artifacts = artifacts_with(best_regularization["params"])

    ranking_candidates = [
        {"name": "current", "params": dict(ranking.DEFAULT_CONFIG)},
        {"name": "personalized", "params": {"personalized_weight": 0.82, "quality_weight": 0.13, "novelty_weight": 0.05, "diversity_strength": 0.12}},
        {"name": "accuracy_first", "params": {"personalized_weight": 0.85, "quality_weight": 0.15, "novelty_weight": 0.0, "diversity_strength": 0.08}},
        {"name": "discovery", "params": {"personalized_weight": 0.68, "quality_weight": 0.20, "novelty_weight": 0.12, "diversity_strength": 0.18}},
    ]

    def score_ranking(candidate):
        return score_config(
            users, fit_by_user, validation_by_user, movies_indexed, pop_df, low_signal,
            content_artifacts, selected_artifacts, candidate["params"],
        )

    best_ranking, ranking_results = evaluate_candidates("reranking", ranking_candidates, score_ranking)
    winner = {
        "n_components": best_dimension["params"]["n_components"],
        **best_regularization["params"],
        "movie_bias_strength": collaborative.MOVIE_BIAS_STRENGTH,
        "ranking": best_ranking["params"],
    }
    report = {
        "protocol": {
            "seed": SEED,
            "sampled_ratings": len(ratings),
            "fit_ratings": len(fit_ratings),
            "validation_ratings": len(validation_ratings),
            "untouched_test_ratings": len(outer_test),
            "validation_users": len(users),
            "objective": "0.40*NDCG@10 + 0.30*HitRate@10 + 0.20*Recall@10 + 0.10*coverage; diversity floor 0.75",
        },
        "rounds": {
            "dimensions": dimension_results,
            "regularization": regularization_results,
            "reranking": ranking_results,
        },
        "winner": winner,
        "winning_validation_metrics": best_ranking["metrics"],
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Winner: {json.dumps(winner, indent=2)}")
    print(f"Wrote {args.output} in {time.time() - started:.1f}s")

    for path in [pop_path, *temp_paths]:
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
