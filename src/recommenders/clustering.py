"""Exploration recommender: KMeans over movie feature vectors (multi-hot
genres + popularity signals). Groups movies into taste clusters; a profile
gets recommendations from whichever cluster their watched movies fall into
most often - good for "more of what I didn't know I liked" discovery.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src import data_utils
from src.recommenders.base import MODELS_DIR, Recommendation

ARTIFACT = MODELS_DIR / "clustering.joblib"
N_CLUSTERS = 15


def _feature_matrix(movies: pd.DataFrame, pop_df: pd.DataFrame, genres: list) -> np.ndarray:
    genre_idx = {g: i for i, g in enumerate(genres)}
    one_hot = np.zeros((len(movies), len(genres)), dtype=np.float32)
    for row_i, gs in enumerate(movies["genre_list"]):
        for g in gs:
            one_hot[row_i, genre_idx[g]] = 1.0

    pop_aligned = pop_df.set_index("movieId").reindex(movies["movieId"])
    log_count = np.log1p(pop_aligned["count"].fillna(0).values).reshape(-1, 1)
    mean_rating = pop_aligned["mean"].fillna(pop_aligned["mean"].mean()).values.reshape(-1, 1)

    return np.hstack([one_hot, log_count, mean_rating])


def train_and_save(movies, ratings, out_path: Path = ARTIFACT, n_clusters: int = N_CLUSTERS):
    pop_df = data_utils.popularity_table(movies, ratings)
    genres = data_utils.all_genres(movies)
    X = _feature_matrix(movies, pop_df, genres)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)

    artifacts = {
        "kmeans": kmeans,
        "scaler": scaler,
        "genres": genres,
        "movie_ids": movies["movieId"].values,
        "labels": labels,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, out_path)
    return artifacts


def load(path: Path = ARTIFACT) -> dict:
    return joblib.load(path)


def cluster_of(movie_id: int, artifacts: dict):
    idx_map = {m: i for i, m in enumerate(artifacts["movie_ids"])}
    if movie_id not in idx_map:
        return None
    return int(artifacts["labels"][idx_map[movie_id]])


def profile_cluster_counts(watched_ids: set, artifacts: dict) -> dict:
    idx_map = {m: i for i, m in enumerate(artifacts["movie_ids"])}
    counts = {}
    for mid in watched_ids:
        if mid in idx_map:
            c = int(artifacts["labels"][idx_map[mid]])
            counts[c] = counts.get(c, 0) + 1
    return counts


def recommend_for_profile(watched_ids: set, artifacts: dict, pop_df, n: int = 10, exclude_ids=None) -> list:
    counts = profile_cluster_counts(watched_ids, artifacts)
    if not counts:
        return []
    top_cluster = max(counts, key=counts.get)

    movie_ids = artifacts["movie_ids"]
    in_cluster = set(int(m) for m, lab in zip(movie_ids, artifacts["labels"]) if lab == top_cluster)
    exclude = set(exclude_ids or set()) | watched_ids

    candidates = pop_df[pop_df["movieId"].isin(in_cluster - exclude)]
    candidates = candidates.sort_values("weighted_score", ascending=False).head(n)
    return [
        Recommendation(
            movie_id=row.movieId,
            score=float(row.weighted_score),
            reason=f"From taste cluster #{top_cluster}, your most-watched cluster",
            model="clustering",
        )
        for row in candidates.itertuples()
    ]
