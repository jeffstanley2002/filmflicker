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
        "cluster_profiles": _build_cluster_profiles(movies, labels),
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


def _build_cluster_profiles(movies: pd.DataFrame, labels: np.ndarray) -> dict:
    profiles = {}
    labeled = movies[["genre_list", "year"]].copy()
    labeled["cluster"] = labels
    for cluster_id, group in labeled.groupby("cluster"):
        genre_counts = {}
        for genres in group["genre_list"]:
            for genre in genres:
                if genre not in {"IMAX", "(no genres listed)"}:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        top_genres = [genre for genre, _ in sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))[:2]]
        label = " + ".join(top_genres) if top_genres else "Eclectic cinema"
        years = pd.to_numeric(group["year"], errors="coerce").dropna()
        profiles[int(cluster_id)] = {
            "label": label,
            "top_genres": top_genres,
            "median_year": int(years.median()) if not years.empty else None,
            "movie_count": int(len(group)),
        }
    return profiles


def cluster_label(cluster_id: int, artifacts: dict) -> str:
    profile = artifacts.get("cluster_profiles", {}).get(cluster_id, {})
    return profile.get("label") or f"Discovery group {cluster_id + 1}"


def profile_cluster_counts(watched_ids: set, artifacts: dict) -> dict:
    idx_map = {m: i for i, m in enumerate(artifacts["movie_ids"])}
    counts = {}
    for mid in watched_ids:
        if mid in idx_map:
            c = int(artifacts["labels"][idx_map[mid]])
            counts[c] = counts.get(c, 0) + 1
    return counts


def profile_cluster_scores(profile, artifacts: dict) -> dict:
    ratings = profile if isinstance(profile, dict) else {movie_id: 3.5 for movie_id in profile}
    idx_map = {int(movie_id): i for i, movie_id in enumerate(artifacts["movie_ids"])}
    scores = {}
    for movie_id, rating in ratings.items():
        if movie_id not in idx_map or rating <= 2.5:
            continue
        cluster_id = int(artifacts["labels"][idx_map[movie_id]])
        scores[cluster_id] = scores.get(cluster_id, 0.0) + float(rating - 2.5)
    return scores


def recommend_for_profile(profile, artifacts: dict, pop_df, n: int = 10, exclude_ids=None) -> list:
    scores = profile_cluster_scores(profile, artifacts)
    if not scores:
        return []
    top_clusters = sorted(scores, key=scores.get, reverse=True)[:3]
    max_affinity = max(scores.values())
    movie_ids = artifacts["movie_ids"]
    label_by_movie = {int(movie_id): int(label) for movie_id, label in zip(movie_ids, artifacts["labels"])}
    exclude = set(exclude_ids or set()) | set(profile.keys() if isinstance(profile, dict) else profile)
    candidates = pop_df[
        pop_df["movieId"].map(label_by_movie).isin(top_clusters) & ~pop_df["movieId"].isin(exclude)
    ].copy()
    candidates["cluster_id"] = candidates["movieId"].map(label_by_movie)
    candidates["cluster_affinity"] = candidates["cluster_id"].map(scores) / max_affinity
    quality = candidates["weighted_score"].astype(float)
    quality = (quality - quality.min()) / (quality.max() - quality.min()) if quality.max() > quality.min() else 1.0
    counts = np.log1p(candidates["count"].astype(float))
    novelty = 1.0 - ((counts - counts.min()) / (counts.max() - counts.min())) if counts.max() > counts.min() else 0.0
    candidates["profile_score"] = (
        0.58 * candidates["cluster_affinity"].astype(float)
        + 0.25 * quality
        + 0.17 * novelty
    )
    candidates = candidates.sort_values("profile_score", ascending=False).head(n)
    return [Recommendation(
        movie_id=int(row.movieId),
        score=float(row.profile_score),
        reason=f"A strong match for your {cluster_label(int(row.cluster_id), artifacts)} taste",
        model="clustering",
    ) for row in candidates.itertuples()]
