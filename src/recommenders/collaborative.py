"""Collaborative filtering via matrix factorization (TruncatedSVD).

Ratings are mean-centered per user before factorizing, so latent factors
capture *deviation* from a user's average rather than absolute rating scale
- the same idea behind classic SVD-based recommenders (e.g. the Netflix
Prize era). New profiles (not in the original MovieLens user set) are
handled with a fold-in projection instead of retraining from scratch.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

from src import data_utils
from src.recommenders.base import MODELS_DIR, Recommendation

ARTIFACT = MODELS_DIR / "collaborative_svd.joblib"
N_COMPONENTS = 50


def train_and_save(ratings, out_path: Path = ARTIFACT, n_components: int = N_COMPONENTS):
    matrix, user_ids, movie_ids = data_utils.user_item_matrix(ratings)

    # Mean-center each user's observed ratings only (leave zeros alone).
    user_means = np.zeros(matrix.shape[0], dtype=np.float32)
    matrix = matrix.tocsr()
    for u in range(matrix.shape[0]):
        start, end = matrix.indptr[u], matrix.indptr[u + 1]
        if end > start:
            m = matrix.data[start:end].mean()
            user_means[u] = m
            matrix.data[start:end] -= m

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(matrix)  # (n_users, k)
    item_factors = svd.components_.T  # (n_items, k)

    global_mean = float(ratings["rating"].mean())
    artifacts = {
        "svd": svd,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_means": user_means,
        "global_mean": global_mean,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, out_path)
    return artifacts


def load(path: Path = ARTIFACT) -> dict:
    return joblib.load(path)


def _predict_scores(user_vec: np.ndarray, artifacts: dict, baseline: float) -> np.ndarray:
    item_factors = artifacts["item_factors"]
    return baseline + item_factors.dot(user_vec)


def fold_in_new_profile(rated: dict, artifacts: dict) -> np.ndarray:
    """Project a new profile's (movieId -> rating) dict into the latent
    space learned from the original MovieLens users, without retraining."""
    movie_ids = artifacts["movie_ids"]
    idx_map = {m: i for i, m in enumerate(movie_ids)}
    item_factors = artifacts["item_factors"]
    global_mean = artifacts["global_mean"]

    known = {m: r for m, r in rated.items() if m in idx_map}
    if not known:
        return np.zeros(item_factors.shape[1], dtype=np.float32)

    centered = np.array([r - global_mean for r in known.values()], dtype=np.float32)
    idxs = [idx_map[m] for m in known]
    # Least-squares fold-in: solve for user_vec minimizing ||item_factors[idxs] @ v - centered||
    A = item_factors[idxs]
    user_vec, *_ = np.linalg.lstsq(A, centered, rcond=None)
    return user_vec


def recommend_for_profile(rated: dict, artifacts: dict, n: int = 10, exclude_ids=None) -> list:
    movie_ids = artifacts["movie_ids"]
    global_mean = artifacts["global_mean"]
    user_vec = fold_in_new_profile(rated, artifacts)
    scores = _predict_scores(user_vec, artifacts, global_mean)

    exclude = set(exclude_ids or set()) | set(rated.keys())
    order = np.argsort(-scores)
    out = []
    for i in order:
        mid = int(movie_ids[i])
        if mid in exclude:
            continue
        pred = float(np.clip(scores[i], 0.5, 5.0))
        out.append(
            Recommendation(
                movie_id=mid,
                score=pred,
                reason=f"Predicted rating {pred:.1f}★ from similar users' patterns",
                model="collaborative",
            )
        )
        if len(out) >= n:
            break
    return out


def predict_rating(rated: dict, movie_id: int, artifacts: dict) -> float:
    movie_ids = artifacts["movie_ids"]
    idx_map = {m: i for i, m in enumerate(movie_ids)}
    if movie_id not in idx_map:
        return artifacts["global_mean"]
    user_vec = fold_in_new_profile(rated, artifacts)
    score = artifacts["global_mean"] + artifacts["item_factors"][idx_map[movie_id]].dot(user_vec)
    return float(np.clip(score, 0.5, 5.0))
