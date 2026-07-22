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
from src.recommenders.base import MODELS_DIR, Recommendation, profile_baseline

ARTIFACT = MODELS_DIR / "collaborative_svd.joblib"
N_COMPONENTS = 64
FOLD_IN_REGULARIZATION = 0.15
MOVIE_BIAS_STRENGTH = 25.0
PROFILE_BASELINE_STRENGTH = 3.0


def train_and_save(
    ratings,
    out_path: Path = ARTIFACT,
    n_components: int = N_COMPONENTS,
    movie_bias_strength: float = MOVIE_BIAS_STRENGTH,
    fold_in_regularization: float = FOLD_IN_REGULARIZATION,
    profile_baseline_strength: float = PROFILE_BASELINE_STRENGTH,
):
    matrix, _, movie_ids = data_utils.user_item_matrix(ratings)
    global_mean = float(ratings["rating"].mean())
    movie_stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reindex(movie_ids)
    counts = movie_stats["count"].fillna(0).to_numpy(dtype=np.float32)
    means = movie_stats["mean"].fillna(global_mean).to_numpy(dtype=np.float32)
    movie_bias = ((means - global_mean) * (counts / (counts + movie_bias_strength))).astype(np.float32)

    # Remove both user baseline and regularized item bias before factorization.
    matrix = matrix.tocsr(copy=True)
    for u in range(matrix.shape[0]):
        start, end = matrix.indptr[u], matrix.indptr[u + 1]
        if end > start:
            m = matrix.data[start:end].mean()
            matrix.data[start:end] -= m + movie_bias[matrix.indices[start:end]]

    usable_components = max(1, min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=usable_components, n_iter=7, random_state=42)
    svd.fit(matrix)
    item_factors = svd.components_.T.astype(np.float32)  # (n_items, k)

    artifacts = {
        "item_factors": item_factors,
        "movie_ids": movie_ids,
        "movie_idx": {int(m): i for i, m in enumerate(movie_ids)},
        "movie_bias": movie_bias,
        "global_mean": global_mean,
        "n_components": usable_components,
        "movie_bias_strength": float(movie_bias_strength),
        "fold_in_regularization": float(fold_in_regularization),
        "profile_baseline_strength": float(profile_baseline_strength),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, out_path)
    return artifacts


def load(path: Path = ARTIFACT) -> dict:
    return joblib.load(path)


def _predict_scores(user_vec: np.ndarray, artifacts: dict, baseline: float) -> np.ndarray:
    item_factors = artifacts["item_factors"]
    movie_bias = artifacts.get("movie_bias", 0)
    return baseline + item_factors.dot(user_vec) + movie_bias


def fold_in_new_profile(rated: dict, artifacts: dict) -> np.ndarray:
    """Project a new profile's (movieId -> rating) dict into the latent
    space learned from the original MovieLens users, without retraining."""
    movie_ids = artifacts["movie_ids"]
    idx_map = artifacts.get("movie_idx") or {int(m): i for i, m in enumerate(movie_ids)}
    item_factors = artifacts["item_factors"]
    global_mean = float(artifacts["global_mean"])

    known = {m: r for m, r in rated.items() if m in idx_map}
    if not known:
        return np.zeros(item_factors.shape[1], dtype=np.float32)

    idxs = [idx_map[m] for m in known]
    baseline = profile_baseline(
        known, global_mean, strength=float(artifacts.get("profile_baseline_strength", PROFILE_BASELINE_STRENGTH))
    )
    movie_bias = artifacts.get("movie_bias")
    biases = movie_bias[idxs] if movie_bias is not None else 0.0
    centered = np.asarray(list(known.values()), dtype=np.float32) - baseline - biases
    # Ridge fold-in is stable even when a new profile has only a few ratings.
    A = item_factors[idxs]
    regularization = float(artifacts.get("fold_in_regularization", FOLD_IN_REGULARIZATION))
    lhs = A.T @ A + regularization * np.eye(A.shape[1], dtype=np.float32)
    return np.linalg.solve(lhs, A.T @ centered)


def recommend_for_profile(rated: dict, artifacts: dict, n: int = 10, exclude_ids=None) -> list:
    movie_ids = artifacts["movie_ids"]
    global_mean = artifacts["global_mean"]
    user_vec = fold_in_new_profile(rated, artifacts)
    baseline = profile_baseline(
        rated, global_mean, strength=float(artifacts.get("profile_baseline_strength", PROFILE_BASELINE_STRENGTH))
    )
    scores = _predict_scores(user_vec, artifacts, baseline)

    exclude = set(exclude_ids or set()) | set(rated.keys())
    valid = np.asarray([int(mid) not in exclude for mid in movie_ids], dtype=bool)
    candidate_indexes = np.flatnonzero(valid)
    pool_size = min(len(candidate_indexes), max(n * 8, n))
    if pool_size == 0:
        return []
    local = np.argpartition(-scores[candidate_indexes], pool_size - 1)[:pool_size]
    order = candidate_indexes[local[np.argsort(-scores[candidate_indexes][local])]]
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
    idx_map = artifacts.get("movie_idx") or {int(m): i for i, m in enumerate(movie_ids)}
    if movie_id not in idx_map:
        return artifacts["global_mean"]
    user_vec = fold_in_new_profile(rated, artifacts)
    movie_bias = artifacts.get("movie_bias")
    bias = movie_bias[idx_map[movie_id]] if movie_bias is not None else 0
    baseline = profile_baseline(
        rated,
        float(artifacts["global_mean"]),
        strength=float(artifacts.get("profile_baseline_strength", PROFILE_BASELINE_STRENGTH)),
    )
    score = baseline + artifacts["item_factors"][idx_map[movie_id]].dot(user_vec) + bias
    return float(np.clip(score, 0.5, 5.0))
