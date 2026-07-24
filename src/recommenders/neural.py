"""Two-tower neural recommender, served without a training dependency.

The offline trainer learns a genuine two-tower model from ratings: a user
tower and an item tower (movie embedding + genre features) project into a
shared latent space, and the prediction is dot(user_vec, item_vec) plus item
and user biases. Only the item tower is exported. Production profiles are
cold-start fold-ins, never MovieLens users with a trained user embedding.

New profiles get an "implied" vector: a rating-weighted (mean-centered)
average of the *item tower's output* for movies they've rated - directly
in the shared latent space, which is the principled two-tower version of
the same fold-in idea used for the SVD model.
"""
from pathlib import Path

import numpy as np

from src.recommenders.base import MODELS_DIR, Recommendation, profile_baseline

ARTIFACT = MODELS_DIR / "neural_weights.npz"
# Keep temporary matrix multiplications small enough for 512 MB deployments.
INFERENCE_BATCH_SIZE = 1024


def relu(x):
    return np.maximum(x, 0)


class NeuralRecommender:
    def __init__(self, weights: dict):
        self.training_mode = str(np.asarray(weights.get("training_mode", "missing")).item())
        if self.training_mode != "two_tower_neural":
            raise ValueError(
                "neural_weights.npz must contain an independently trained "
                f"two_tower_neural artifact, found {self.training_mode!r}"
            )
        self.movie_emb = weights["movie_emb"]  # (n_movies, embed_dim)
        self.movie_bias = weights["movie_bias"]  # (n_movies,)
        self.genre_matrix = weights["genre_matrix"]  # (n_movies, n_genres)
        self.Wi, self.bi = weights["Wi"], weights["bi"]  # item tower: Dense(embed_dim+n_genres -> latent_dim)
        self.global_mean = float(weights["global_mean"])
        self.profile_baseline_strength = float(weights.get("profile_baseline_strength", 5.0))
        self.movie_ids = weights["movie_ids"]
        self.movie_idx = {int(m): i for i, m in enumerate(self.movie_ids)}

    def _item_tower(self, movie_idxs: np.ndarray) -> np.ndarray:
        x = np.concatenate([self.movie_emb[movie_idxs], self.genre_matrix[movie_idxs]], axis=1)
        return relu(x @ self.Wi + self.bi)  # (n, latent_dim)

    def implied_user_vector(self, rated: dict) -> np.ndarray:
        known = {m: r for m, r in rated.items() if m in self.movie_idx}
        if not known:
            return np.zeros(self.Wi.shape[1], dtype=np.float32)
        idxs = np.array([self.movie_idx[m] for m in known], dtype=np.int64)
        item_vecs = self._item_tower(idxs)
        # Mean-centered (not sum-normalized): weights can be negative for
        # disliked movies, and normalizing by their sum would blow up or
        # flip sign when positive/negative ratings nearly cancel out.
        baseline = profile_baseline(known, self.global_mean, strength=self.profile_baseline_strength)
        weights = np.array([r - baseline for r in known.values()], dtype=np.float32)
        return (item_vecs * weights[:, None]).mean(axis=0)

    def predict_for_profile(self, rated: dict, movie_ids) -> np.ndarray:
        idxs = np.array([self.movie_idx[m] for m in movie_ids if m in self.movie_idx], dtype=np.int64)
        if len(idxs) == 0:
            return idxs, np.array([], dtype=np.float32)
        user_vec = self.implied_user_vector(rated)
        baseline = profile_baseline(rated, self.global_mean, strength=self.profile_baseline_strength)
        preds = np.empty(len(idxs), dtype=np.float32)
        for start in range(0, len(idxs), INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, len(idxs))
            batch_indexes = idxs[start:stop]
            item_vectors = self._item_tower(batch_indexes)
            preds[start:stop] = (
                baseline + item_vectors.dot(user_vec) + self.movie_bias[batch_indexes]
            )
        return idxs, np.clip(preds, 0.5, 5.0)

    def recommend_for_profile(self, rated: dict, n: int = 10, exclude_ids=None) -> list:
        exclude = set(exclude_ids or set()) | set(rated.keys())
        candidate_ids = [m for m in self.movie_ids if int(m) not in exclude]
        idxs, preds = self.predict_for_profile(rated, candidate_ids)
        pool_size = min(len(preds), max(n * 8, n))
        if pool_size == 0:
            return []
        pool = np.argpartition(-preds, pool_size - 1)[:pool_size]
        order = pool[np.argsort(-preds[pool])]
        out = []
        for o in order[: n]:
            mid = int(self.movie_ids[idxs[o]])
            out.append(
                Recommendation(
                    movie_id=mid,
                    score=float(preds[o]),
                    reason=f"Two-tower taste model predicts {preds[o]:.1f}★ for you",
                    model="neural",
                )
            )
        return out


def save_weights(weights: dict, out_path: Path = ARTIFACT):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **weights)


def load(path: Path = ARTIFACT) -> NeuralRecommender:
    with np.load(path) as data:
        weights = {key: data[key] for key in data.files}
    return NeuralRecommender(weights)
