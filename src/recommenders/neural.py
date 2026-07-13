"""Neural collaborative filtering, served WITHOUT a TensorFlow dependency.

The model (user embedding + movie embedding + genre features -> small MLP
-> predicted rating) is trained offline with Keras in scripts/train_models.py.
Only the trained weight matrices are exported (see NeuralWeights below); at
serve time this module reimplements the forward pass in plain numpy, which
keeps the deployed app's dependency footprint tiny and its cold start fast.

New profiles (outside the original MovieLens user set) get an "implied"
embedding: a rating-weighted average of the embeddings of movies they've
rated - the same fold-in idea used for the SVD model, just at the
embedding layer instead of the factor layer.
"""
from pathlib import Path

import numpy as np

from src.recommenders.base import MODELS_DIR, Recommendation

ARTIFACT = MODELS_DIR / "neural_weights.npz"


def relu(x):
    return np.maximum(x, 0)


class NeuralRecommender:
    def __init__(self, weights: dict):
        self.user_emb = weights["user_emb"]  # (n_users, d)
        self.movie_emb = weights["movie_emb"]  # (n_movies, d)
        self.genre_matrix = weights["genre_matrix"]  # (n_movies, n_genres)
        self.W1, self.b1 = weights["W1"], weights["b1"]
        self.W2, self.b2 = weights["W2"], weights["b2"]
        self.W3, self.b3 = weights["W3"], weights["b3"]
        self.user_ids = weights["user_ids"]
        self.movie_ids = weights["movie_ids"]
        self.movie_idx = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.user_idx = {int(u): i for i, u in enumerate(self.user_ids)}

    def _forward(self, user_vecs: np.ndarray, movie_idxs: np.ndarray) -> np.ndarray:
        x = np.concatenate([user_vecs, self.movie_emb[movie_idxs], self.genre_matrix[movie_idxs]], axis=1)
        h1 = relu(x @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        return out.ravel()

    def implied_user_vector(self, rated: dict) -> np.ndarray:
        known = {m: r for m, r in rated.items() if m in self.movie_idx}
        if not known:
            return np.zeros(self.user_emb.shape[1], dtype=np.float32)
        idxs = [self.movie_idx[m] for m in known]
        weights = np.array(list(known.values()), dtype=np.float32)
        weights = weights / weights.sum()
        return self.movie_emb[idxs].T.dot(weights)

    def predict_for_profile(self, rated: dict, movie_ids) -> np.ndarray:
        user_vec = self.implied_user_vector(rated)
        idxs = np.array([self.movie_idx[m] for m in movie_ids if m in self.movie_idx], dtype=np.int64)
        if len(idxs) == 0:
            return idxs, np.array([], dtype=np.float32)
        user_vecs = np.tile(user_vec, (len(idxs), 1))
        preds = self._forward(user_vecs, idxs)
        return idxs, np.clip(preds, 0.5, 5.0)

    def recommend_for_profile(self, rated: dict, n: int = 10, exclude_ids=None) -> list:
        exclude = set(exclude_ids or set()) | set(rated.keys())
        candidate_ids = [m for m in self.movie_ids if int(m) not in exclude]
        idxs, preds = self.predict_for_profile(rated, candidate_ids)
        order = np.argsort(-preds)
        out = []
        for o in order[: n]:
            mid = int(self.movie_ids[idxs[o]])
            out.append(
                Recommendation(
                    movie_id=mid,
                    score=float(preds[o]),
                    reason=f"Neural net predicts {preds[o]:.1f}★ for you",
                    model="neural",
                )
            )
        return out


def save_weights(weights: dict, out_path: Path = ARTIFACT):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **weights)


def load(path: Path = ARTIFACT) -> NeuralRecommender:
    data = np.load(path)
    weights = {k: data[k] for k in data.files}
    return NeuralRecommender(weights)
