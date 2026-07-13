"""Content-based filtering: TF-IDF over genres + tags, cosine similarity.

Good for "more like this" and for profiles with a handful of ratings, since
it only needs the *content* of movies the profile already liked - no other
users required.
"""
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import data_utils
from src.recommenders.base import MODELS_DIR, Recommendation

VECTORIZER_PATH = MODELS_DIR / "content_tfidf_vectorizer.joblib"
MATRIX_PATH = MODELS_DIR / "content_tfidf_matrix.npz"
MOVIE_IDS_PATH = MODELS_DIR / "content_movie_ids.npy"


def train_and_save(movies, tags, out_dir: Path = MODELS_DIR):
    corpus = data_utils.movie_text_corpus(movies, tags)
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.6)
    matrix = vectorizer.fit_transform(corpus.values)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, matrix)
    np.save(MOVIE_IDS_PATH, corpus.index.values)
    return vectorizer, matrix, corpus.index.values


def load():
    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = sparse.load_npz(MATRIX_PATH)
    movie_ids = np.load(MOVIE_IDS_PATH)
    return vectorizer, matrix, movie_ids


def similar_to_movie(movie_id: int, artifacts, n: int = 10, exclude_ids=None) -> list:
    _, matrix, movie_ids = artifacts
    idx_map = {m: i for i, m in enumerate(movie_ids)}
    if movie_id not in idx_map:
        return []
    idx = idx_map[movie_id]
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    # A movie is trivially "similar" to itself (cosine similarity 1.0, the
    # max possible) - always exclude the seed regardless of what the caller
    # passes, rather than relying on every call site to remember to.
    exclude = set(exclude_ids or set()) | {movie_id}
    return _rank(sims, movie_ids, n, exclude, reason_prefix="Similar content/genres")


def recommend_for_profile(rated: dict, artifacts, n: int = 10, exclude_ids=None) -> list:
    """Profile taste vector = rating-weighted mean of TF-IDF rows for movies
    they rated >= 3.5 (liked). Falls back to empty if nothing liked yet."""
    _, matrix, movie_ids = artifacts
    idx_map = {m: i for i, m in enumerate(movie_ids)}
    liked = {m: r for m, r in rated.items() if r >= 3.5 and m in idx_map}
    if not liked:
        return []
    idxs = [idx_map[m] for m in liked]
    weights = np.array(list(liked.values()), dtype=np.float32)
    weights = weights / weights.sum()
    profile_vec = matrix[idxs].T.dot(weights).reshape(1, -1)
    profile_vec = sparse.csr_matrix(profile_vec)
    sims = cosine_similarity(profile_vec, matrix).ravel()
    exclude = set(exclude_ids or set()) | set(rated.keys())
    return _rank(sims, movie_ids, n, exclude, reason_prefix="Matches your taste profile (content)")


def _rank(sims, movie_ids, n, exclude_ids, reason_prefix):
    order = np.argsort(-sims)
    out = []
    for i in order:
        mid = int(movie_ids[i])
        if mid in exclude_ids or sims[i] <= 0:
            continue
        out.append(
            Recommendation(movie_id=mid, score=float(sims[i]), reason=reason_prefix, model="content_based")
        )
        if len(out) >= n:
            break
    return out
