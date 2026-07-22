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
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.6,
        max_features=120_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus.values)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / VECTORIZER_PATH.name)
    sparse.save_npz(out_dir / MATRIX_PATH.name, matrix)
    np.save(out_dir / MOVIE_IDS_PATH.name, corpus.index.values)
    return vectorizer, matrix, corpus.index.values


def load():
    vectorizer = joblib.load(VECTORIZER_PATH)
    matrix = sparse.load_npz(MATRIX_PATH)
    movie_ids = np.load(MOVIE_IDS_PATH)
    return vectorizer, matrix, movie_ids, {int(movie_id): i for i, movie_id in enumerate(movie_ids)}


def _parts(artifacts):
    vectorizer, matrix, movie_ids = artifacts[:3]
    idx_map = artifacts[3] if len(artifacts) > 3 else {int(movie_id): i for i, movie_id in enumerate(movie_ids)}
    return vectorizer, matrix, movie_ids, idx_map


def similar_to_movie(movie_id: int, artifacts, n: int = 10, exclude_ids=None) -> list:
    _, matrix, movie_ids, idx_map = _parts(artifacts)
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
    _, matrix, movie_ids, idx_map = _parts(artifacts)
    known = {m: r for m, r in rated.items() if m in idx_map and abs(r - 3.0) >= 0.5}
    if not known:
        return []
    idxs = [idx_map[m] for m in known]
    weights = np.array([r - 3.0 for r in known.values()], dtype=np.float32)
    weights = weights / max(np.abs(weights).sum(), 1e-6)
    profile_vec = matrix[idxs].T.dot(weights).reshape(1, -1)
    profile_vec = sparse.csr_matrix(profile_vec)
    sims = cosine_similarity(profile_vec, matrix).ravel()
    exclude = set(exclude_ids or set()) | set(rated.keys())
    return _rank(sims, movie_ids, n, exclude, reason_prefix="Matches your taste profile (content)")


def _rank(sims, movie_ids, n, exclude_ids, reason_prefix):
    valid = np.asarray([int(mid) not in exclude_ids and sims[i] > 0 for i, mid in enumerate(movie_ids)])
    candidate_indexes = np.flatnonzero(valid)
    pool_size = min(len(candidate_indexes), max(n * 8, n))
    if pool_size == 0:
        return []
    pool = np.argpartition(-sims[candidate_indexes], pool_size - 1)[:pool_size]
    order = candidate_indexes[pool[np.argsort(-sims[candidate_indexes][pool])]]
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
