"""Loads MovieLens data and trained model artifacts once per process."""
import gc
import hashlib
import json
import threading

from src import data_utils, omdb
from src.recommenders import clustering, collaborative, content_based, neural, popularity
from src.recommenders.base import MODELS_DIR, low_signal_movie_ids

_cache = {}
_personalized_cache = {}
_personalized_lock = threading.Lock()
_sorted_catalog_key = None
_sorted_catalog_value = None
_sorted_catalog_lock = threading.Lock()

REQUIRED_ARTIFACTS = (
    "popularity.csv",
    "content_tfidf_matrix.npz",
    "content_tfidf_vectorizer.joblib",
    "content_movie_ids.npy",
    "collaborative_svd.joblib",
    "clustering.joblib",
    "neural_weights.npz",
    "metrics.json",
)


def _get(key, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def _get_personalized(key, loader):
    """Keep one optional personalized model resident to bound process memory."""
    with _personalized_lock:
        if key not in _personalized_cache:
            _personalized_cache.clear()
            gc.collect()
            _personalized_cache[key] = loader()
        return _personalized_cache[key]


def movies():
    return _get("movies", data_utils.load_movies)


def movies_indexed():
    return _get("movies_indexed", lambda: movies().set_index("movieId"))


def links_indexed():
    return _get("links_indexed", lambda: links().set_index("movieId"))


def catalog_sorted(sort_by: str):
    """Return one cached catalog ordering; replace it when the sort changes."""
    global _sorted_catalog_key, _sorted_catalog_value
    sort_map = {
        "popularity": ("weighted_score", False),
        "rating": ("mean", False),
        "newest": ("year", False),
        "title": ("title", True),
    }

    with _sorted_catalog_lock:
        if _sorted_catalog_key == sort_by:
            return _sorted_catalog_value
        catalog = movies().join(
            popularity_table().set_index("movieId")[["count", "mean", "weighted_score"]],
            on="movieId",
        )
        column, ascending = sort_map[sort_by]
        _sorted_catalog_value = catalog.sort_values(
            column, ascending=ascending, na_position="last"
        )
        _sorted_catalog_key = sort_by
        return _sorted_catalog_value


def catalog_manifest():
    def _load():
        path = data_utils.DATA_DIR / "catalog_manifest.json"
        return json.loads(path.read_text()) if path.exists() else None

    return _get("catalog_manifest", _load)


def links():
    return _get("links", data_utils.load_links)


def popularity_table():
    return _get("popularity", popularity.load)


def low_signal():
    return _get("low_signal", lambda: low_signal_movie_ids(popularity_table()))


def all_genres_cached():
    return _get("all_genres", lambda: data_utils.all_genres(movies()))


def movie_ids_set():
    return _get("movie_ids_set", lambda: set(movies()["movieId"]))


def content_based_artifacts():
    return _get("content_based", content_based.load)


def collaborative_artifacts():
    return _get_personalized("collaborative", collaborative.load)


def clustering_artifacts():
    return _get_personalized("clustering", clustering.load)


def neural_model():
    return _get_personalized("neural", neural.load)


def metrics():
    def _load():
        path = MODELS_DIR / "metrics.json"
        return json.loads(path.read_text()) if path.exists() else None

    return _get("metrics", _load)


def artifact_status(*, verify_checksums: bool = False) -> dict:
    """Report the actual serving bundle, optionally checking its manifest hashes."""
    manifest_path = MODELS_DIR / "artifact_manifest.json"
    missing = [name for name in REQUIRED_ARTIFACTS if not (MODELS_DIR / name).exists()]
    errors = [f"missing artifact: {name}" for name in missing]
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid artifact manifest: {exc}")
    else:
        errors.append("missing artifact manifest")

    if verify_checksums and manifest:
        checksums = manifest.get("checksums", {})
        for name in REQUIRED_ARTIFACTS:
            if name not in checksums:
                errors.append(f"missing checksum: {name}")
        for name, expected in checksums.items():
            path = MODELS_DIR / name
            if not path.exists():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                errors.append(f"checksum mismatch: {name}")
        catalog_path = data_utils.DATA_DIR / "catalog_manifest.json"
        expected_catalog = manifest.get("catalog_manifest_sha256")
        if not expected_catalog:
            errors.append("missing catalog manifest checksum")
        elif not catalog_path.exists():
            errors.append("missing catalog manifest")
        elif hashlib.sha256(catalog_path.read_bytes()).hexdigest() != expected_catalog:
            errors.append("catalog manifest checksum mismatch")
    return {
        "ready": not errors,
        "errors": errors,
        "generation_id": (manifest or {}).get("generation_id"),
        "manifest": manifest,
        "available": [name for name in REQUIRED_ARTIFACTS if (MODELS_DIR / name).exists()],
    }


def preload() -> None:
    """Warm the default request path while keeping enough memory for traffic."""
    movies_indexed()
    links_indexed()
    popularity_table()
    content_based_artifacts()
    collaborative_artifacts()
    metrics()


def poster_reference(existing_url, imdb_id) -> str | None:
    if existing_url and str(existing_url) != "nan":
        return str(existing_url)
    if not imdb_id or str(imdb_id) == "nan":
        return None
    return f"/posters/{omdb.normalize_imdb_id(imdb_id)}"
