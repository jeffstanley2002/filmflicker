"""Loads MovieLens data and trained model artifacts once per process."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_utils, omdb
from src.recommenders import clustering, collaborative, content_based, neural, popularity
from src.recommenders.base import MODELS_DIR, low_signal_movie_ids

_cache = {}


def _get(key, loader):
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]


def movies():
    return _get("movies", data_utils.load_movies)


def movies_indexed():
    return _get("movies_indexed", lambda: movies().set_index("movieId"))


def links_indexed():
    return _get("links_indexed", lambda: links().set_index("movieId"))


def catalog_sorted(sort_by: str):
    """Return a pre-joined, pre-sorted catalog for fast request-time filtering."""
    sort_map = {
        "popularity": ("weighted_score", False),
        "rating": ("mean", False),
        "newest": ("year", False),
        "title": ("title", True),
    }

    def _load():
        catalog = movies().join(
            popularity_table().set_index("movieId")[["count", "mean", "weighted_score"]],
            on="movieId",
        )
        column, ascending = sort_map[sort_by]
        return catalog.sort_values(column, ascending=ascending, na_position="last")

    return _get(f"catalog_sorted_{sort_by}", _load)


def catalog_manifest():
    def _load():
        path = data_utils.DATA_DIR / "catalog_manifest.json"
        return json.loads(path.read_text()) if path.exists() else None

    return _get("catalog_manifest", _load)


def ratings():
    return _get("ratings", data_utils.load_ratings)


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
    return _get("collaborative", collaborative.load)


def clustering_artifacts():
    return _get("clustering", clustering.load)


def neural_model():
    return _get("neural", neural.load)


def metrics():
    def _load():
        path = MODELS_DIR / "metrics.json"
        return json.loads(path.read_text()) if path.exists() else None

    return _get("metrics", _load)


def poster_reference(existing_url, imdb_id) -> str | None:
    if existing_url and str(existing_url) != "nan":
        return str(existing_url)
    if not imdb_id or str(imdb_id) == "nan":
        return None
    return f"/posters/{omdb.normalize_imdb_id(imdb_id)}"
