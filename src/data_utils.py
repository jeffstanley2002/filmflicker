"""Loading & feature-engineering helpers for the MovieLens data.

Deliberately free of frontend or web-framework imports so it can be used by
the API and by offline training/evaluation scripts.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "ml-latest-small"
DATA_DIR = Path(
    # Explicit env override is useful for one-off training runs.
    __import__("os").environ.get(
        "CINEMATCH_DATA_DIR",
        str(PROCESSED_DATA_DIR if (PROCESSED_DATA_DIR / "movies.csv").exists() else DEFAULT_DATA_DIR),
    )
)
MODELS_DIR = ROOT_DIR / "models"

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def load_movies() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "movies.csv")
    if "year" not in df.columns:
        df["year"] = df["title"].apply(_extract_year)
    else:
        parsed_year = df["title"].apply(_extract_year)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(parsed_year)
    df["genre_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    for optional in ["poster_url", "overview", "tmdb_release_date", "metadata_source"]:
        if optional not in df.columns:
            df[optional] = None
    return df


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "ratings.csv",
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
    )


def load_tags() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "tags.csv")


def load_links() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "links.csv", dtype={"imdbId": str, "tmdbId": "Int64"})


def _extract_year(title: str):
    m = _YEAR_RE.search(title)
    return int(m.group(1)) if m else None


def all_genres(movies: pd.DataFrame) -> list:
    genres = set()
    for g in movies["genre_list"]:
        genres.update(g)
    return sorted(genres)


def movie_text_corpus(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.Series:
    """One text blob per movie: genres (repeated for weight) + aggregated tags.

    Used as the input to TF-IDF for content-based filtering.
    """
    tag_agg = (
        tags.groupby("movieId")["tag"]
        .apply(lambda s: " ".join(str(t).lower().replace(" ", "_") for t in s))
        .reindex(movies["movieId"])
        .fillna("")
    )
    genre_text = movies["genre_list"].apply(
        lambda gs: " ".join(g.lower() for g in gs for _ in range(3))
    )
    corpus = (genre_text.values + " " + tag_agg.values).astype(str)
    return pd.Series(corpus, index=movies["movieId"])


def popularity_table(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Per-movie count/mean rating + a Bayesian-adjusted weighted score.

    Weighted score pulls low-count movies toward the global mean so a movie
    with two 5-star ratings doesn't outrank one with 5,000 ratings averaging
    4.3 - the same trick IMDb's "top 250" uses.
    """
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    global_mean = ratings["rating"].mean()
    min_count = stats["count"].quantile(0.60)  # only well-rated-enough movies surface as "popular"

    stats["weighted_score"] = (
        (stats["count"] / (stats["count"] + min_count)) * stats["mean"]
        + (min_count / (stats["count"] + min_count)) * global_mean
    )
    out = movies.set_index("movieId").join(stats, how="left")
    out[["count", "mean", "weighted_score"]] = out[["count", "mean", "weighted_score"]].fillna(
        {"count": 0, "mean": global_mean, "weighted_score": global_mean}
    )
    return out.reset_index()


def user_item_matrix(ratings: pd.DataFrame):
    """Returns (sparse matrix, user_id index, movie_id index) for CF models."""
    from scipy.sparse import csr_matrix

    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_pos = {u: i for i, u in enumerate(user_ids)}
    movie_pos = {m: i for i, m in enumerate(movie_ids)}

    rows = ratings["userId"].map(user_pos).values
    cols = ratings["movieId"].map(movie_pos).values
    vals = ratings["rating"].values.astype(np.float32)

    mat = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    return mat, user_ids, movie_ids
