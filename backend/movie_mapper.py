"""Translate catalog/model records into stable API response schemas."""
from collections.abc import Mapping
from typing import Any

from backend import data
from backend.schemas import MovieOut, RecommendationOut
from backend.utils import safe_float, safe_int, safe_str


def _poster_url(movie_id: int, row: Mapping[str, Any], links) -> str | None:
    imdb_id = links.loc[movie_id, "imdbId"] if movie_id in links.index else None
    return data.poster_reference(safe_str(row.get("poster_url")), imdb_id)


def movie_out(
    movie_id: int,
    row: Mapping[str, Any],
    links,
    *,
    watched: bool,
    user_rating: float | None,
) -> MovieOut:
    """Build the common movie card response used by Browse and Watched."""
    return MovieOut(
        movie_id=movie_id,
        title=row["title"],
        year=safe_int(row.get("year")),
        genres=row["genre_list"],
        avg_rating=safe_float(row.get("mean")),
        rating_count=safe_int(row.get("count")) or 0,
        poster_url=_poster_url(movie_id, row, links),
        watched=watched,
        user_rating=user_rating,
    )


def recommendation_out(recommendation, movies, links) -> RecommendationOut | None:
    """Join a model result to catalog display data; ignore unknown movie IDs."""
    movie_id = recommendation.movie_id
    if movie_id not in movies.index:
        return None
    row = movies.loc[movie_id]
    return RecommendationOut(
        movie_id=movie_id,
        title=row["title"],
        year=safe_int(row.get("year")),
        genres=row["genre_list"],
        poster_url=_poster_url(movie_id, row, links),
        score=recommendation.score,
        reason=recommendation.reason,
        model=recommendation.model,
        source_model=recommendation.source_model,
    )


def recommendation_list(recommendations, movies, links) -> list[RecommendationOut]:
    mapped = [recommendation_out(item, movies, links) for item in recommendations]
    return [item for item in mapped if item is not None]
