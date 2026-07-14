import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException, Query

import data as data_module
import db
from auth import get_current_user_id
from schemas import MovieOut, MoviePage, WatchedIn, WatchedToggleOut
from utils import safe_float, safe_int, safe_str

router = APIRouter(prefix="/movies", tags=["movies"])


def _to_movie_out(row, watched_ids: set, user_ratings: dict, imdb_id) -> MovieOut:
    poster_url = data_module.poster_reference(safe_str(getattr(row, "poster_url", None)), imdb_id)
    return MovieOut(
        movie_id=row.movieId,
        title=row.title,
        year=safe_int(row.year),
        genres=row.genre_list,
        avg_rating=safe_float(getattr(row, "mean", None)),
        rating_count=safe_int(getattr(row, "count", 0)) or 0,
        poster_url=poster_url,
        watched=row.movieId in watched_ids,
        user_rating=user_ratings.get(row.movieId),
    )


@router.get("", response_model=MoviePage)
def list_movies(
    query: Optional[str] = None,
    genre: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    sort_by: str = Query("popularity", pattern="^(popularity|rating|newest|title)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
):
    links = data_module.links_indexed()
    filtered = data_module.catalog_sorted(sort_by)
    if query:
        # regex=False keeps the search box literal, so punctuation in movie
        # titles cannot turn into an invalid regular expression.
        filtered = filtered[filtered["title"].str.contains(query, case=False, na=False, regex=False)]
    if genre:
        filtered = filtered[filtered["genre_list"].apply(lambda gs: genre in gs)]
    year_col = filtered["year"]
    if year_min is not None:
        filtered = filtered[year_col.isna() | (year_col >= year_min)]
    if year_max is not None:
        filtered = filtered[year_col.isna() | (year_col <= year_max)]

    total = len(filtered)
    start = (page - 1) * page_size
    page_df = filtered.iloc[start : start + page_size]

    engine = db.get_connection()
    watched_ids, user_ratings = db.get_profile(engine, user_id)

    results = []
    for row in page_df.itertuples():
        imdb_id = links.loc[row.movieId, "imdbId"] if row.movieId in links.index else None
        results.append(_to_movie_out(row, watched_ids, user_ratings, imdb_id))

    return MoviePage(total=total, page=page, page_size=page_size, results=results)


@router.get("/genres", response_model=list[str])
def list_genres(user_id: str = Depends(get_current_user_id)):
    return data_module.all_genres_cached()


@router.post("/{movie_id}/watch", response_model=WatchedToggleOut)
def toggle_watch(movie_id: int, user_id: str = Depends(get_current_user_id)):
    if movie_id not in data_module.movie_ids_set():
        raise HTTPException(status_code=404, detail="Movie not found")
    engine = db.get_connection()
    watched = db.toggle_watched(engine, user_id, movie_id)
    return WatchedToggleOut(movie_id=movie_id, watched=watched)


@router.put("/{movie_id}/watch", response_model=WatchedToggleOut)
def set_watch(movie_id: int, body: WatchedIn, user_id: str = Depends(get_current_user_id)):
    if movie_id not in data_module.movie_ids_set():
        raise HTTPException(status_code=404, detail="Movie not found")
    engine = db.get_connection()
    watched = db.set_watched(engine, user_id, movie_id, body.watched)
    return WatchedToggleOut(movie_id=movie_id, watched=watched)
