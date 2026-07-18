from fastapi import APIRouter, Depends, HTTPException, Query

from backend import data as data_module
from backend import db
from backend.auth import get_current_user_id
from backend.movie_mapper import movie_out
from backend.schemas import MoviePage, WatchedIn, WatchedToggleOut

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("", response_model=MoviePage)
def list_movies(
    query: str | None = Query(None, max_length=120),
    genre: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    sort_by: str = Query("popularity", pattern="^(popularity|rating|newest|title)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
):
    if year_min is not None and year_max is not None and year_min > year_max:
        raise HTTPException(status_code=422, detail="year_min cannot be greater than year_max")
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
        filtered = filtered[year_col.notna() & (year_col >= year_min)]
    if year_max is not None:
        filtered = filtered[year_col.notna() & (year_col <= year_max)]

    total = len(filtered)
    start = (page - 1) * page_size
    page_df = filtered.iloc[start : start + page_size]

    engine = db.get_connection()
    watched_ids, user_ratings = db.get_movie_states(engine, user_id, page_df["movieId"].tolist())

    results = []
    for row in page_df.itertuples():
        record = row._asdict()
        movie_id = int(row.movieId)
        results.append(
            movie_out(
                movie_id,
                record,
                links,
                watched=movie_id in watched_ids,
                user_rating=user_ratings.get(movie_id),
            )
        )

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
