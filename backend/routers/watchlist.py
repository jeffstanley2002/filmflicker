from fastapi import APIRouter, Depends, HTTPException, Query

from backend import data as data_module
from backend import db
from backend.auth import get_current_user_id
from backend.movie_mapper import movie_out
from backend.schemas import MoviePage, WatchlistIn, WatchlistToggleOut

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.get("", response_model=MoviePage)
def get_watchlist(
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
):
    engine = db.get_connection()
    total, movie_ids = db.get_watchlist_page(engine, user_id, page, page_size)
    movies_df = data_module.movies_indexed()
    links = data_module.links_indexed()

    results = []
    for movie_id in movie_ids:
        if movie_id not in movies_df.index:
            continue
        row = movies_df.loc[movie_id]
        results.append(movie_out(movie_id, row, links, watched=False, user_rating=None))
    return MoviePage(total=total, page=page, page_size=page_size, results=results)


@router.put("/{movie_id}", response_model=WatchlistToggleOut)
def set_watchlist(movie_id: int, body: WatchlistIn, user_id: str = Depends(get_current_user_id)):
    if movie_id not in data_module.movie_ids_set():
        raise HTTPException(status_code=404, detail="Movie not found")
    watchlisted = db.set_watchlist(db.get_connection(), user_id, movie_id, body.watchlisted)
    return WatchlistToggleOut(movie_id=movie_id, watchlisted=watchlisted)
