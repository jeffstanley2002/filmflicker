from fastapi import APIRouter, Depends, Query

from backend import data as data_module
from backend import db
from backend.auth import get_current_user_id
from backend.movie_mapper import movie_out
from backend.schemas import MoviePage

router = APIRouter(prefix="/watched", tags=["watched"])


@router.get("", response_model=MoviePage)
def get_watched(
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
    user_id: str = Depends(get_current_user_id),
):
    engine = db.get_connection()
    total, states = db.get_watched_page(engine, user_id, page, page_size)
    movies_df = data_module.movies_indexed()
    links = data_module.links_indexed()

    results = []
    for mid, rating in states:
        if mid not in movies_df.index:
            continue
        row = movies_df.loc[mid]
        results.append(movie_out(mid, row, links, watched=True, user_rating=rating))
    return MoviePage(total=total, page=page, page_size=page_size, results=results)
