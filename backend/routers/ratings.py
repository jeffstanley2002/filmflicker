from fastapi import APIRouter, Depends, HTTPException

from backend import data as data_module
from backend import db
from backend.auth import get_current_user_id
from backend.schemas import RatingIn

router = APIRouter(prefix="/ratings", tags=["ratings"])


@router.get("", response_model=dict[int, float])
def get_ratings(user_id: str = Depends(get_current_user_id)):
    engine = db.get_connection()
    return db.get_ratings(engine, user_id)


@router.put("/{movie_id}", response_model=dict)
def set_rating(movie_id: int, body: RatingIn, user_id: str = Depends(get_current_user_id)):
    if movie_id not in data_module.movie_ids_set():
        raise HTTPException(status_code=404, detail="Movie not found")
    engine = db.get_connection()
    db.set_rating(engine, user_id, movie_id, body.rating)
    return {"movie_id": movie_id, "rating": body.rating}
