import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException

import data as data_module
import db
from auth import get_current_user_id

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.put("/{movie_id}/not-interested", response_model=dict)
def mark_not_interested(movie_id: int, user_id: str = Depends(get_current_user_id)):
    if movie_id not in data_module.movie_ids_set():
        raise HTTPException(status_code=404, detail="Movie not found")
    db.set_not_interested(db.get_connection(), user_id, movie_id, True)
    return {"movie_id": movie_id, "not_interested": True}


@router.delete("/{movie_id}/not-interested", response_model=dict)
def clear_not_interested(movie_id: int, user_id: str = Depends(get_current_user_id)):
    db.set_not_interested(db.get_connection(), user_id, movie_id, False)
    return {"movie_id": movie_id, "not_interested": False}
