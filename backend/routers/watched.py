import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends

import data as data_module
import db
from auth import get_current_user_id
from schemas import MovieOut
from utils import safe_int, safe_str

router = APIRouter(prefix="/watched", tags=["watched"])


@router.get("", response_model=list[MovieOut])
def get_watched(user_id: str = Depends(get_current_user_id)):
    engine = db.get_connection()
    watched_ids, ratings = db.get_profile(engine, user_id)
    if not watched_ids:
        return []
    movies_df = data_module.movies_indexed()
    links = data_module.links_indexed()
    sorted_ids = sorted(watched_ids, key=lambda m: ratings.get(m, 0), reverse=True)

    results = []
    for mid in sorted_ids:
        if mid not in movies_df.index:
            continue
        row = movies_df.loc[mid]
        imdb_id = links.loc[mid, "imdbId"] if mid in links.index else None
        poster_url = data_module.poster_reference(safe_str(row.get("poster_url")), imdb_id)
        results.append(
            MovieOut(
                movie_id=mid,
                title=row["title"],
                year=safe_int(row["year"]),
                genres=row["genre_list"],
                rating_count=0,
                poster_url=poster_url,
                watched=True,
                user_rating=ratings.get(mid),
            )
        )
    return results
