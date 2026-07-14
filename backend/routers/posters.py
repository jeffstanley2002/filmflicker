from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from src import omdb

router = APIRouter(prefix="/posters", tags=["posters"])


@router.get("/{imdb_id}", name="poster_image")
def poster_image(imdb_id: str):
    normalized = omdb.normalize_imdb_id(imdb_id)
    if not normalized.startswith("tt") or not normalized[2:].isdigit():
        raise HTTPException(status_code=404, detail="Poster not found")

    poster_url = omdb.fetch_metadata(normalized).get("poster_url")
    if not poster_url:
        raise HTTPException(status_code=404, detail="Poster not found")
    return RedirectResponse(
        poster_url,
        status_code=307,
        headers={"Cache-Control": "public, max-age=604800, stale-while-revalidate=86400"},
    )
