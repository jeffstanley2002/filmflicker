"""Optional OMDb integration for posters/plot/director, with a SQLite cache
so the app never re-fetches the same movie twice (OMDb's free tier caps at
1,000 requests/day). Fully optional: with no API key configured, callers
get None back and the UI falls back to a clean placeholder card.
"""
import os
import sqlite3
from pathlib import Path

import requests

CACHE_DB = Path(__file__).resolve().parent.parent / "app_data" / "omdb_cache.db"
OMDB_URL = "https://www.omdbapi.com/"

# Deterministic placeholder color per genre so cards stay visually distinct
# even without real posters.
GENRE_COLORS = {
    "Action": "#7f1d1d", "Adventure": "#78350f", "Animation": "#3730a3",
    "Children": "#0e7490", "Comedy": "#a16207", "Crime": "#374151",
    "Documentary": "#065f46", "Drama": "#5b21b6", "Fantasy": "#6d28d9",
    "Film-Noir": "#1f2937", "Horror": "#450a0a", "Musical": "#9d174d",
    "Mystery": "#312e81", "Romance": "#9f1239", "Sci-Fi": "#164e63",
    "Thriller": "#7c2d12", "War": "#44403c", "Western": "#78350f",
    "IMAX": "#1e3a8a",
}
DEFAULT_COLOR = "#374151"


def genre_color(genres: list) -> str:
    if not genres:
        return DEFAULT_COLOR
    return GENRE_COLORS.get(genres[0], DEFAULT_COLOR)


def _get_api_key():
    # Only touch st.secrets if a secrets file actually exists - Streamlit
    # renders a visible "no secrets found" notice on first access otherwise,
    # even when the lookup is wrapped in try/except.
    secrets_paths = [
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    if any(p.exists() for p in secrets_paths):
        try:
            import streamlit as st

            key = st.secrets.get("OMDB_API_KEY")
            if key:
                return key
        except Exception:
            pass
    return os.environ.get("OMDB_API_KEY")


def get_cache_conn() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS omdb_cache (
            imdb_id TEXT PRIMARY KEY,
            poster_url TEXT,
            plot TEXT,
            director TEXT,
            runtime TEXT,
            fetched_at TEXT DEFAULT (datetime('now'))
        )"""
    )
    conn.commit()
    return conn


def _normalize_imdb_id(imdb_id) -> str:
    s = str(imdb_id)
    return s if s.startswith("tt") else f"tt{s.zfill(7)}"


def fetch_metadata(imdb_id, conn: sqlite3.Connection = None) -> dict:
    """Returns dict with poster_url/plot/director/runtime keys (values may be
    None). Never raises - network/API failures just mean less metadata."""
    empty = {"poster_url": None, "plot": None, "director": None, "runtime": None}
    if not imdb_id or (isinstance(imdb_id, float) and imdb_id != imdb_id):  # NaN check
        return empty

    imdb_id = _normalize_imdb_id(imdb_id)
    conn = conn or get_cache_conn()

    row = conn.execute(
        "SELECT poster_url, plot, director, runtime FROM omdb_cache WHERE imdb_id = ?",
        (imdb_id,),
    ).fetchone()
    if row:
        return {"poster_url": row[0], "plot": row[1], "director": row[2], "runtime": row[3]}

    api_key = _get_api_key()
    if not api_key:
        return empty

    try:
        resp = requests.get(OMDB_URL, params={"i": imdb_id, "apikey": api_key}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return empty

    if data.get("Response") != "True":
        # Cache the miss too, so we don't hammer OMDb for movies it doesn't have.
        conn.execute(
            "INSERT OR REPLACE INTO omdb_cache (imdb_id, poster_url, plot, director, runtime) VALUES (?,?,?,?,?)",
            (imdb_id, None, None, None, None),
        )
        conn.commit()
        return empty

    def clean(v):
        return v if v and v != "N/A" else None

    result = {
        "poster_url": clean(data.get("Poster")),
        "plot": clean(data.get("Plot")),
        "director": clean(data.get("Director")),
        "runtime": clean(data.get("Runtime")),
    }
    conn.execute(
        "INSERT OR REPLACE INTO omdb_cache (imdb_id, poster_url, plot, director, runtime) VALUES (?,?,?,?,?)",
        (imdb_id, result["poster_url"], result["plot"], result["director"], result["runtime"]),
    )
    conn.commit()
    return result
