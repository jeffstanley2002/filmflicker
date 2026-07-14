"""Optional OMDb integration for posters/plot/director, with a SQLite cache
so the app never re-fetches the same movie twice (OMDb's free tier caps at
1,000 requests/day). Fully optional: with no API key configured, callers
get None back and the UI falls back to a clean placeholder card.
"""
import os
import sqlite3
import threading
from pathlib import Path

import requests

CACHE_DB = Path(__file__).resolve().parent.parent / "app_data" / "omdb_cache.db"
OMDB_URL = "https://www.omdbapi.com/"
IMDB_SUGGESTION_URL = "https://v2.sg.media-imdb.com/suggestion/x/{imdb_id}.json"

# fetch_metadata is called concurrently from a thread pool. SQLite's C library is safe for that
# (threadsafety level 3), but the *Python* Connection object's own
# transaction bookkeeping isn't - concurrent execute()/commit() calls on one
# Connection from different threads can interleave and raise "cannot commit
# - no transaction is active". This lock only guards the fast local read/
# write; the slow network call stays outside it, so pages still fetch
# posters in genuine parallel.
_CACHE_LOCK = threading.Lock()

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
    return os.environ.get("OMDB_API_KEY")


def get_cache_conn() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
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


def normalize_imdb_id(imdb_id) -> str:
    s = str(imdb_id)
    return s if s.startswith("tt") else f"tt{s.zfill(7)}"


def _optimize_imdb_poster(url: str | None) -> str | None:
    if url and "m.media-amazon.com" in url and url.endswith("._V1_.jpg"):
        return f"{url[:-len('._V1_.jpg')]}._V1_QL75_UX500_.jpg"
    return url


def _fetch_imdb_poster(imdb_id: str) -> str | None:
    """Use IMDb's public suggestion response when OMDb is not configured."""
    try:
        response = requests.get(IMDB_SUGGESTION_URL.format(imdb_id=imdb_id), timeout=5)
        response.raise_for_status()
        matches = response.json().get("d", [])
    except (requests.RequestException, ValueError, AttributeError):
        return None

    match = next((item for item in matches if item.get("id") == imdb_id), None)
    image = (match or {}).get("i") or {}
    return _optimize_imdb_poster(image.get("imageUrl") or image.get("imageURL"))


def fetch_metadata(imdb_id, conn: sqlite3.Connection = None) -> dict:
    """Returns dict with poster_url/plot/director/runtime keys (values may be
    None). Never raises - network/API failures just mean less metadata."""
    empty = {"poster_url": None, "plot": None, "director": None, "runtime": None}
    if not imdb_id or (isinstance(imdb_id, float) and imdb_id != imdb_id):  # NaN check
        return empty

    imdb_id = normalize_imdb_id(imdb_id)
    conn = conn or get_cache_conn()

    with _CACHE_LOCK:
        row = conn.execute(
            """SELECT poster_url, plot, director, runtime,
                      fetched_at >= datetime('now', '-1 day') AS fresh
               FROM omdb_cache WHERE imdb_id = ?""",
            (imdb_id,),
        ).fetchone()
    if row and (row[0] or row[4]):
        return {"poster_url": _optimize_imdb_poster(row[0]), "plot": row[1], "director": row[2], "runtime": row[3]}

    api_key = _get_api_key()
    if not api_key:
        result = {**empty, "poster_url": _fetch_imdb_poster(imdb_id)}
        with _CACHE_LOCK:
            conn.execute(
                "INSERT OR REPLACE INTO omdb_cache (imdb_id, poster_url, plot, director, runtime) VALUES (?,?,?,?,?)",
                (imdb_id, result["poster_url"], None, None, None),
            )
            conn.commit()
        return result

    try:
        resp = requests.get(OMDB_URL, params={"i": imdb_id, "apikey": api_key}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        # ValueError covers resp.json() failing to parse (e.g. OMDb returning
        # an HTML error/rate-limit page instead of JSON) - a network hiccup
        # here should degrade to a placeholder card, not crash the page.
        return empty

    if data.get("Response") != "True":
        poster_url = _fetch_imdb_poster(imdb_id)
        # Cache the miss too, so we don't hammer OMDb for movies it doesn't have.
        with _CACHE_LOCK:
            conn.execute(
                "INSERT OR REPLACE INTO omdb_cache (imdb_id, poster_url, plot, director, runtime) VALUES (?,?,?,?,?)",
                (imdb_id, poster_url, None, None, None),
            )
            conn.commit()
        return {**empty, "poster_url": poster_url}

    def clean(v):
        return v if v and v != "N/A" else None

    result = {
        "poster_url": clean(data.get("Poster")),
        "plot": clean(data.get("Plot")),
        "director": clean(data.get("Director")),
        "runtime": clean(data.get("Runtime")),
    }
    with _CACHE_LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO omdb_cache (imdb_id, poster_url, plot, director, runtime) VALUES (?,?,?,?,?)",
            (imdb_id, result["poster_url"], result["plot"], result["director"], result["runtime"]),
        )
        conn.commit()
    return result
