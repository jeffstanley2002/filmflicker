"""SQLite-backed storage for user profiles, watched movies, and ratings.

Deliberately password-free: profiles are just named buckets so a single
deployed demo can support several "users" without any auth machinery. The
DB lives outside the repo's version-controlled data (see APP_DB_PATH) since
it's per-deployment, mutable, user-generated state.
"""
import sqlite3
from pathlib import Path

APP_DB_PATH = Path(__file__).resolve().parent.parent / "app_data" / "app.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS watched (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    watched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(profile_id, movie_id)
);

CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
    rated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(profile_id, movie_id)
);
"""


def get_connection(db_path: Path = APP_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def get_profiles(conn: sqlite3.Connection):
    rows = conn.execute("SELECT id, name FROM profiles ORDER BY name").fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]


def get_or_create_profile(conn: sqlite3.Connection, name: str) -> int:
    name = name.strip()
    row = conn.execute("SELECT id FROM profiles WHERE name = ?", (name,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO profiles (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def toggle_watched(conn: sqlite3.Connection, profile_id: int, movie_id: int) -> bool:
    """Returns True if now watched, False if just removed."""
    row = conn.execute(
        "SELECT 1 FROM watched WHERE profile_id = ? AND movie_id = ?",
        (profile_id, movie_id),
    ).fetchone()
    if row:
        conn.execute(
            "DELETE FROM watched WHERE profile_id = ? AND movie_id = ?",
            (profile_id, movie_id),
        )
        conn.commit()
        return False
    conn.execute(
        "INSERT INTO watched (profile_id, movie_id) VALUES (?, ?)",
        (profile_id, movie_id),
    )
    conn.commit()
    return True


def get_watched_ids(conn: sqlite3.Connection, profile_id: int) -> set:
    rows = conn.execute(
        "SELECT movie_id FROM watched WHERE profile_id = ?", (profile_id,)
    ).fetchall()
    return {r[0] for r in rows}


def set_rating(conn: sqlite3.Connection, profile_id: int, movie_id: int, rating: float):
    conn.execute(
        """INSERT INTO ratings (profile_id, movie_id, rating) VALUES (?, ?, ?)
           ON CONFLICT(profile_id, movie_id) DO UPDATE SET
             rating = excluded.rating, rated_at = datetime('now')""",
        (profile_id, movie_id, rating),
    )
    # Rating something implies watched.
    conn.execute(
        "INSERT OR IGNORE INTO watched (profile_id, movie_id) VALUES (?, ?)",
        (profile_id, movie_id),
    )
    conn.commit()


def get_ratings(conn: sqlite3.Connection, profile_id: int) -> dict:
    rows = conn.execute(
        "SELECT movie_id, rating FROM ratings WHERE profile_id = ?", (profile_id,)
    ).fetchall()
    return {r[0]: r[1] for r in rows}
