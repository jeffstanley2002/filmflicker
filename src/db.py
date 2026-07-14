"""Legacy local profile storage kept for tests and offline experiments.

The production React application uses `backend/db.py`, Supabase Auth, and
the `cinematch_v2` schema. This module remains dependency-light and reads
configuration from ordinary environment variables only.
"""
import os
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "app_data" / "app.db"

_SQLITE_SCHEMA = """
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

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS watched (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    watched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(profile_id, movie_id)
);

CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
    rated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(profile_id, movie_id)
);
"""


def _resolve_database_url() -> str:
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


def _now_sql(engine: Engine) -> str:
    return "now()" if engine.dialect.name == "postgresql" else "datetime('now')"


def get_connection(database_url: str = None) -> Engine:
    url = database_url or _resolve_database_url()
    is_sqlite = url.startswith("sqlite")

    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)

    if is_sqlite:
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.execute("PRAGMA busy_timeout = 5000")
            cur.close()

    schema = _SQLITE_SCHEMA if is_sqlite else _POSTGRES_SCHEMA
    with engine.begin() as conn:
        for stmt in schema.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))

    return engine


def get_profiles(engine: Engine):
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, name FROM profiles ORDER BY name")).fetchall()
    return [{"id": r[0], "name": r[1]} for r in rows]


def get_or_create_profile(engine: Engine, name: str) -> int:
    name = name.strip()
    with engine.begin() as conn:
        row = conn.execute(text("SELECT id FROM profiles WHERE name = :name"), {"name": name}).fetchone()
        if row:
            return row[0]
        conn.execute(text("INSERT INTO profiles (name) VALUES (:name)"), {"name": name})
        # Query back rather than rely on RETURNING/lastrowid - keeps this
        # portable across SQLite versions older than 3.35 (no RETURNING)
        # that some hosts may still bundle.
        row = conn.execute(text("SELECT id FROM profiles WHERE name = :name"), {"name": name}).fetchone()
        return row[0]


def toggle_watched(engine: Engine, profile_id: int, movie_id: int) -> bool:
    """Returns True if now watched, False if just removed."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT 1 FROM watched WHERE profile_id = :pid AND movie_id = :mid"),
            {"pid": profile_id, "mid": movie_id},
        ).fetchone()
        if row:
            conn.execute(
                text("DELETE FROM watched WHERE profile_id = :pid AND movie_id = :mid"),
                {"pid": profile_id, "mid": movie_id},
            )
            return False
        conn.execute(
            text("INSERT INTO watched (profile_id, movie_id) VALUES (:pid, :mid)"),
            {"pid": profile_id, "mid": movie_id},
        )
        return True


def get_watched_ids(engine: Engine, profile_id: int) -> set:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT movie_id FROM watched WHERE profile_id = :pid"), {"pid": profile_id}
        ).fetchall()
    return {r[0] for r in rows}


def set_rating(engine: Engine, profile_id: int, movie_id: int, rating: float):
    now_fn = _now_sql(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""INSERT INTO ratings (profile_id, movie_id, rating) VALUES (:pid, :mid, :rating)
                   ON CONFLICT (profile_id, movie_id) DO UPDATE SET
                     rating = excluded.rating, rated_at = {now_fn}"""
            ),
            {"pid": profile_id, "mid": movie_id, "rating": rating},
        )
        # Rating something implies watched.
        conn.execute(
            text(
                """INSERT INTO watched (profile_id, movie_id) VALUES (:pid, :mid)
                   ON CONFLICT (profile_id, movie_id) DO NOTHING"""
            ),
            {"pid": profile_id, "mid": movie_id},
        )


def get_ratings(engine: Engine, profile_id: int) -> dict:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT movie_id, rating FROM ratings WHERE profile_id = :pid"), {"pid": profile_id}
        ).fetchall()
    return {r[0]: r[1] for r in rows}
