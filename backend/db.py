"""SQLAlchemy Core data layer for the authenticated React application.

The API stores application-owned rows in a dedicated `cinematch_v2` schema
and uses Supabase Auth user IDs as the tenant boundary for every query.
"""
import os
from pathlib import Path

from sqlalchemy import Engine, bindparam, create_engine, text

from backend.migrate import run_migrations


def _resolve_database_url() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Add it to backend/.env (see backend/.env.example)."
        )
    return url


_engine: Engine = None


def get_connection() -> Engine:
    global _engine
    if _engine is not None:
        return _engine
    url = _resolve_database_url()
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=300)
    run_migrations(engine)
    _engine = engine
    return engine


def toggle_watched(engine: Engine, user_id: str, movie_id: int) -> bool:
    """Returns True if now watched, False if just removed."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT 1 FROM cinematch_v2.watched WHERE user_id = :uid AND movie_id = :mid"),
            {"uid": user_id, "mid": movie_id},
        ).fetchone()
        if row:
            conn.execute(
                text("DELETE FROM cinematch_v2.watched WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            conn.execute(
                text("DELETE FROM cinematch_v2.ratings WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            return False
        conn.execute(
            text("INSERT INTO cinematch_v2.watched (user_id, movie_id) VALUES (:uid, :mid)"),
            {"uid": user_id, "mid": movie_id},
        )
        conn.execute(
            text("DELETE FROM cinematch_v2.watchlist WHERE user_id = :uid AND movie_id = :mid"),
            {"uid": user_id, "mid": movie_id},
        )
        return True


def set_watched(engine: Engine, user_id: str, movie_id: int, watched: bool) -> bool:
    """Idempotently marks a movie watched or unwatched for one user."""
    with engine.begin() as conn:
        if watched:
            conn.execute(
                text(
                    """INSERT INTO cinematch_v2.watched (user_id, movie_id) VALUES (:uid, :mid)
                       ON CONFLICT (user_id, movie_id) DO NOTHING"""
                ),
                {"uid": user_id, "mid": movie_id},
            )
            conn.execute(
                text("DELETE FROM cinematch_v2.not_interested WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            conn.execute(
                text("DELETE FROM cinematch_v2.watchlist WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            return True

        conn.execute(
            text("DELETE FROM cinematch_v2.ratings WHERE user_id = :uid AND movie_id = :mid"),
            {"uid": user_id, "mid": movie_id},
        )
        conn.execute(
            text("DELETE FROM cinematch_v2.watched WHERE user_id = :uid AND movie_id = :mid"),
            {"uid": user_id, "mid": movie_id},
        )
        return False


def get_watched_ids(engine: Engine, user_id: str) -> set:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT movie_id FROM cinematch_v2.watched WHERE user_id = :uid"), {"uid": user_id}
        ).fetchall()
    return {r[0] for r in rows}


def get_watched_page(engine: Engine, user_id: str, page: int, page_size: int) -> tuple[int, list[tuple]]:
    offset = (page - 1) * page_size
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT count(*) FROM cinematch_v2.watched WHERE user_id = :uid"),
            {"uid": user_id},
        ).scalar_one()
        rows = conn.execute(
            text(
                """SELECT w.movie_id, r.rating
                   FROM cinematch_v2.watched AS w
                   LEFT JOIN cinematch_v2.ratings AS r
                     ON r.user_id = w.user_id AND r.movie_id = w.movie_id
                   WHERE w.user_id = :uid
                   ORDER BY w.watched_at DESC, w.movie_id
                   LIMIT :limit OFFSET :offset"""
            ),
            {"uid": user_id, "limit": page_size, "offset": offset},
        ).fetchall()
    return int(total), rows


def set_rating(engine: Engine, user_id: str, movie_id: int, rating: float):
    with engine.begin() as conn:
        conn.execute(
            text(
                """INSERT INTO cinematch_v2.ratings (user_id, movie_id, rating) VALUES (:uid, :mid, :rating)
                   ON CONFLICT (user_id, movie_id) DO UPDATE SET
                     rating = excluded.rating, rated_at = now()"""
            ),
            {"uid": user_id, "mid": movie_id, "rating": rating},
        )
        conn.execute(
            text("DELETE FROM cinematch_v2.not_interested WHERE user_id = :uid AND movie_id = :mid"),
            {"uid": user_id, "mid": movie_id},
        )
        conn.execute(
            text("DELETE FROM cinematch_v2.watchlist WHERE user_id = :uid AND movie_id = :mid"),
            {"uid": user_id, "mid": movie_id},
        )
        # Rating something implies watched.
        conn.execute(
            text(
                """INSERT INTO cinematch_v2.watched (user_id, movie_id) VALUES (:uid, :mid)
                   ON CONFLICT (user_id, movie_id) DO NOTHING"""
            ),
            {"uid": user_id, "mid": movie_id},
        )


def get_ratings(engine: Engine, user_id: str) -> dict:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT movie_id, rating FROM cinematch_v2.ratings WHERE user_id = :uid"), {"uid": user_id}
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def get_profile(engine: Engine, user_id: str) -> tuple[set, dict]:
    """Load watched IDs and their optional ratings in one database query."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """SELECT w.movie_id, r.rating
                   FROM cinematch_v2.watched AS w
                   LEFT JOIN cinematch_v2.ratings AS r
                     ON r.user_id = w.user_id AND r.movie_id = w.movie_id
                   WHERE w.user_id = :uid"""
            ),
            {"uid": user_id},
        ).fetchall()
    watched_ids = {row[0] for row in rows}
    ratings = {row[0]: row[1] for row in rows if row[1] is not None}
    return watched_ids, ratings


def get_movie_states(engine: Engine, user_id: str, movie_ids: list[int]) -> tuple[set, dict]:
    """Load state only for the movies rendered on the current page."""
    if not movie_ids:
        return set(), {}
    query = text(
        """SELECT w.movie_id, r.rating
           FROM cinematch_v2.watched AS w
           LEFT JOIN cinematch_v2.ratings AS r
             ON r.user_id = w.user_id AND r.movie_id = w.movie_id
           WHERE w.user_id = :uid AND w.movie_id IN :movie_ids"""
    ).bindparams(bindparam("movie_ids", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(query, {"uid": user_id, "movie_ids": movie_ids}).fetchall()
    return {row[0] for row in rows}, {row[0]: row[1] for row in rows if row[1] is not None}


def get_not_interested_ids(engine: Engine, user_id: str) -> set:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT movie_id FROM cinematch_v2.not_interested WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchall()
    return {row[0] for row in rows}


def get_watchlist_ids(engine: Engine, user_id: str) -> set:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT movie_id FROM cinematch_v2.watchlist WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchall()
    return {row[0] for row in rows}


def get_watchlist_page(engine: Engine, user_id: str, page: int, page_size: int) -> tuple[int, list[int]]:
    offset = (page - 1) * page_size
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT count(*) FROM cinematch_v2.watchlist WHERE user_id = :uid"),
            {"uid": user_id},
        ).scalar_one()
        rows = conn.execute(
            text(
                """SELECT movie_id
                   FROM cinematch_v2.watchlist
                   WHERE user_id = :uid
                   ORDER BY added_at DESC, movie_id
                   LIMIT :limit OFFSET :offset"""
            ),
            {"uid": user_id, "limit": page_size, "offset": offset},
        ).fetchall()
    return int(total), [row[0] for row in rows]


def set_watchlist(engine: Engine, user_id: str, movie_id: int, value: bool) -> bool:
    with engine.begin() as conn:
        if value:
            conn.execute(
                text("DELETE FROM cinematch_v2.not_interested WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            conn.execute(
                text(
                    """INSERT INTO cinematch_v2.watchlist (user_id, movie_id)
                       VALUES (:uid, :mid) ON CONFLICT (user_id, movie_id) DO NOTHING"""
                ),
                {"uid": user_id, "mid": movie_id},
            )
        else:
            conn.execute(
                text("DELETE FROM cinematch_v2.watchlist WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
    return value


def set_not_interested(engine: Engine, user_id: str, movie_id: int, value: bool) -> bool:
    with engine.begin() as conn:
        if value:
            conn.execute(
                text("DELETE FROM cinematch_v2.ratings WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            conn.execute(
                text("DELETE FROM cinematch_v2.watched WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            conn.execute(
                text("DELETE FROM cinematch_v2.watchlist WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
            conn.execute(
                text(
                    """INSERT INTO cinematch_v2.not_interested (user_id, movie_id)
                       VALUES (:uid, :mid) ON CONFLICT (user_id, movie_id) DO NOTHING"""
                ),
                {"uid": user_id, "mid": movie_id},
            )
        else:
            conn.execute(
                text("DELETE FROM cinematch_v2.not_interested WHERE user_id = :uid AND movie_id = :mid"),
                {"uid": user_id, "mid": movie_id},
            )
    return value
