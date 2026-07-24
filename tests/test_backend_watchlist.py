import pytest
from sqlalchemy import create_engine, event, text

pytest.importorskip("fastapi")

from backend import db
from backend.routers import recommendations, watchlist
from backend.schemas import WatchlistIn


USER_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def backend_engine():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def setup_connection(raw_connection, _connection_record):
        raw_connection.create_function("now", 0, lambda: "2026-07-24 00:00:00")

    with engine.begin() as conn:
        conn.execute(text("ATTACH DATABASE ':memory:' AS cinematch_v2"))
        conn.execute(
            text(
                """CREATE TABLE cinematch_v2.watched (
                       user_id TEXT NOT NULL,
                       movie_id INTEGER NOT NULL,
                       watched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                       UNIQUE(user_id, movie_id)
                   )"""
            )
        )
        conn.execute(
            text(
                """CREATE TABLE cinematch_v2.ratings (
                       user_id TEXT NOT NULL,
                       movie_id INTEGER NOT NULL,
                       rating REAL NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
                       rated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                       UNIQUE(user_id, movie_id)
                   )"""
            )
        )
        conn.execute(
            text(
                """CREATE TABLE cinematch_v2.not_interested (
                       user_id TEXT NOT NULL,
                       movie_id INTEGER NOT NULL,
                       created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                       UNIQUE(user_id, movie_id)
                   )"""
            )
        )
        conn.execute(
            text(
                """CREATE TABLE cinematch_v2.watchlist (
                       user_id TEXT NOT NULL,
                       movie_id INTEGER NOT NULL,
                       added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                       UNIQUE(user_id, movie_id)
                   )"""
            )
        )
    return engine


def test_watchlist_database_round_trip_and_profile_boundaries(backend_engine):
    assert db.get_watchlist_ids(backend_engine, USER_ID) == set()

    assert db.set_watchlist(backend_engine, USER_ID, 42, True) is True
    assert db.set_watchlist(backend_engine, USER_ID, 99, True) is True
    assert db.get_watchlist_ids(backend_engine, USER_ID) == {42, 99}

    total, movie_ids = db.get_watchlist_page(backend_engine, USER_ID, page=1, page_size=1)
    assert total == 2
    assert len(movie_ids) == 1
    assert db.get_watchlist_ids(backend_engine, "00000000-0000-0000-0000-000000000002") == set()

    assert db.set_watchlist(backend_engine, USER_ID, 42, False) is False
    assert db.get_watchlist_ids(backend_engine, USER_ID) == {99}


def test_watchlist_state_is_cleared_by_rating_watched_and_not_interested(backend_engine):
    db.set_watchlist(backend_engine, USER_ID, 42, True)
    db.set_rating(backend_engine, USER_ID, 42, 4.0)
    assert db.get_watchlist_ids(backend_engine, USER_ID) == set()
    assert db.get_watched_ids(backend_engine, USER_ID) == {42}

    db.set_watchlist(backend_engine, USER_ID, 43, True)
    db.set_watched(backend_engine, USER_ID, 43, True)
    assert db.get_watchlist_ids(backend_engine, USER_ID) == set()
    assert 43 in db.get_watched_ids(backend_engine, USER_ID)

    db.set_watchlist(backend_engine, USER_ID, 44, True)
    db.set_not_interested(backend_engine, USER_ID, 44, True)
    assert db.get_watchlist_ids(backend_engine, USER_ID) == set()
    assert db.get_not_interested_ids(backend_engine, USER_ID) == {44}


def test_watchlist_api_rejects_unknown_movies(monkeypatch):
    monkeypatch.setattr(watchlist.data_module, "movie_ids_set", lambda: {1, 2, 3})

    with pytest.raises(watchlist.HTTPException) as exc:
        watchlist.set_watchlist(99, WatchlistIn(watchlisted=True), user_id=USER_ID)

    assert exc.value.status_code == 404


def test_recommendations_api_passes_watchlist_as_extra_exclusions(monkeypatch):
    captured = {}

    monkeypatch.setattr(recommendations.db, "get_connection", lambda: object())
    monkeypatch.setattr(recommendations.db, "get_profile", lambda engine, user_id: ({1}, {1: 5.0}))
    monkeypatch.setattr(recommendations.db, "get_not_interested_ids", lambda engine, user_id: {2})
    monkeypatch.setattr(recommendations.db, "get_watchlist_ids", lambda engine, user_id: {3, 4})
    monkeypatch.setattr(recommendations.data_module, "low_signal", lambda: set())
    monkeypatch.setattr(recommendations.data_module, "popularity_table", lambda: object())
    monkeypatch.setattr(recommendations.data_module, "movies_indexed", lambda: object())
    monkeypatch.setattr(recommendations.data_module, "links_indexed", lambda: object())
    monkeypatch.setattr(recommendations.data_module, "content_based_artifacts", lambda: object())
    monkeypatch.setattr(recommendations, "recommendation_list", lambda recs, movies, links: recs)

    def fake_recommend(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(recommendations.ensemble, "recommend", fake_recommend)

    assert recommendations.get_recommendations(model="content_based", n=12, user_id=USER_ID) == []
    assert captured["extra_exclude_ids"] == {3, 4}
