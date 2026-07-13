import pytest
from sqlalchemy.exc import IntegrityError

from src import db


@pytest.fixture
def conn(tmp_path):
    return db.get_connection(f"sqlite:///{tmp_path}/test.db")


def test_get_or_create_profile_is_idempotent(conn):
    id1 = db.get_or_create_profile(conn, "alice")
    id2 = db.get_or_create_profile(conn, "alice")
    assert id1 == id2
    assert len(db.get_profiles(conn)) == 1


def test_get_or_create_profile_strips_whitespace(conn):
    id1 = db.get_or_create_profile(conn, "  bob  ")
    id2 = db.get_or_create_profile(conn, "bob")
    assert id1 == id2


def test_toggle_watched_round_trip(conn):
    pid = db.get_or_create_profile(conn, "alice")
    assert db.get_watched_ids(conn, pid) == set()

    now_watched = db.toggle_watched(conn, pid, movie_id=42)
    assert now_watched is True
    assert db.get_watched_ids(conn, pid) == {42}

    now_watched = db.toggle_watched(conn, pid, movie_id=42)
    assert now_watched is False
    assert db.get_watched_ids(conn, pid) == set()


def test_set_rating_implies_watched(conn):
    pid = db.get_or_create_profile(conn, "alice")
    db.set_rating(conn, pid, movie_id=7, rating=4.5)
    assert 7 in db.get_watched_ids(conn, pid)
    assert db.get_ratings(conn, pid) == {7: 4.5}


def test_set_rating_upserts_existing_rating(conn):
    pid = db.get_or_create_profile(conn, "alice")
    db.set_rating(conn, pid, movie_id=7, rating=3.0)
    db.set_rating(conn, pid, movie_id=7, rating=5.0)
    ratings = db.get_ratings(conn, pid)
    assert ratings[7] == 5.0
    assert len(ratings) == 1  # upsert, not a duplicate row


def test_rating_out_of_range_is_rejected_at_db_level(conn):
    pid = db.get_or_create_profile(conn, "alice")
    with pytest.raises(IntegrityError):
        db.set_rating(conn, pid, movie_id=7, rating=6.0)
    with pytest.raises(IntegrityError):
        db.set_rating(conn, pid, movie_id=7, rating=0.0)


def test_profiles_are_isolated_from_each_other(conn):
    alice = db.get_or_create_profile(conn, "alice")
    bob = db.get_or_create_profile(conn, "bob")

    db.toggle_watched(conn, alice, movie_id=1)
    db.set_rating(conn, bob, movie_id=2, rating=4.0)

    assert db.get_watched_ids(conn, alice) == {1}
    assert db.get_watched_ids(conn, bob) == {2}
    assert db.get_ratings(conn, alice) == {}
    assert db.get_ratings(conn, bob) == {2: 4.0}
