import pytest
import pandas as pd

pytest.importorskip("fastapi")
from fastapi import HTTPException

from backend.routers import movies, posters
from backend.main import uptime_probe
from backend.movie_mapper import movie_out, recommendation_out
from src import omdb
from src.recommenders.base import Recommendation


def test_uptime_probe_is_lightweight():
    assert uptime_probe() == {"status": "ok"}


def test_poster_fetch_closes_owned_connection(monkeypatch):
    class Connection:
        closed = False

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(omdb, "get_cache_conn", lambda: connection)
    monkeypatch.setattr(
        omdb,
        "_fetch_metadata_with_conn",
        lambda imdb_id, conn: {"poster_url": "https://m.media-amazon.com/test.jpg"},
    )
    omdb.fetch_metadata("tt1234567")
    assert connection.closed


def test_poster_redirect_rejects_untrusted_host(monkeypatch):
    monkeypatch.setattr(omdb, "fetch_metadata", lambda imdb_id: {"poster_url": "https://attacker.example/poster.jpg"})
    with pytest.raises(HTTPException) as exc:
        posters.poster_image("tt1234567")
    assert exc.value.status_code == 404


def test_movie_filter_rejects_inverted_year_range():
    with pytest.raises(HTTPException) as exc:
        movies.list_movies(year_min=2020, year_max=2010, user_id="00000000-0000-0000-0000-000000000000")
    assert exc.value.status_code == 422


def test_movie_mapper_builds_consistent_card_response():
    row = {
        "title": "Readable Movie",
        "year": 2024,
        "genre_list": ["Drama"],
        "mean": 4.2,
        "count": 99,
        "poster_url": None,
    }
    links = pd.DataFrame({"imdbId": ["1234567"]}, index=[42])
    output = movie_out(42, row, links, watched=True, user_rating=4.5)
    assert output.movie_id == 42
    assert output.poster_url == "/posters/tt1234567"
    assert output.watched is True
    assert output.user_rating == 4.5


def test_recommendation_mapper_preserves_strategy_and_source():
    movies_index = pd.DataFrame(
        {
            "title": ["Readable Movie"],
            "year": [2024],
            "genre_list": [["Drama"]],
            "poster_url": [None],
        },
        index=[42],
    )
    links = pd.DataFrame({"imdbId": ["1234567"]}, index=[42])
    recommendation = Recommendation(
        movie_id=42,
        score=0.9,
        reason="Matches your profile",
        model="collaborative",
        source_model="content_based",
    )
    output = recommendation_out(recommendation, movies_index, links)
    assert output is not None
    assert output.model == "collaborative"
    assert output.source_model == "content_based"
