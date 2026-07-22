import numpy as np
import pandas as pd

from src import data_utils


def test_extract_year_parses_trailing_parens():
    assert data_utils._extract_year("Toy Story (1995)") == 1995


def test_extract_year_returns_none_when_missing():
    assert data_utils._extract_year("Moonlight") is None


def test_extract_year_ignores_non_trailing_parens():
    # A parenthetical that isn't a trailing year shouldn't be misread.
    assert data_utils._extract_year("Alien (Director's Cut) 1979") is None


def test_all_genres_deduplicates_and_sorts(synthetic_movies_featured):
    genres = data_utils.all_genres(synthetic_movies_featured)
    assert genres == sorted(set(genres))
    assert "Action" in genres and "Romance" in genres
    assert len(genres) == len(set(genres))


def test_popularity_table_shrinks_low_count_movies_toward_global_mean(synthetic_movies_featured, synthetic_ratings):
    pop = data_utils.popularity_table(synthetic_movies_featured, synthetic_ratings)
    global_mean = synthetic_ratings["rating"].mean()

    # Movie 1 has 3 ratings (5.0, 4.0, 2.0 -> mean ~3.67); movie 5 has 1 rating of 5.0.
    row1 = pop[pop["movieId"] == 1].iloc[0]
    row5 = pop[pop["movieId"] == 5].iloc[0]

    assert row1["count"] == 3
    assert row5["count"] == 1
    # The single-rating movie's weighted score should be pulled further toward
    # the global mean than the 3-rating movie's, even though its raw mean (5.0)
    # is higher - this is the whole point of the Bayesian shrinkage.
    assert abs(row5["weighted_score"] - global_mean) < abs(row5["mean"] - global_mean)
    assert row5["weighted_score"] < row5["mean"]


def test_popularity_table_fills_unrated_movies_with_global_mean(synthetic_movies_featured, synthetic_ratings):
    # Add a 6th movie that never appears in synthetic_ratings at all (every
    # movieId 1-5 in the base fixture has at least one rating).
    unrated_movie = pd.DataFrame(
        [{"movieId": 6, "title": "Unrated Film (2020)", "genres": "Drama", "year": 2020, "genre_list": ["Drama"]}]
    )
    movies_with_unrated = pd.concat([synthetic_movies_featured, unrated_movie], ignore_index=True)

    pop = data_utils.popularity_table(movies_with_unrated, synthetic_ratings)
    row6 = pop[pop["movieId"] == 6].iloc[0]
    assert row6["count"] == 0
    assert row6["weighted_score"] == synthetic_ratings["rating"].mean()


def test_user_item_matrix_shape_matches_unique_counts(synthetic_ratings):
    mat, user_ids, movie_ids = data_utils.user_item_matrix(synthetic_ratings)
    assert mat.shape == (len(user_ids), len(movie_ids))
    assert set(user_ids) == set(synthetic_ratings["userId"].unique())
    assert set(movie_ids) == set(synthetic_ratings["movieId"].unique())
    # Spot-check a known rating lands in the right cell.
    u_pos = list(user_ids).index(1)
    m_pos = list(movie_ids).index(1)
    assert mat[u_pos, m_pos] == 5.0


def test_load_ratings_falls_back_when_processed_ratings_are_not_committed(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    fallback = tmp_path / "ml-latest-small"
    processed.mkdir()
    fallback.mkdir()
    (processed / "movies.csv").write_text("movieId,title,genres\n1,Movie (2000),Drama\n")
    (fallback / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,1,4.5,123\n")

    monkeypatch.setattr(data_utils, "DATA_DIR", processed)
    monkeypatch.setattr(data_utils, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(data_utils, "DEFAULT_DATA_DIR", fallback)

    ratings = data_utils.load_ratings()

    assert ratings.loc[0, "movieId"] == 1
    assert ratings.loc[0, "rating"] == 4.5


def test_movie_text_corpus_includes_genres_for_every_movie(synthetic_movies_featured):
    tags = pd.DataFrame({"userId": [1], "movieId": [1], "tag": ["revenge"], "timestamp": [0]})
    corpus = data_utils.movie_text_corpus(synthetic_movies_featured, tags)
    assert len(corpus) == len(synthetic_movies_featured)
    assert "action" in corpus.loc[1].lower()
    assert "revenge" in corpus.loc[1].lower()
    # Movie with no tags shouldn't error or produce NaN.
    assert isinstance(corpus.loc[2], str)
