from src import data_utils
from src.recommenders import popularity


def test_top_trending_excludes_requested_ids(synthetic_movies_featured, synthetic_ratings):
    pop = data_utils.popularity_table(synthetic_movies_featured, synthetic_ratings)
    recs = popularity.top_trending(pop, n=10, exclude_ids={1, 2})
    ids = {r.movie_id for r in recs}
    assert 1 not in ids
    assert 2 not in ids


def test_top_trending_filters_by_genre(synthetic_movies_featured, synthetic_ratings):
    pop = data_utils.popularity_table(synthetic_movies_featured, synthetic_ratings)
    recs = popularity.top_trending(pop, n=10, genre="Romance")
    ids = {r.movie_id for r in recs}
    # Movies 2 and 5 have Romance in their genre list; 1, 3, 4 do not.
    assert ids <= {2, 5}
    assert len(ids) > 0


def test_top_trending_respects_n(synthetic_movies_featured, synthetic_ratings):
    pop = data_utils.popularity_table(synthetic_movies_featured, synthetic_ratings)
    recs = popularity.top_trending(pop, n=2)
    assert len(recs) <= 2


def test_because_you_watched_falls_back_to_trending_with_no_seed_genres(synthetic_movies_featured, synthetic_ratings):
    pop = data_utils.popularity_table(synthetic_movies_featured, synthetic_ratings)
    recs_no_seed = popularity.because_you_watched(pop, seed_genres=[], n=5)
    recs_trending = popularity.top_trending(pop, n=5)
    assert [r.movie_id for r in recs_no_seed] == [r.movie_id for r in recs_trending]


def test_because_you_watched_prioritizes_genre_overlap(synthetic_movies_featured, synthetic_ratings):
    pop = data_utils.popularity_table(synthetic_movies_featured, synthetic_ratings)
    recs = popularity.because_you_watched(pop, seed_genres=["Romance", "Comedy"], n=10)
    ids = [r.movie_id for r in recs]
    # Movie 2 (Comedy|Romance) shares both seed genres and should rank first.
    assert ids[0] == 2


def test_because_you_watched_excludes_requested_ids(synthetic_movies_featured, synthetic_ratings):
    pop = data_utils.popularity_table(synthetic_movies_featured, synthetic_ratings)
    recs = popularity.because_you_watched(pop, seed_genres=["Action"], exclude_ids={1}, n=10)
    assert 1 not in {r.movie_id for r in recs}
