"""Integration tests against the real trained artifacts committed under
models/. These exist specifically to catch the class of bug found during
manual QA: a recommender silently collapsing to the same low-signal items
for every profile, or leaking excluded/low-count movies into results.
"""
import numpy as np
import pytest

from src import data_utils
from src.recommenders import clustering, collaborative, content_based, neural, popularity
from src.recommenders.base import low_signal_movie_ids


@pytest.fixture(scope="module")
def movies():
    return data_utils.load_movies()


@pytest.fixture(scope="module")
def ratings():
    return data_utils.load_ratings()


@pytest.fixture(scope="module")
def pop_df(movies, ratings):
    return data_utils.popularity_table(movies, ratings)


@pytest.fixture(scope="module")
def low_signal(pop_df):
    return low_signal_movie_ids(pop_df)


@pytest.fixture(scope="module")
def sample_rated(pop_df):
    """A synthetic-but-realistic profile: 5 stars for the most popular
    movie, 4.5 for the second most popular, so content-based/collaborative/
    neural all have a clear, real signal to personalize from."""
    top = pop_df.sort_values("weighted_score", ascending=False).head(5)["movieId"].tolist()
    return {top[0]: 5.0, top[1]: 4.5, top[2]: 2.0}


def test_low_signal_movie_ids_excludes_sparse_items(pop_df):
    low = low_signal_movie_ids(pop_df, min_count=5)
    sparse = pop_df[pop_df["count"] < 5]["movieId"]
    well_rated = pop_df[pop_df["count"] >= 5]["movieId"]
    assert set(sparse) == low
    assert set(well_rated).isdisjoint(low)


class TestContentBased:
    @classmethod
    @pytest.fixture(scope="class")
    def artifacts(cls):
        return content_based.load()

    def test_similar_to_movie_excludes_seed(self, artifacts):
        seed_id = int(artifacts[2][0])
        recs = content_based.similar_to_movie(seed_id, artifacts, n=10)
        assert seed_id not in {r.movie_id for r in recs}
        assert len(recs) > 0

    def test_recommend_for_profile_excludes_requested_ids(self, artifacts, sample_rated, low_signal):
        exclude = set(sample_rated.keys()) | low_signal
        recs = content_based.recommend_for_profile(sample_rated, artifacts, n=10, exclude_ids=exclude)
        rec_ids = {r.movie_id for r in recs}
        assert rec_ids.isdisjoint(exclude)

    def test_recommend_for_profile_empty_when_nothing_liked(self, artifacts):
        # All ratings below the 3.5 "liked" threshold -> no taste vector to build from.
        assert content_based.recommend_for_profile({1: 2.0, 2: 3.0}, artifacts, n=10) == []


class TestCollaborative:
    @classmethod
    @pytest.fixture(scope="class")
    def artifacts(cls):
        return collaborative.load()

    def test_fold_in_empty_profile_returns_zero_vector(self, artifacts):
        vec = collaborative.fold_in_new_profile({}, artifacts)
        assert np.allclose(vec, 0)

    def test_predict_rating_is_within_valid_range(self, artifacts, sample_rated):
        movie_id = int(artifacts["movie_ids"][100])
        pred = collaborative.predict_rating(sample_rated, movie_id, artifacts)
        assert 0.5 <= pred <= 5.0

    def test_recommend_for_profile_excludes_requested_ids(self, artifacts, sample_rated, low_signal):
        exclude = set(sample_rated.keys()) | low_signal
        recs = collaborative.recommend_for_profile(sample_rated, artifacts, n=10, exclude_ids=exclude)
        rec_ids = {r.movie_id for r in recs}
        assert rec_ids.isdisjoint(exclude)
        assert all(0.5 <= r.score <= 5.0 for r in recs)


class TestClustering:
    @classmethod
    @pytest.fixture(scope="class")
    def artifacts(cls):
        return clustering.load()

    def test_cluster_of_known_movie_returns_int(self, artifacts):
        movie_id = int(artifacts["movie_ids"][0])
        assert isinstance(clustering.cluster_of(movie_id, artifacts), int)

    def test_cluster_of_unknown_movie_returns_none(self, artifacts):
        assert clustering.cluster_of(-1, artifacts) is None

    def test_recommend_for_profile_excludes_watched_and_low_signal(self, artifacts, pop_df, sample_rated, low_signal):
        watched = set(sample_rated.keys())
        exclude = watched | low_signal
        recs = clustering.recommend_for_profile(watched, artifacts, pop_df, n=10, exclude_ids=exclude)
        rec_ids = {r.movie_id for r in recs}
        assert rec_ids.isdisjoint(exclude)


class TestNeural:
    @classmethod
    @pytest.fixture(scope="class")
    def model(cls):
        return neural.load()

    def test_recommend_for_profile_never_returns_low_signal_movies(self, model, sample_rated, low_signal):
        """Direct regression test for the bug found during manual QA: before
        L2 regularization + the shared low-signal floor, the neural model
        recommended the same handful of 1-rating movies to every profile."""
        exclude = set(sample_rated.keys()) | low_signal
        recs = model.recommend_for_profile(sample_rated, n=10, exclude_ids=exclude)
        rec_ids = {r.movie_id for r in recs}
        assert rec_ids.isdisjoint(low_signal)
        assert len(recs) > 0

    def test_recommendations_vary_by_profile(self, model, pop_df):
        top = pop_df.sort_values("weighted_score", ascending=False).head(6)["movieId"].tolist()
        profile_a = {top[0]: 5.0}
        profile_b = {top[3]: 5.0}
        recs_a = {r.movie_id for r in model.recommend_for_profile(profile_a, n=10)}
        recs_b = {r.movie_id for r in model.recommend_for_profile(profile_b, n=10)}
        assert recs_a != recs_b

    def test_predict_for_profile_handles_unknown_movie_gracefully(self, model, sample_rated):
        idxs, preds = model.predict_for_profile(sample_rated, [-999])
        assert len(idxs) == 0
        assert len(preds) == 0

    def test_batched_predictions_preserve_requested_movie_count(self, model, sample_rated):
        movie_ids = [int(movie_id) for movie_id in model.movie_ids[:5000]]
        idxs, preds = model.predict_for_profile(sample_rated, movie_ids)
        assert len(idxs) == len(movie_ids)
        assert len(preds) == len(movie_ids)
        assert np.isfinite(preds).all()
