import pandas as pd

from src.recommenders.base import Recommendation, profile_baseline
from src.recommenders.ranking import rerank_candidates


def test_profile_baseline_shrinks_small_profiles_toward_catalog_mean():
    baseline = profile_baseline({1: 5.0}, global_mean=3.0, strength=5.0)

    assert 3.0 < baseline < 5.0
    assert profile_baseline({}, global_mean=3.4) == 3.4


def test_reranker_deduplicates_candidates_and_preserves_primary_model():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "genre_list": [["Action"], ["Drama"], ["Comedy"]],
        }
    ).set_index("movieId", drop=False)
    popularity = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "weighted_score": [4.5, 4.0, 3.8],
            "count": [500, 200, 20],
        }
    )
    candidates = [
        Recommendation(1, 4.9, "primary", "collaborative"),
        Recommendation(1, 0.8, "duplicate", "content_based"),
        Recommendation(2, 4.5, "primary", "collaborative"),
        Recommendation(3, 0.9, "supplement", "content_based"),
    ]

    results = rerank_candidates(candidates, movies, popularity, n=3, primary_model="collaborative")

    assert len(results) == 3
    assert len({result.movie_id for result in results}) == 3
    assert all(result.model == "collaborative" for result in results)
