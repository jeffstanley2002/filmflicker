import numpy as np
import pandas as pd
import pytest

from scripts.train_models import build_and_train_neural
from src.recommenders import neural
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


def test_reranker_accepts_normalized_weight_configuration():
    movies = pd.DataFrame(
        {"movieId": [1, 2], "genre_list": [["Action"], ["Drama"]]}
    ).set_index("movieId", drop=False)
    popularity = pd.DataFrame(
        {"movieId": [1, 2], "weighted_score": [3.0, 5.0], "count": [10, 1000]}
    )
    candidates = [
        Recommendation(1, 5.0, "personal", "collaborative"),
        Recommendation(2, 4.0, "quality", "collaborative"),
    ]

    personalized = rerank_candidates(
        candidates,
        movies,
        popularity,
        n=2,
        primary_model="collaborative",
        config={"personalized_weight": 1.0, "quality_weight": 0.0, "novelty_weight": 0.0},
    )
    quality = rerank_candidates(
        candidates,
        movies,
        popularity,
        n=2,
        primary_model="collaborative",
        config={"personalized_weight": 0.0, "quality_weight": 1.0, "novelty_weight": 0.0},
    )

    assert personalized[0].movie_id == 1
    assert quality[0].movie_id == 2


def test_neural_trainer_exports_independent_two_tower_artifact():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["A (2000)", "B (2000)", "C (2000)", "D (2000)"],
            "genres": ["Action|Adventure", "Action", "Drama", "Drama|Romance"],
            "genre_list": [["Action", "Adventure"], ["Action"], ["Drama"], ["Drama", "Romance"]],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movieId": [1, 2, 3, 1, 3, 4, 2, 3, 4],
            "rating": [5.0, 4.5, 1.0, 1.0, 4.5, 5.0, 4.0, 2.0, 1.5],
            "timestamp": list(range(9)),
        }
    )

    weights = build_and_train_neural(movies, ratings, epochs=1)
    model = neural.NeuralRecommender(weights)

    assert str(weights["training_mode"]) == "two_tower_neural"
    assert model.movie_emb.shape[0] == len(weights["movie_ids"])
    assert model.Wi.shape[0] == model.movie_emb.shape[1] + model.genre_matrix.shape[1]
    _, preds = model.predict_for_profile({1: 5.0, 3: 1.0}, [2, 4])
    assert len(preds) == 2


def test_neural_recommender_rejects_svd_fallback_artifact():
    weights = {
        "movie_emb": np.zeros((1, 2), dtype="float32"),
        "movie_bias": np.zeros(1, dtype="float32"),
        "genre_matrix": np.zeros((1, 0), dtype="float32"),
        "Wi": np.eye(2, dtype="float32"),
        "bi": np.zeros(2, dtype="float32"),
        "global_mean": np.float32(3.5),
        "movie_ids": np.array([1], dtype="int32"),
        "training_mode": np.array("svd_embedding_fallback"),
    }

    with pytest.raises(ValueError, match="two_tower_neural"):
        neural.NeuralRecommender(weights)
