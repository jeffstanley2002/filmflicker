from src import evaluate
import pandas as pd


def test_rmse_zero_for_perfect_predictions():
    assert evaluate.rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    # errors of 1, -1 -> squared errors 1, 1 -> mean 1 -> sqrt 1
    assert evaluate.rmse([2, 3], [1, 4]) == 1.0


def test_mae_basic():
    assert evaluate.mae([1, 2, 3], [2, 2, 5]) == (1 + 0 + 2) / 3


def test_precision_recall_at_k_all_hits():
    p, r = evaluate.precision_recall_at_k([1, 2, 3], {1, 2, 3}, k=3)
    assert p == 1.0
    assert r == 1.0


def test_precision_recall_at_k_partial_hits():
    p, r = evaluate.precision_recall_at_k([1, 2, 3, 4], {1, 5, 6}, k=4)
    assert p == 1 / 4
    assert r == 1 / 3


def test_precision_recall_at_k_respects_k_cutoff():
    # Only the first k=2 recommendations should count, even though a hit
    # exists further down the list.
    p, r = evaluate.precision_recall_at_k([1, 2, 3], {3}, k=2)
    assert p == 0.0
    assert r == 0.0


def test_precision_recall_at_k_empty_relevant_set():
    p, r = evaluate.precision_recall_at_k([1, 2], set(), k=2)
    assert p == 0.0
    assert r == 0.0


def test_precision_recall_at_k_empty_recommendations():
    p, r = evaluate.precision_recall_at_k([], {1, 2}, k=5)
    assert p == 0.0
    assert r == 0.0


def test_ranking_metrics_reward_early_hits():
    relevant = {3, 8}
    assert evaluate.hit_rate_at_k([3, 1, 2], relevant, 3) == 1.0
    assert evaluate.reciprocal_rank_at_k([3, 1, 2], relevant, 3) == 1.0
    assert evaluate.reciprocal_rank_at_k([1, 3, 2], relevant, 3) == 0.5
    assert evaluate.ndcg_at_k([3, 1, 2], relevant, 3) > evaluate.ndcg_at_k([1, 3, 2], relevant, 3)


def test_intra_list_diversity_uses_genre_distance():
    genres = {1: {"Action"}, 2: {"Action"}, 3: {"Comedy"}}
    assert evaluate.intra_list_diversity([1, 2], genres) == 0.0
    assert evaluate.intra_list_diversity([1, 3], genres) == 1.0


def test_temporal_split_holds_out_latest_items_per_user():
    ratings = pd.DataFrame(
        {
            "userId": [1] * 8 + [2] * 4,
            "movieId": list(range(1, 9)) + list(range(20, 24)),
            "rating": [4.0] * 12,
            "timestamp": list(range(1, 9)) + list(range(1, 5)),
        }
    )
    train, test = evaluate.temporal_user_split(ratings, test_fraction=0.25, min_train=5, max_test=5)
    assert set(test.loc[test["userId"] == 1, "movieId"]) == {7, 8}
    assert set(test.loc[test["userId"] == 2, "movieId"]) == set()
    assert set(train.loc[train["userId"] == 1, "movieId"]) == set(range(1, 7))
