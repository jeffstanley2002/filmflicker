from src import evaluate


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
