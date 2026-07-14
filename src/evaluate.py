"""Shared evaluation metrics for rating-prediction and top-N recommenders."""
import numpy as np
import pandas as pd


def rmse(preds, actuals) -> float:
    preds, actuals = np.asarray(preds, dtype=np.float64), np.asarray(actuals, dtype=np.float64)
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def mae(preds, actuals) -> float:
    preds, actuals = np.asarray(preds, dtype=np.float64), np.asarray(actuals, dtype=np.float64)
    return float(np.mean(np.abs(preds - actuals)))


def precision_recall_at_k(recommended_ids, relevant_ids: set, k: int):
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0, 0.0
    hits = len(set(top_k) & relevant_ids)
    precision = hits / len(top_k)
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return precision, recall


def ndcg_at_k(recommended_ids, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    gains = [1.0 if movie_id in relevant_ids else 0.0 for movie_id in recommended_ids[:k]]
    dcg = sum(gain / np.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal_hits = min(len(relevant_ids), k)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return float(dcg / ideal) if ideal else 0.0


def reciprocal_rank_at_k(recommended_ids, relevant_ids: set, k: int) -> float:
    for rank, movie_id in enumerate(recommended_ids[:k], start=1):
        if movie_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(recommended_ids, relevant_ids: set, k: int) -> float:
    return float(bool(set(recommended_ids[:k]) & relevant_ids))


def intra_list_diversity(recommended_ids, genres_by_movie: dict[int, set]) -> float:
    ids = [movie_id for movie_id in recommended_ids if movie_id in genres_by_movie]
    if len(ids) < 2:
        return 0.0
    distances = []
    for left_index, left_id in enumerate(ids):
        left = genres_by_movie[left_id]
        for right_id in ids[left_index + 1 :]:
            right = genres_by_movie[right_id]
            union = left | right
            distances.append(1.0 - (len(left & right) / len(union) if union else 0.0))
    return float(np.mean(distances)) if distances else 0.0


def temporal_user_split(ratings: pd.DataFrame, test_fraction: float = 0.2, min_train: int = 5, max_test: int = 5):
    """Hold out each eligible user's latest interactions without future leakage."""
    if ratings.empty:
        return ratings.copy(), ratings.copy()
    ordered = ratings.sort_values(["userId", "timestamp"], kind="stable").copy()
    positions = ordered.groupby("userId").cumcount()
    sizes = ordered.groupby("userId")["movieId"].transform("size")
    test_sizes = np.ceil(sizes * test_fraction).astype(int).clip(lower=1, upper=max_test)
    test_mask = (sizes > min_train) & (positions >= sizes - test_sizes) & (positions >= min_train)
    return ordered.loc[~test_mask].reset_index(drop=True), ordered.loc[test_mask].reset_index(drop=True)
