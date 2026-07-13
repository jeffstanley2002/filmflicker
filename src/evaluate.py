"""Shared evaluation metrics for rating-prediction and top-N recommenders."""
import numpy as np


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
