"""Shared quality, novelty, and diversity reranking for all candidate generators."""
import numpy as np

from src.recommenders.base import Recommendation


def _minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def _genre_distance(left: set, right: set) -> float:
    union = left | right
    return 1.0 - (len(left & right) / len(union) if union else 0.0)


def rerank_candidates(candidates: list[Recommendation], movies_df, pop_df, n: int, primary_model: str, diversity_strength: float = 0.16) -> list[Recommendation]:
    if not candidates or n <= 0:
        return []

    scores_by_model = {}
    for model in {rec.model for rec in candidates}:
        model_indexes = [index for index, rec in enumerate(candidates) if rec.model == model]
        normalized = _minmax(np.asarray([candidates[index].score for index in model_indexes], dtype=np.float64))
        for index, score in zip(model_indexes, normalized):
            source_weight = 1.0 if model == primary_model else (0.94 if model == "content_based" else 0.88)
            scores_by_model[index] = float(score * source_weight)

    unique = {}
    for index, rec in enumerate(candidates):
        current = unique.get(rec.movie_id)
        if current is None or scores_by_model[index] > current[1]:
            unique[rec.movie_id] = (rec, scores_by_model[index])
    candidates = [entry[0] for entry in unique.values()]
    personalized = np.asarray([entry[1] for entry in unique.values()], dtype=np.float64)

    pop = pop_df.set_index("movieId")
    quality_values = np.asarray([
        float(pop.loc[rec.movie_id, "weighted_score"]) if rec.movie_id in pop.index else 0.0
        for rec in candidates
    ])
    quality = _minmax(quality_values)
    counts = np.asarray([
        float(pop.loc[rec.movie_id, "count"]) if rec.movie_id in pop.index else 0.0
        for rec in candidates
    ])
    novelty = 1.0 - _minmax(np.log1p(counts))
    base_scores = 0.72 * personalized + 0.23 * quality + 0.05 * novelty

    genres = {
        rec.movie_id: set(movies_df.loc[rec.movie_id, "genre_list"])
        if rec.movie_id in movies_df.index else set()
        for rec in candidates
    }
    remaining = set(range(len(candidates)))
    selected = []
    while remaining and len(selected) < n:
        best_index, best_score = None, -np.inf
        for index in remaining:
            if selected:
                similarity_penalty = max(
                    1.0 - _genre_distance(genres[candidates[index].movie_id], genres[candidates[other].movie_id])
                    for other in selected
                )
            else:
                similarity_penalty = 0.0
            score = float(base_scores[index] - diversity_strength * similarity_penalty)
            if score > best_score:
                best_index, best_score = index, score
        selected.append(best_index)
        remaining.remove(best_index)

    return [
        Recommendation(
            movie_id=candidates[index].movie_id,
            score=float(base_scores[index]),
            reason=candidates[index].reason,
            model=primary_model,
        )
        for index in selected
    ]
