"""Shared types and paths for all recommender modules."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Movies with fewer ratings than this are excluded as recommendation
# *candidates* everywhere (all 5 models) - sparse items give embedding-based
# models noisy, overconfident signal (their factors fit noise, not taste),
# and low-count items are a bad user experience regardless of model.
MIN_RATING_COUNT = 5


def low_signal_movie_ids(pop_df, min_count: int = MIN_RATING_COUNT) -> set:
    return set(pop_df.loc[pop_df["count"] < min_count, "movieId"])


def profile_baseline(rated: dict, global_mean: float, strength: float = 5.0) -> float:
    """Shrink a small profile's mean toward the catalog mean."""
    if not rated:
        return float(global_mean)
    values = np.asarray(list(rated.values()), dtype=np.float32)
    return float((values.sum() + strength * global_mean) / (len(values) + strength))


@dataclass
class Recommendation:
    movie_id: int
    score: float
    reason: str = ""
    model: str = ""
    source_model: str | None = None


@dataclass
class ModelMeta:
    name: str
    description: str
    strengths: str
    weaknesses: str = field(default="")
