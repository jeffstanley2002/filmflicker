"""Shared types and paths for all recommender modules."""
from dataclasses import dataclass, field
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Movies with fewer ratings than this are excluded as recommendation
# *candidates* everywhere (all 5 models) - sparse items give embedding-based
# models noisy, overconfident signal (their factors fit noise, not taste),
# and low-count items are a bad user experience regardless of model.
MIN_RATING_COUNT = 5


def low_signal_movie_ids(pop_df, min_count: int = MIN_RATING_COUNT) -> set:
    return set(pop_df.loc[pop_df["count"] < min_count, "movieId"])


@dataclass
class Recommendation:
    movie_id: int
    score: float
    reason: str = ""
    model: str = ""


@dataclass
class ModelMeta:
    name: str
    description: str
    strengths: str
    weaknesses: str = field(default="")
