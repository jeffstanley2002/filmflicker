from typing import Optional

from pydantic import BaseModel, Field


class MovieOut(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    genres: list[str]
    avg_rating: Optional[float] = None  # community average, from the MovieLens dataset
    rating_count: int = 0
    poster_url: Optional[str] = None
    watched: bool = False
    user_rating: Optional[float] = None  # this profile's own rating, if any


class MoviePage(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[MovieOut]


class RatingIn(BaseModel):
    rating: float = Field(ge=0.5, le=5.0)


class WatchedIn(BaseModel):
    watched: bool


class WatchedToggleOut(BaseModel):
    movie_id: int
    watched: bool


class RecommendationOut(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    genres: list[str]
    poster_url: Optional[str] = None
    score: float
    reason: str
    model: str


class AnalyticsOut(BaseModel):
    movies_watched: int
    movies_rated: int
    avg_rating: Optional[float] = None
    genre_breakdown: dict[str, int]
    rating_distribution: dict[str, int]
    decade_breakdown: dict[str, int]
    cluster_breakdown: dict[str, int]


class ModelMetricOut(BaseModel):
    model: str
    plain_label: str
    summary: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    hit_rate_at_k: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    catalog_coverage: Optional[float] = None
    intra_list_diversity: Optional[float] = None
    accuracy_index: Optional[float] = None
    health: str


class SystemMetricsOut(BaseModel):
    status: str
    production_readiness: str
    readiness_score: int
    generated_from: str
    k: Optional[int] = None
    n_eval_users: Optional[int] = None
    total_movies: int
    total_ratings: int
    total_genres: int
    dataset_min_year: Optional[int] = None
    dataset_max_year: Optional[int] = None
    catalog_freshness: str
    training_workflow: list[str]
    models_available: list[str]
    metrics: list[ModelMetricOut]
    notes: Optional[str] = None
