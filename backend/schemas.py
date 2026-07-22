from pydantic import BaseModel, Field


class MovieOut(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str]
    avg_rating: float | None = None  # community average, from the MovieLens dataset
    rating_count: int = 0
    poster_url: str | None = None
    watched: bool = False
    user_rating: float | None = None  # this profile's own rating, if any


class MoviePage(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[MovieOut]


class RatingIn(BaseModel):
    rating: float = Field(ge=0.5, le=5.0)


class WatchedIn(BaseModel):
    watched: bool


class WatchlistIn(BaseModel):
    watchlisted: bool


class WatchedToggleOut(BaseModel):
    movie_id: int
    watched: bool


class WatchlistToggleOut(BaseModel):
    movie_id: int
    watchlisted: bool


class RecommendationOut(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str]
    poster_url: str | None = None
    score: float
    reason: str
    model: str
    source_model: str | None = None


class AnalyticsOut(BaseModel):
    movies_watched: int
    movies_rated: int
    avg_rating: float | None = None
    genre_breakdown: dict[str, int]
    rating_distribution: dict[str, int]
    decade_breakdown: dict[str, int]
    cluster_breakdown: dict[str, int]


class ModelMetricOut(BaseModel):
    model: str
    plain_label: str
    summary: str
    rmse: float | None = None
    mae: float | None = None
    precision_at_k: float | None = None
    recall_at_k: float | None = None
    hit_rate_at_k: float | None = None
    ndcg_at_k: float | None = None
    catalog_coverage: float | None = None
    intra_list_diversity: float | None = None
    accuracy_index: float | None = None
    health: str


class SystemMetricsOut(BaseModel):
    status: str
    production_readiness: str
    readiness_score: int
    generated_from: str
    k: int | None = None
    n_eval_users: int | None = None
    total_movies: int
    total_ratings: int
    total_genres: int
    dataset_min_year: int | None = None
    dataset_max_year: int | None = None
    catalog_freshness: str
    training_workflow: list[str]
    models_available: list[str]
    metrics: list[ModelMetricOut]
    notes: str | None = None
