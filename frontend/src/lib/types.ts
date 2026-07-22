export type Movie = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  avg_rating: number | null;
  rating_count: number;
  poster_url: string | null;
  watched: boolean;
  user_rating: number | null;
};

export type MoviePage = {
  total: number;
  page: number;
  page_size: number;
  results: Movie[];
};

export type Recommendation = {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  poster_url: string | null;
  score: number;
  reason: string;
  model: ModelKey;
  source_model: ModelKey | null;
};

export type Analytics = {
  movies_watched: number;
  movies_rated: number;
  avg_rating: number | null;
  genre_breakdown: Record<string, number>;
  rating_distribution: Record<string, number>;
  decade_breakdown: Record<string, number>;
  cluster_breakdown: Record<string, number>;
};

export type ModelKey = "popularity" | "content_based" | "collaborative" | "clustering" | "neural";

export type SystemMetric = {
  model: ModelKey;
  plain_label: string;
  summary: string;
  rmse: number | null;
  mae: number | null;
  precision_at_k: number | null;
  recall_at_k: number | null;
  hit_rate_at_k: number | null;
  ndcg_at_k: number | null;
  catalog_coverage: number | null;
  intra_list_diversity: number | null;
  accuracy_index: number | null;
  health: string;
};

export type SystemMetrics = {
  status: string;
  production_readiness: string;
  readiness_score: number;
  generated_from: string;
  k: number | null;
  n_eval_users: number | null;
  total_movies: number;
  total_ratings: number;
  total_genres: number;
  dataset_min_year: number | null;
  dataset_max_year: number | null;
  catalog_freshness: string;
  training_workflow: string[];
  models_available: ModelKey[];
  metrics: SystemMetric[];
  notes: string | null;
};
