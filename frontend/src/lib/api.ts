import type { Analytics, ModelKey, MoviePage, Recommendation, SystemMetrics } from "./types";

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

type ApiOptions = RequestInit & { token?: string | null };

async function request<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (options.token) headers.set("Authorization", `Bearer ${options.token}`);

  const response = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      message = body.detail ?? message;
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function getSystemMetrics() {
  return request<SystemMetrics>("/metrics/system");
}

export function getGenres(token: string) {
  return request<string[]>("/movies/genres", { token });
}

export function getMovies(token: string, params: URLSearchParams, signal?: AbortSignal) {
  return request<MoviePage>(`/movies?${params.toString()}`, { token, signal });
}

export function resolveAssetUrl(url: string | null) {
  if (!url) return null;
  return url.startsWith("/") ? `${API_URL}${url}` : url;
}

export function setWatched(token: string, movieId: number, watched: boolean) {
  return request<{ movie_id: number; watched: boolean }>(`/movies/${movieId}/watch`, {
    method: "PUT",
    token,
    body: JSON.stringify({ watched }),
  });
}

export function setRating(token: string, movieId: number, rating: number) {
  return request<{ movie_id: number; rating: number }>(`/ratings/${movieId}`, {
    method: "PUT",
    token,
    body: JSON.stringify({ rating }),
  });
}

export function setNotInterested(token: string, movieId: number) {
  return request<{ movie_id: number; not_interested: boolean }>(`/feedback/${movieId}/not-interested`, {
    method: "PUT",
    token,
  });
}

export function getWatched(token: string) {
  return request<MoviePage["results"]>("/watched", { token });
}

export function getRecommendations(token: string, model: ModelKey, n = 12) {
  return request<Recommendation[]>(`/recommendations?model=${model}&n=${n}`, { token });
}

export function getSimilar(token: string, movieId: number, n = 8) {
  return request<Recommendation[]>(`/recommendations/similar/${movieId}?n=${n}`, { token });
}

export function getAnalytics(token: string) {
  return request<Analytics>("/analytics", { token });
}
