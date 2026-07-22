import type { Analytics, ModelKey, MoviePage, Recommendation, SystemMetrics } from "./types";

const configuredApiUrl = import.meta.env.VITE_API_URL as string | undefined;
const API_URL = (configuredApiUrl ?? (import.meta.env.DEV ? "http://localhost:8000" : "")).replace(/\/$/, "");

function apiUrl() {
  if (!API_URL) throw new Error("VITE_API_URL is required in production.");
  return API_URL;
}

type ApiOptions = RequestInit & { token?: string | null };

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options: ApiOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body != null) headers.set("Content-Type", "application/json");
  if (options.token) headers.set("Authorization", `Bearer ${options.token}`);

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(new DOMException("Request timed out", "TimeoutError")), 15_000);
  const abort = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) abort();
  else options.signal?.addEventListener("abort", abort, { once: true });
  const fetchOptions: ApiOptions = { ...options };
  delete fetchOptions.token;
  delete fetchOptions.signal;
  try {
    const response = await fetch(`${apiUrl()}${path}`, { ...fetchOptions, headers, signal: controller.signal });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const body = (await response.json()) as { detail?: string };
        message = body.detail ?? message;
      } catch {
        // Keep the HTTP status message.
      }
      if (response.status === 401) window.dispatchEvent(new Event("cinematch:unauthorized"));
      throw new ApiError(message, response.status);
    }
    return response.json() as Promise<T>;
  } finally {
    window.clearTimeout(timeout);
    options.signal?.removeEventListener("abort", abort);
  }
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
  return url.startsWith("/") ? `${apiUrl()}${url}` : url;
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

export function getWatched(token: string, page = 1, signal?: AbortSignal) {
  return request<MoviePage>(`/watched?page=${page}&page_size=24`, { token, signal });
}

export function getRecommendations(token: string, model: ModelKey, n = 12, signal?: AbortSignal) {
  return request<Recommendation[]>(`/recommendations?model=${model}&n=${n}`, { token, signal });
}

export function getSimilar(token: string, movieId: number, n = 8) {
  return request<Recommendation[]>(`/recommendations/similar/${movieId}?n=${n}`, { token });
}

export function getAnalytics(token: string, signal?: AbortSignal) {
  return request<Analytics>("/analytics", { token, signal });
}
