import { Database, Radar, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { ModelSelector } from "../components/ModelSelector";
import { MovieCard } from "../components/MovieCard";
import { MovieGridSkeleton } from "../components/MovieGridSkeleton";
import { MODEL_COPY } from "../lib/models";
import { getRecommendations, setNotInterested, setWatchlist } from "../lib/api";
import { errorMessage } from "../lib/errors";
import { bumpTasteVersion, readCachedRecommendations, writeCachedRecommendations } from "../lib/recommendationCache";
import type { ModelKey, Recommendation } from "../lib/types";

const RECOMMENDATION_LIMIT = 12;

export function RecommendationsPage({ token }: { token: string }) {
  const [model, setModel] = useState<ModelKey>("collaborative");
  const [data, setData] = useState<Recommendation[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [fromCache, setFromCache] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const emptyCopy = model === "popularity"
    ? "No crowd favorites are available right now."
    : `Rate or watch a few movies to unlock ${MODEL_COPY[model].label}. Popular picks stay available while your taste warms up.`;

  useEffect(() => {
    const controller = new AbortController();
    const cached = readCachedRecommendations(token, model, RECOMMENDATION_LIMIT);

    setError(null);
    setActionError(null);
    setFromCache(Boolean(cached));
    setData(cached?.items ?? null);
    setLoading(!cached);

    if (cached) return () => controller.abort();

    getRecommendations(token, model, RECOMMENDATION_LIMIT, controller.signal)
      .then((items) => {
        if (controller.signal.aborted) return;
        writeCachedRecommendations(token, model, RECOMMENDATION_LIMIT, items);
        setData(items);
        setFromCache(false);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) setError(errorMessage(err, "Unable to load recommendations"));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [model, token]);

  function removeRecommendation(movieId: number) {
    setData((current) => {
      const next = current?.filter((item) => item.movie_id !== movieId) ?? [];
      writeCachedRecommendations(token, model, RECOMMENDATION_LIMIT, next);
      return next;
    });
  }

  async function saveToWatchlist(movie: Recommendation) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setWatchlist(token, movie.movie_id, true);
      bumpTasteVersion(token);
      removeRecommendation(movie.movie_id);
    } catch (err) {
      setActionError(errorMessage(err, "Unable to add movie to your watchlist"));
    } finally {
      setBusyId(null);
    }
  }

  async function dismiss(movie: Recommendation) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setNotInterested(token, movie.movie_id);
      bumpTasteVersion(token);
      removeRecommendation(movie.movie_id);
    } catch (err) {
      setActionError(errorMessage(err, "Unable to update your taste feedback"));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Recommendation studio</p>
          <h1>For You</h1>
          <p>{MODEL_COPY[model].detail}</p>
          <div className="header-meta">
            <span>Top {RECOMMENDATION_LIMIT} per model</span>
            <span>Save sparks to Watchlist</span>
            <span>Refreshes after ratings or Watched changes</span>
          </div>
        </div>
      </div>
      <ModelSelector value={model} onChange={setModel} />
      {fromCache && !loading ? <div className="cache-note"><Database size={16} /> Same taste, same treasure map. Using your saved picks.</div> : null}
      {loading ? <MovieGridSkeleton count={RECOMMENDATION_LIMIT} withDismiss /> : null}
      {error ? <EmptyState icon={Radar} title="Could not load recommendations" body={error} /> : null}
      {actionError ? <div className="inline-error">{actionError}</div> : null}
      {!loading && data?.length === 0 ? <EmptyState icon={Sparkles} title="No picks yet" body={emptyCopy} /> : null}
      {!loading && !error ? <section className="movie-grid">
        {data?.map((movie) => (
          <MovieCard key={movie.movie_id} movie={movie} busy={busyId === movie.movie_id} onWatch={() => saveToWatchlist(movie)} watchTitle="Add to watchlist" onDismiss={() => dismiss(movie)} />
        ))}
      </section> : null}
    </div>
  );
}
