import { Radar, Sparkles } from "lucide-react";
import { useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { ModelSelector } from "../components/ModelSelector";
import { MovieCard } from "../components/MovieCard";
import { MODEL_COPY } from "../lib/models";
import { getRecommendations, setNotInterested, setWatched } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import type { ModelKey, Recommendation } from "../lib/types";

export function RecommendationsPage({ token }: { token: string }) {
  const [model, setModel] = useState<ModelKey>("collaborative");
  const { data, loading, error, setData } = useAsync(() => getRecommendations(token, model, 12), [token, model]);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  async function markWatched(movie: Recommendation) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setWatched(token, movie.movie_id, true);
      setData((current) => current?.filter((item) => item.movie_id !== movie.movie_id) ?? null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to save movie");
    } finally {
      setBusyId(null);
    }
  }

  async function dismiss(movie: Recommendation) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setNotInterested(token, movie.movie_id);
      setData((current) => current?.filter((item) => item.movie_id !== movie.movie_id) ?? null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to update your taste feedback");
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
        </div>
      </div>
      <ModelSelector value={model} onChange={setModel} />
      {loading ? <Loading label="Scoring movies" /> : null}
      {error ? <EmptyState icon={Radar} title="Could not load recommendations" body={error} /> : null}
      {actionError ? <div className="inline-error">{actionError}</div> : null}
      {!loading && data?.length === 0 ? <EmptyState icon={Sparkles} title="No recommendations yet" body="Rate a few more movies or try Crowd favorites." /> : null}
      {!loading && !error ? <section className="movie-grid">
        {data?.map((movie) => (
          <MovieCard key={movie.movie_id} movie={movie} busy={busyId === movie.movie_id} onWatch={() => markWatched(movie)} onDismiss={() => dismiss(movie)} />
        ))}
      </section> : null}
    </div>
  );
}
