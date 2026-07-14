import { Heart } from "lucide-react";
import { useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { MovieCard } from "../components/MovieCard";
import { getWatched, setRating, setWatched } from "../lib/api";
import { useAsync } from "../lib/useAsync";
import type { Movie } from "../lib/types";

export function WatchedPage({ token }: { token: string }) {
  const { data, loading, error, setData } = useAsync(() => getWatched(token), [token]);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function remove(movie: Movie) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setWatched(token, movie.movie_id, false);
      setData((current) => current?.filter((item) => item.movie_id !== movie.movie_id) ?? null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to remove movie");
    } finally {
      setBusyId(null);
    }
  }

  async function rate(movie: Movie, rating: number) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setRating(token, movie.movie_id, rating);
      setData((current) => current?.map((item) => (item.movie_id === movie.movie_id ? { ...item, user_rating: rating } : item)) ?? null);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to save rating");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Your history</p>
          <h1>Watched Movies</h1>
          <p>Keep this sharp and your recommendations improve fast.</p>
        </div>
      </div>
      {loading ? <Loading label="Loading watched movies" /> : null}
      {error ? <EmptyState icon={Heart} title="Could not load watched movies" body={error} /> : null}
      {actionError ? <div className="inline-error">{actionError}</div> : null}
      {!loading && data?.length === 0 ? <EmptyState icon={Heart} title="Nothing watched yet" body="Mark a few movies watched from Browse, then rate your favorites." /> : null}
      {!loading && !error ? <section className="movie-grid">
        {data?.map((movie) => (
          <MovieCard key={movie.movie_id} movie={movie} watched userRating={movie.user_rating} busy={busyId === movie.movie_id} onWatch={() => remove(movie)} onRate={(rating) => rate(movie, rating)} />
        ))}
      </section> : null}
    </div>
  );
}
