import { Heart } from "lucide-react";
import { useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { MovieCard } from "../components/MovieCard";
import { getWatched, setRating, setWatched } from "../lib/api";
import { errorMessage } from "../lib/errors";
import { useAsync } from "../lib/useAsync";
import type { Movie } from "../lib/types";

export function WatchedPage({ token }: { token: string }) {
  const [page, setPage] = useState(1);
  const { data, loading, error, setData } = useAsync((signal) => getWatched(token, page, signal), [token, page]);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function remove(movie: Movie) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setWatched(token, movie.movie_id, false);
      setData((current) => current ? { ...current, total: Math.max(0, current.total - 1), results: current.results.filter((item) => item.movie_id !== movie.movie_id) } : null);
    } catch (err) {
      setActionError(errorMessage(err, "Unable to remove movie"));
    } finally {
      setBusyId(null);
    }
  }

  async function rate(movie: Movie, rating: number) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setRating(token, movie.movie_id, rating);
      setData((current) => current ? { ...current, results: current.results.map((item) => (item.movie_id === movie.movie_id ? { ...item, user_rating: rating } : item)) } : null);
    } catch (err) {
      setActionError(errorMessage(err, "Unable to save rating"));
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
      {!loading && data?.results.length === 0 ? <EmptyState icon={Heart} title="Nothing watched yet" body="Mark a few movies watched from Browse, then rate your favorites." /> : null}
      {!loading && !error ? <section className="movie-grid">
        {data?.results.map((movie) => (
          <MovieCard key={movie.movie_id} movie={movie} watched userRating={movie.user_rating} busy={busyId === movie.movie_id} onWatch={() => remove(movie)} onRate={(rating) => rate(movie, rating)} />
        ))}
      </section> : null}
      {!loading && data && data.total > data.page_size ? <div className="pagination">
        <button className="secondary-button" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>Previous</button>
        <span>Page {data.page} · {data.total.toLocaleString()} watched</span>
        <button className="secondary-button" disabled={page * data.page_size >= data.total} onClick={() => setPage((value) => value + 1)}>Next</button>
      </div> : null}
    </div>
  );
}
