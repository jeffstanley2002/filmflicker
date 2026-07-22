import { Bookmark, Sparkles } from "lucide-react";
import { useState } from "react";
import { EmptyState } from "../components/EmptyState";
import { MovieCard } from "../components/MovieCard";
import { MovieGridSkeleton } from "../components/MovieGridSkeleton";
import { getWatchlist, setRating, setWatchlist } from "../lib/api";
import { errorMessage } from "../lib/errors";
import { bumpTasteVersion } from "../lib/recommendationCache";
import { useAsync } from "../lib/useAsync";
import type { Movie } from "../lib/types";

export function WatchlistPage({ token }: { token: string }) {
  const [page, setPage] = useState(1);
  const { data, loading, error, setData } = useAsync((signal) => getWatchlist(token, page, signal), [token, page]);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function remove(movie: Movie) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setWatchlist(token, movie.movie_id, false);
      setData((current) => current ? { ...current, total: Math.max(0, current.total - 1), results: current.results.filter((item) => item.movie_id !== movie.movie_id) } : null);
    } catch (err) {
      setActionError(errorMessage(err, "Unable to remove movie from watchlist"));
    } finally {
      setBusyId(null);
    }
  }

  async function markWatched(movie: Movie, rating: number) {
    setBusyId(movie.movie_id);
    setActionError(null);
    try {
      await setRating(token, movie.movie_id, rating);
      bumpTasteVersion(token);
      setData((current) => current ? { ...current, total: Math.max(0, current.total - 1), results: current.results.filter((item) => item.movie_id !== movie.movie_id) } : null);
    } catch (err) {
      setActionError(errorMessage(err, "Unable to move movie to watched"));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Saved for later</p>
          <h1>Watchlist</h1>
          <p>Your maybe pile. When a pick earns its popcorn, tap a star and it jumps into Watched.</p>
        </div>
      </div>
      {loading ? <MovieGridSkeleton count={8} withDismiss /> : null}
      {error ? <EmptyState icon={Bookmark} title="Could not load watchlist" body={error} /> : null}
      {actionError ? <div className="inline-error">{actionError}</div> : null}
      {!loading && data?.results.length === 0 ? <EmptyState icon={Sparkles} title="Your watchlist is empty" body="Save sparks from For You when something looks worth a couch slot." /> : null}
      {!loading && !error ? <section className="movie-grid">
        {data?.results.map((movie) => (
          <MovieCard
            key={movie.movie_id}
            movie={movie}
            busy={busyId === movie.movie_id}
            onRate={(rating) => markWatched(movie, rating)}
            onDismiss={() => remove(movie)}
            dismissTitle="Remove from watchlist"
          />
        ))}
      </section> : null}
      {!loading && data && data.total > data.page_size ? <div className="pagination">
        <button className="secondary-button" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>Previous</button>
        <span>Page {data.page} · {data.total.toLocaleString()} saved</span>
        <button className="secondary-button" disabled={page * data.page_size >= data.total} onClick={() => setPage((value) => value + 1)}>Next</button>
      </div> : null}
    </div>
  );
}
