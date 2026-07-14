import { Film, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CustomSelect } from "../components/CustomSelect";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { MovieCard } from "../components/MovieCard";
import { getGenres, getMovies, setRating, setWatched } from "../lib/api";
import type { Movie, MoviePage } from "../lib/types";

export function BrowsePage({ token }: { token: string }) {
  const [genres, setGenres] = useState<string[]>([]);
  const [movies, setMovies] = useState<MoviePage | null>(null);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [genre, setGenre] = useState("");
  const [sortBy, setSortBy] = useState("popularity");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  const params = useMemo(() => {
    const next = new URLSearchParams({ page: String(page), page_size: "20", sort_by: sortBy });
    if (debouncedQuery.trim()) next.set("query", debouncedQuery.trim());
    if (genre) next.set("genre", genre);
    return next;
  }, [debouncedQuery, genre, page, sortBy]);

  const genreOptions = useMemo(() => [{ value: "", label: "All genres" }, ...genres.map((item) => ({ value: item, label: item }))], [genres]);
  const sortOptions = [
    { value: "popularity", label: "Most trusted" },
    { value: "rating", label: "Highest rated" },
    { value: "newest", label: "Newest" },
    { value: "title", label: "Title" },
  ];

  useEffect(() => {
    getGenres(token).then(setGenres).catch(() => setGenres([]));
  }, [token]);

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedQuery(query), 180);
    return () => window.clearTimeout(timeout);
  }, [query]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    getMovies(token, params, controller.signal)
      .then((value) => {
        if (!cancelled) setMovies(value);
      })
      .catch((err: unknown) => {
        if (!cancelled && !(err instanceof DOMException && err.name === "AbortError")) {
          setError(err instanceof Error ? err.message : "Unable to load movies");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [params, token]);

  async function updateMovie(movie: Movie, patch: Partial<Movie>) {
    setMovies((current) =>
      current
        ? { ...current, results: current.results.map((item) => (item.movie_id === movie.movie_id ? { ...item, ...patch } : item)) }
        : current,
    );
  }

  async function markWatched(movie: Movie, watched: boolean) {
    setBusyId(movie.movie_id);
    setError(null);
    try {
      await setWatched(token, movie.movie_id, watched);
      await updateMovie(movie, { watched, user_rating: watched ? movie.user_rating : null });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update watched state");
    } finally {
      setBusyId(null);
    }
  }

  async function rate(movie: Movie, rating: number) {
    setBusyId(movie.movie_id);
    setError(null);
    try {
      await setRating(token, movie.movie_id, rating);
      await updateMovie(movie, { watched: true, user_rating: rating });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save rating");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Library</p>
          <h1>Browse Movies</h1>
          <p>Search, filter, save, and rate movies so the models learn your taste.</p>
        </div>
      </div>
      <div className="filter-bar" role="search">
        <label><Search size={17} /><input value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} placeholder="Search titles" /></label>
        <CustomSelect value={genre} options={genreOptions} ariaLabel="Filter by genre" onChange={(value) => { setGenre(value); setPage(1); }} />
        <CustomSelect value={sortBy} options={sortOptions} ariaLabel="Sort movies" onChange={(value) => { setSortBy(value); setPage(1); }} />
      </div>
      {loading ? <Loading label="Loading movies" /> : null}
      {error ? <EmptyState icon={Film} title="Could not load movies" body={error} /> : null}
      {!loading && movies?.results.length === 0 ? <EmptyState icon={Film} title="No movies found" body="Try a broader search or another genre." /> : null}
      {!loading && !error ? <section className="movie-grid">
        {movies?.results.map((movie) => (
          <MovieCard key={movie.movie_id} movie={movie} watched={movie.watched} userRating={movie.user_rating} busy={busyId === movie.movie_id} onWatch={(watched) => markWatched(movie, watched)} onRate={(rating) => rate(movie, rating)} />
        ))}
      </section> : null}
      {!loading && movies ? (
        <div className="pagination">
          <button className="secondary-button" disabled={page <= 1} onClick={() => setPage((current) => current - 1)}>Previous</button>
          <span>Page {movies.page} · {movies.total.toLocaleString()} matches</span>
          <button className="secondary-button" disabled={page * movies.page_size >= movies.total} onClick={() => setPage((current) => current + 1)}>Next</button>
        </div>
      ) : null}
    </div>
  );
}
