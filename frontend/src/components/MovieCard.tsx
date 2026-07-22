import { Check, Clapperboard, Plus, Star, ThumbsDown } from "lucide-react";
import { useState } from "react";
import { resolveAssetUrl } from "../lib/api";
import type { Movie, Recommendation } from "../lib/types";

type Props = {
  movie: Movie | Recommendation;
  watched?: boolean;
  userRating?: number | null;
  busy?: boolean;
  onWatch?: (watched: boolean) => void;
  onRate?: (rating: number) => void;
  onDismiss?: () => void;
};

function cleanTitle(title: string) {
  return title.replace(/\s*\([^)]*\)\s*$/g, "").trim();
}

export function MovieCard({ movie, watched = false, userRating = null, busy = false, onWatch, onRate, onDismiss }: Props) {
  const gradient = `poster poster-${movie.movie_id % 6}`;
  const [posterFailed, setPosterFailed] = useState(false);
  const showPoster = Boolean(movie.poster_url) && !posterFailed;
  const posterUrl = resolveAssetUrl(movie.poster_url);

  return (
    <article className="movie-card">
      <div className={gradient}>
        {showPoster ? (
          <img src={posterUrl ?? undefined} alt={`${cleanTitle(movie.title)} poster`} loading="lazy" decoding="async" referrerPolicy="no-referrer" onError={() => setPosterFailed(true)} />
        ) : (
          <div className="poster-fallback" aria-label={`${movie.title} poster`}>
            <Clapperboard size={34} />
            <strong>{cleanTitle(movie.title)}</strong>
            <small>{movie.year ?? "CineMatch"} · {movie.genres[0] ?? "Movie"}</small>
          </div>
        )}
        {watched ? <div className="poster-badge">Watched</div> : null}
      </div>
      <div className="movie-body">
        <div>
          <h3>{movie.title}</h3>
          <p>{movie.year ?? "Unknown year"} · {movie.genres.slice(0, 3).join(", ") || "No genre"}</p>
        </div>
        {"reason" in movie ? <p className="reason">{movie.reason}</p> : null}
        <div className="movie-actions">
          {onWatch ? (
            <button className={watched ? "watch-button active" : "watch-button"} disabled={busy} onClick={() => onWatch(!watched)} aria-label={watched ? "Remove from watched" : "Mark watched"} title={watched ? "Remove from watched" : "Mark watched"}>
              {watched ? <Check size={17} /> : <Plus size={17} />}
            </button>
          ) : null}
          {onDismiss ? (
            <button className="watch-button dismiss-button" disabled={busy} onClick={onDismiss} aria-label="Not interested" title="Not interested">
              <ThumbsDown size={16} />
            </button>
          ) : null}
          {onRate ? (
            <div className="rating-row" aria-label="Rate movie">
              {[1, 2, 3, 4, 5].map((rating) => (
                <button key={rating} disabled={busy} className={userRating && userRating >= rating ? "star active" : "star"} onClick={() => onRate(rating)} title={`${rating} stars`}>
                  <Star size={16} />
                </button>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </article>
  );
}
