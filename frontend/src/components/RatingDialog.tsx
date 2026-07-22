import { Check, Star, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import type { Movie, Recommendation } from "../lib/types";
import { useModalFocus } from "../lib/useModalFocus";

type Props = {
  movie: Movie | Recommendation | null;
  busy?: boolean;
  onClose: () => void;
  onConfirm: (rating: number | null) => void;
};

export function RatingDialog({ movie, busy = false, onClose, onConfirm }: Props) {
  const [rating, setRating] = useState<number | null>(4);
  const dialogRef = useRef<HTMLElement>(null);
  const titleId = useId();
  const descriptionId = useId();
  useEffect(() => setRating(4), [movie?.movie_id]);
  useModalFocus(Boolean(movie), dialogRef, onClose, busy);

  if (!movie) return null;

  return (
    <div className="dialog-backdrop" role="presentation" onClick={() => { if (!busy) onClose(); }}>
      <section ref={dialogRef} className="rating-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={descriptionId} onClick={(event) => event.stopPropagation()}>
        <button className="dialog-close" type="button" onClick={onClose} aria-label="Close dialog" disabled={busy}>
          <X size={18} />
        </button>
        <p className="eyebrow">Add to watched</p>
        <h2 id={titleId}>{movie.title}</h2>
        <p id={descriptionId} className="dialog-subtitle">Rate it now to improve your recommendations immediately, or add it without a rating.</p>
        <div className="dialog-stars" role="radiogroup" aria-label="Select rating">
          {[1, 2, 3, 4, 5].map((value) => (
            <button key={value} type="button" role="radio" aria-checked={rating === value} className={rating && rating >= value ? "dialog-star active" : "dialog-star"} onClick={() => setRating(value)} title={`${value} stars`}>
              <Star size={26} />
            </button>
          ))}
        </div>
        <div className="dialog-actions">
          <button className="secondary-button" type="button" disabled={busy} onClick={() => onConfirm(null)}>
            Add without rating
          </button>
          <button className="primary-button" type="button" disabled={busy || !rating} onClick={() => onConfirm(rating)}>
            <Check size={18} /> Save watched
          </button>
        </div>
      </section>
    </div>
  );
}
