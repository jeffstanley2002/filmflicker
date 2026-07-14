import { Check, Star, X } from "lucide-react";
import { useState } from "react";
import type { Movie, Recommendation } from "../lib/types";

type Props = {
  movie: Movie | Recommendation | null;
  busy?: boolean;
  onClose: () => void;
  onConfirm: (rating: number | null) => void;
};

export function RatingDialog({ movie, busy = false, onClose, onConfirm }: Props) {
  const [rating, setRating] = useState<number | null>(4);

  if (!movie) return null;

  return (
    <div className="dialog-backdrop" role="presentation" onClick={onClose}>
      <section className="rating-dialog" role="dialog" aria-modal="true" aria-labelledby="rating-dialog-title" onClick={(event) => event.stopPropagation()}>
        <button className="dialog-close" type="button" onClick={onClose} title="Close">
          <X size={18} />
        </button>
        <p className="eyebrow">Add to watched</p>
        <h2 id="rating-dialog-title">{movie.title}</h2>
        <p className="dialog-subtitle">Rate it now to improve your recommendations immediately, or add it without a rating.</p>
        <div className="dialog-stars" aria-label="Select rating">
          {[1, 2, 3, 4, 5].map((value) => (
            <button key={value} type="button" className={rating && rating >= value ? "dialog-star active" : "dialog-star"} onClick={() => setRating(value)} title={`${value} stars`}>
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
