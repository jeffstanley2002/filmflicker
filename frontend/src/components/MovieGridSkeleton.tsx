import { Star } from "lucide-react";

export function MovieGridSkeleton({ count = 8, watched = false, withDismiss = false }: { count?: number; watched?: boolean; withDismiss?: boolean }) {
  return (
    <section className="movie-grid" aria-label="Loading movies">
      {Array.from({ length: count }, (_, index) => (
        <article className="movie-card skeleton-card" aria-hidden="true" key={index}>
          <div className="poster skeleton-poster">
            {watched ? <div className="poster-badge skeleton-badge">Watched</div> : null}
          </div>
          <div className="movie-body">
            <div>
              <span className="skeleton-line title" />
              <span className="skeleton-line meta" />
            </div>
            <div className="movie-actions">
              <span className="watch-button skeleton-button" />
              {withDismiss ? <span className="watch-button skeleton-button small" /> : null}
              <div className="rating-row" aria-hidden="true">
                {[1, 2, 3, 4, 5].map((rating) => (
                  <span className="star skeleton-star" key={rating}><Star size={16} /></span>
                ))}
              </div>
            </div>
          </div>
        </article>
      ))}
    </section>
  );
}
