import { Award, BarChart3, Bookmark, ChevronLeft, ChevronRight, Heart, LogOut, Radar, Search, Sparkles, Tags, TrendingUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { BrandMark } from "./BrandMark";
import { ConfirmDialog } from "./ConfirmDialog";
import { getAnalytics } from "../lib/api";
import { errorMessage } from "../lib/errors";
import type { Analytics } from "../lib/types";

function topEntries(record: Record<string, number>, limit = 3) {
  return Object.entries(record)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([name, value]) => ({ name, value }));
}

function tasteBadge(analytics: Analytics | null) {
  if (!analytics || analytics.movies_watched === 0) {
    return {
      label: "Taste warming up",
      reason: "Rate and watch a few movies to unlock a sharper taste badge.",
      genres: ["No genre signal yet"],
      clusters: ["Discovery style pending"],
      stats: "0 movies watched",
    };
  }

  const genres = topEntries(analytics.genre_breakdown);
  const clusters = topEntries(analytics.cluster_breakdown, 2);
  const favoriteGenre = genres[0]?.name ?? "Mixed";
  const averageRating = analytics.avg_rating ?? 0;
  const hasStrongRatings = averageRating >= 4;
  const hasWideTaste = genres.length >= 3;
  const label = hasWideTaste ? `${favoriteGenre} explorer` : hasStrongRatings ? `${favoriteGenre} fan` : `${favoriteGenre} scout`;

  return {
    label,
    reason: hasWideTaste
      ? "Assigned from your strongest genre signals and varied watch history."
      : "Assigned from the movies you have watched and rated so far.",
    genres: genres.length ? genres.map((entry) => `${entry.name} (${entry.value})`) : ["No genre signal yet"],
    clusters: clusters.length ? clusters.map((entry) => `${entry.name} (${entry.value})`) : ["Discovery style pending"],
    stats: `${analytics.movies_watched} watched · ${analytics.movies_rated} rated · ${averageRating.toFixed(1)} avg`,
  };
}

export function AppShell({ email, token, onSignOut }: { email: string; token: string; onSignOut: () => Promise<void> }) {
  const [collapsed, setCollapsed] = useState(false);
  const [peeking, setPeeking] = useState(false);
  const [confirmingSignOut, setConfirmingSignOut] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const [signOutError, setSignOutError] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const name = email.split("@")[0] || "Movie friend";
  const badge = useMemo(() => tasteBadge(analytics), [analytics]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const loadAnalytics = () => getAnalytics(token, controller.signal)
      .then((nextAnalytics) => {
        if (active) setAnalytics(nextAnalytics);
      })
      .catch(() => {
        if (active && !controller.signal.aborted) setAnalytics(null);
      });
    const refreshAnalytics = () => {
      void loadAnalytics();
    };

    void loadAnalytics();
    window.addEventListener("filmflicker:taste-changed", refreshAnalytics);
    return () => {
      active = false;
      controller.abort();
      window.removeEventListener("filmflicker:taste-changed", refreshAnalytics);
    };
  }, [token]);

  async function confirmSignOut() {
    setSigningOut(true);
    setSignOutError(null);
    try {
      await onSignOut();
    } catch (error) {
      setSignOutError(errorMessage(error, "Unable to sign out. Please try again."));
    } finally {
      setSigningOut(false);
    }
  }

  return (
    <div className={`app-shell${collapsed ? " sidebar-collapsed" : ""}${peeking ? " sidebar-peeking" : ""}`}>
      <aside className="sidebar" aria-label="Primary navigation" onMouseLeave={() => setPeeking(false)}>
        <button type="button" className="sidebar-toggle" onClick={() => { setPeeking(false); setCollapsed((current) => !current); }} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
        </button>
        <NavLink to="/app/browse" className="brand">
          <span className="brand-mark"><BrandMark /></span>
          <span>FilmFlicker</span>
        </NavLink>
        <nav>
          <NavLink to="/app/browse"><Search size={18} /><span>Taste Builder</span></NavLink>
          <NavLink to="/app/recommendations"><Radar size={18} /><span>For You</span></NavLink>
          <NavLink to="/app/watchlist"><Bookmark size={18} /><span>Watchlist</span></NavLink>
          <NavLink to="/app/watched"><Heart size={18} /><span>Watched</span></NavLink>
          <NavLink to="/app/analytics"><BarChart3 size={18} /><span>Taste</span></NavLink>
        </nav>
        <div className="sidebar-user">
          <div className="profile-card" tabIndex={0} aria-describedby="taste-badge-popover" aria-label="Taste badge">
            <div className="profile-avatar"><Sparkles size={16} /></div>
            <div className="profile-copy">
              <strong>{name}</strong>
              <span>{badge.label}</span>
            </div>
            <div className="profile-popover" id="taste-badge-popover" role="tooltip">
              <div className="badge-popover-heading">
                <span><Award size={16} /></span>
                <div>
                  <strong>{badge.label}</strong>
                  <p>{badge.reason}</p>
                </div>
              </div>
              <dl>
                <div>
                  <dt><Tags size={13} /> Top tastes</dt>
                  <dd>{badge.genres.join(", ")}</dd>
                </div>
                <div>
                  <dt><Sparkles size={13} /> Discovery style</dt>
                  <dd>{badge.clusters.join(", ")}</dd>
                </div>
                <div>
                  <dt><TrendingUp size={13} /> Profile signal</dt>
                  <dd>{badge.stats}</dd>
                </div>
              </dl>
            </div>
          </div>
          <button className="ghost-button" onClick={() => { setSignOutError(null); setConfirmingSignOut(true); }}><LogOut size={16} /><span>Sign out</span></button>
        </div>
        {collapsed ? <div className="sidebar-reveal" onMouseEnter={() => setPeeking(true)} aria-hidden="true" /> : null}
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
      <ConfirmDialog
        open={confirmingSignOut}
        busy={signingOut}
        title="Sign out of FilmFlicker?"
        body={signOutError ?? "Your ratings and watched list are saved. You can jump back in anytime."}
        confirmLabel="Sign out"
        onCancel={() => { setConfirmingSignOut(false); setSignOutError(null); }}
        onConfirm={confirmSignOut}
      />
    </div>
  );
}
