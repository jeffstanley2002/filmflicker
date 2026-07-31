import { Award, BarChart3, Bookmark, Check, ChevronLeft, ChevronRight, Heart, LogOut, Pencil, Radar, Search, Sparkles } from "lucide-react";
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

const genrePersonas: Record<string, { suffix: string; description: string }> = {
  action: {
    suffix: "explorer",
    description: "Big set pieces, clean momentum, and heroes making impossible choices are your comfort zone.",
  },
  adventure: {
    suffix: "quest chaser",
    description: "You like stories that pack a bag, cross the map, and find trouble with a grin.",
  },
  animation: {
    suffix: "wonder collector",
    description: "You chase color, heart, and imagination that can sneak up and flatten you emotionally.",
  },
  children: {
    suffix: "storybook scout",
    description: "Warm, bright, easy-to-love movies have a permanent seat in your watch queue.",
  },
  comedy: {
    suffix: "laugh hunter",
    description: "Your taste has a soft spot for sharp timing, chaos, and a perfectly landed joke.",
  },
  crime: {
    suffix: "case cracker",
    description: "Schemes, motives, double-crosses, and morally messy choices keep you leaning forward.",
  },
  documentary: {
    suffix: "truth seeker",
    description: "You like movies that open a real door and leave you thinking about it later.",
  },
  drama: {
    suffix: "story deep-diver",
    description: "You gravitate toward complicated people, quiet tension, and feelings with consequences.",
  },
  fantasy: {
    suffix: "realm roamer",
    description: "Give you strange worlds, old magic, and impossible rules that somehow make emotional sense.",
  },
  "film-noir": {
    suffix: "shadow walker",
    description: "You appreciate smoky motives, sharp silhouettes, and trouble arriving in excellent lighting.",
  },
  horror: {
    suffix: "midnight braveheart",
    description: "You are here for dread, atmosphere, and the delicious mistake of opening the wrong door.",
  },
  musical: {
    suffix: "showtime soul",
    description: "If the feelings are too large to speak, you are fully willing to let them sing.",
  },
  mystery: {
    suffix: "clue chaser",
    description: "You enjoy a movie that trusts you to notice the small thing before the big reveal.",
  },
  romance: {
    suffix: "heart-reader",
    description: "You like chemistry, longing, timing, and the tiny looks that do all the damage.",
  },
  "sci-fi": {
    suffix: "future mapper",
    description: "You chase big ideas, strange tech, and human questions hiding inside impossible futures.",
  },
  thriller: {
    suffix: "tension tuner",
    description: "You like the pulse rising one careful beat at a time until sitting still becomes a sport.",
  },
  war: {
    suffix: "frontline historian",
    description: "You are drawn to pressure, sacrifice, and stories where every choice carries weight.",
  },
  western: {
    suffix: "frontier rider",
    description: "Open horizons, old codes, and hard choices under a wide sky are very much your lane.",
  },
};

function personaForGenre(genre: string) {
  return genrePersonas[genre.toLowerCase()] ?? {
    suffix: "curator",
    description: `Your ${genre} streak gives your recommendations a distinct little signature.`,
  };
}

function tasteBadge(analytics: Analytics | null) {
  if (!analytics || analytics.movies_watched === 0) {
    return {
      label: "Taste warming up",
      reason: "Rate a few movies and FilmFlicker will turn those sparks into a proper taste alter ego.",
    };
  }

  const genres = topEntries(analytics.genre_breakdown);
  const favoriteGenre = genres[0]?.name ?? "Mixed";
  const hasWideTaste = genres.length >= 3;
  const persona = favoriteGenre === "Mixed"
    ? {
        suffix: "sampler",
        description: "Your taste wanders across lanes, which makes your For You shelf harder to predict in the best way.",
      }
    : personaForGenre(favoriteGenre);
  const label = hasWideTaste ? `${favoriteGenre} ${persona.suffix}` : `${favoriteGenre} fan`;

  return {
    label,
    reason: persona.description,
  };
}

export function AppShell({
  email,
  displayName,
  token,
  onSignOut,
  onUpdateDisplayName,
}: {
  email: string;
  displayName: string;
  token: string;
  onSignOut: () => Promise<void>;
  onUpdateDisplayName: (nextName: string) => Promise<void>;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [peeking, setPeeking] = useState(false);
  const [confirmingSignOut, setConfirmingSignOut] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const [signOutError, setSignOutError] = useState<string | null>(null);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(displayName);
  const [savingName, setSavingName] = useState(false);
  const [nameError, setNameError] = useState<string | null>(null);
  const name = displayName || email.split("@")[0] || "Movie friend";
  const badge = useMemo(() => tasteBadge(analytics), [analytics]);

  useEffect(() => {
    setNameDraft(displayName);
  }, [displayName]);

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

  async function saveDisplayName() {
    setSavingName(true);
    setNameError(null);
    try {
      await onUpdateDisplayName(nameDraft);
      setEditingName(false);
    } catch (error) {
      setNameError(errorMessage(error, "Unable to update your display name."));
    } finally {
      setSavingName(false);
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
              <small>{email}</small>
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
              <div className="profile-name-editor">
                {editingName ? (
                  <>
                    <input
                      value={nameDraft}
                      onChange={(event) => setNameDraft(event.target.value)}
                      placeholder="Display name"
                      maxLength={40}
                      aria-label="Display name"
                    />
                    <button type="button" onClick={saveDisplayName} disabled={savingName} aria-label="Save display name">
                      <Check size={15} />
                    </button>
                  </>
                ) : (
                  <button type="button" onClick={() => setEditingName(true)}>
                    <Pencil size={13} />
                    Change display name
                  </button>
                )}
              </div>
              {nameError ? <p className="profile-name-error">{nameError}</p> : null}
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
