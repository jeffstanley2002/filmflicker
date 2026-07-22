import {
  ArrowRight,
  BadgeCheck,
  BarChart3,
  Boxes,
  Database,
  Fingerprint,
  Layers3,
  LockKeyhole,
  Network,
  Radar,
  Server,
  ShieldCheck,
  Sparkles,
  Workflow,
  Zap,
} from "lucide-react";
import { Link } from "react-router-dom";
import { BrandMark } from "../components/BrandMark";

const modelStack = [
  {
    icon: Radar,
    name: "Collaborative hybrid",
    role: "Main ranking engine",
    metric: "12.2%",
    label: "Top-10 test",
    strength: "Best launch personalizer",
    detail: "Learns patterns from similar movie fans and drives the strongest launch picks.",
    meaning: "Strict offline test result: this was the strongest personalized ranker for surfacing future-liked movies near the top.",
  },
  {
    icon: Fingerprint,
    name: "Latent taste embeddings",
    role: "Taste fingerprint",
    metric: "0.932",
    label: "Rating error",
    strength: "Fast taste matching",
    detail: "Turns a viewer profile into a compact preference map for fast matching.",
    meaning: "Strict offline test result: this model gives a compact taste fingerprint that is accurate enough for fast matching.",
  },
  {
    icon: Layers3,
    name: "Content TF-IDF",
    role: "Similarity model",
    metric: "11.3%",
    label: "Top-10 test",
    strength: "Strong for new profiles",
    detail: "Connects films by genres, tags, and catalog signals so early recommendations still make sense.",
    meaning: "Strict offline test result: this model stays useful when a user has only a small taste profile.",
  },
  {
    icon: Network,
    name: "Taste neighborhoods",
    role: "Discovery lanes",
    metric: "12.8",
    label: "Novelty score",
    strength: "Better discovery range",
    detail: "Groups viewers into taste neighborhoods to widen discovery without getting random.",
    meaning: "Strict offline test result: this model increases discovery by reaching beyond the most obvious popular picks.",
  },
  {
    icon: BarChart3,
    name: "Popularity baseline",
    role: "Reliable fallback",
    metric: "69.6%",
    label: "Variety score",
    strength: "Safe fallback layer",
    detail: "Keeps the app useful for brand-new profiles and model edge cases.",
    meaning: "Strict offline test result: this baseline keeps recommendations broad enough for cold starts and fallback paths.",
  },
];

const architectureFlow = [
  ["MovieLens 32M", "Movies, tags, links, and 32,000,204 ratings."],
  ["Train and test", "Models compete on time-based user history."],
  ["Artifact vault", "Checked model files load once at API startup."],
  ["FastAPI", "Scores, filters, explains, and saves feedback."],
  ["React app", "Turns the model stack into a clear product loop."],
];

const releaseState = [
  ["Dataset", "MovieLens 32M catalog through 2023."],
  ["Testing", "Checked against 1,000 real user histories."],
  ["Export", "Checksum manifest: July 16, 2026."],
  ["Runtime", "Cached models, single API worker, deploy-ready."],
];

const systemPlanes = [
  {
    icon: Database,
    title: "Data plane",
    copy: "Catalog files train the models. Supabase stores each user's ratings, watched list, watchlist, and feedback.",
  },
  {
    icon: Server,
    title: "Serving plane",
    copy: "FastAPI loads the trained artifacts once, then serves browsing, recommendations, analytics, and health checks.",
  },
  {
    icon: LockKeyhole,
    title: "Trust plane",
    copy: "Supabase auth scopes every request to the signed-in user, with database policies as a second guard.",
  },
  {
    icon: BadgeCheck,
    title: "Release plane",
    copy: "The launch build ships only after metrics, checksums, export validation, API readiness, and frontend checks pass.",
  },
];

export function SystemDesignPage() {
  return (
    <div className="system-page">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark"><BrandMark /></span>
          <span>FilmFlicker</span>
        </Link>
        <nav>
          <Link to="/signin">Sign in</Link>
          <Link to="/register" className="nav-cta">Open app</Link>
        </nav>
      </header>

      <main className="page system-main">
        <section className="system-hero">
          <div className="system-hero-copy">
            <p className="eyebrow">Initial release architecture</p>
            <h1>Five trained movie models. One launch-ready system.</h1>
            <p>
              FilmFlicker combines trained recommendation models, authenticated user feedback, and a fast React app into
              one explainable release.
            </p>
            <div className="hero-actions">
              <Link className="primary-button" to="/signin">Try the release <ArrowRight size={18} /></Link>
              <Link className="secondary-button" to="/">Back to overview</Link>
            </div>
          </div>
          <div className="system-status-card" aria-label="Current system state">
            <span><ShieldCheck size={16} /> Launch state</span>
            <strong>Validated models are ready for the first public build.</strong>
            <dl>
              <div>
                <dt>Strategies</dt>
                <dd>5</dd>
              </div>
              <div>
                <dt>Ratings</dt>
                <dd>32M</dd>
              </div>
              <div>
                <dt>Tested users</dt>
                <dd>1K</dd>
              </div>
            </dl>
          </div>
        </section>

        <section className="model-showcase" aria-labelledby="model-showcase-title">
          <div className="model-intro">
            <div>
              <p className="eyebrow">Model core</p>
              <h2 id="model-showcase-title">A trained model stack, not a single ranking trick.</h2>
              <p>Five recommendation strategies work together so the first release feels personal, explainable, and steady from the first rating.</p>
            </div>
            <div className="model-proof-strip" aria-label="Model stack highlights">
              <span><strong>5</strong> model strategies</span>
              <span><strong>32M</strong> training ratings</span>
              <span><strong>1K</strong> tested users</span>
            </div>
          </div>
          <div className="model-showcase-grid">
            {modelStack.map(({ icon: Icon, name, role, metric, label, strength, detail, meaning }) => (
              <article key={name} tabIndex={0}>
                <div className="model-card-top">
                  <span><Icon size={20} /></span>
                  <small>{role}</small>
                </div>
                <h3>{name}</h3>
                <p>{detail}</p>
                <div className="model-metric">
                  <strong>{strength}</strong>
                  <span>Tested metric: {metric} {label}</span>
                  <small>{meaning}</small>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="system-flow" aria-labelledby="system-flow-title">
          <div className="flow-intro">
            <p className="eyebrow">End-to-end architecture</p>
            <h2 id="system-flow-title">How a rating becomes a recommendation.</h2>
            <p>Clean data becomes trained artifacts, then the API turns those artifacts into ranked movies the app can explain.</p>
          </div>
          <div className="flow-rail">
            {architectureFlow.map(([title, copy], index) => (
              <article key={title}>
                <span>{String(index + 1).padStart(2, "0")}</span>
                <div>
                  <h3>{title}</h3>
                  <p>{copy}</p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="release-grid" aria-label="Current release state">
          {releaseState.map(([title, body]) => (
            <article key={title}>
              <strong>{title}</strong>
              <span>{body}</span>
            </article>
          ))}
        </section>

        <section className="system-planes" aria-labelledby="system-planes-title">
          <div className="section-title">
            <div>
              <p className="eyebrow">System planes</p>
              <h2 id="system-planes-title">Production shape around the models.</h2>
            </div>
          </div>
          <div className="system-plane-grid">
            {systemPlanes.map(({ icon: Icon, title, copy }) => (
              <article key={title}>
                <Icon size={24} />
                <h3>{title}</h3>
                <p>{copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="launch-readiness">
          <div>
            <p className="eyebrow">What initial release means</p>
            <h2>Ready now, with clear boundaries.</h2>
            <p>
              The first release is trained and validated on the current catalog. Newer movies can be added through the
              next catalog refresh, retraining pass, and export check.
            </p>
          </div>
          <div className="readiness-stack" aria-label="Release verification">
            <span><Workflow size={18} /> time-based testing</span>
            <span><Boxes size={18} /> artifact checksums</span>
            <span><Zap size={18} /> cached model serving</span>
            <span><Sparkles size={18} /> explainable product loop</span>
          </div>
        </section>
      </main>
    </div>
  );
}
