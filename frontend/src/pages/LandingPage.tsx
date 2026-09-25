import {
  ArrowRight,
  BadgeCheck,
  BarChart3,
  Boxes,
  Brain,
  BriefcaseBusiness,
  Code2,
  Database,
  Layers3,
  LockKeyhole,
  Network,
  Radar,
  Server,
  ShieldCheck,
  Sparkles,
  UserPlus,
  Workflow,
  Zap,
} from "lucide-react";
import { Link } from "react-router";
import { BrandMark } from "../components/BrandMark";

const recommenderStack = [
  {
    icon: Radar,
    name: "Collaborative hybrid",
    role: "Rating predictor",
    metric: "0.899",
    label: "RMSE",
    detail: "Learns shared watching patterns and remains the strongest rating predictor.",
  },
  {
    icon: Brain,
    name: "Taste embeddings",
    role: "Neural challenger",
    metric: "17.3%",
    label: "Top-10 overlap",
    detail: "A real two-tower model that contributes different candidate signal from collaborative.",
  },
  {
    icon: Layers3,
    name: "Content TF-IDF",
    role: "Top-10 leader",
    metric: "12.5%",
    label: "Hit Rate@10",
    detail: "Connects movies through genres and tags and leads the full-32M release gate.",
  },
  {
    icon: Network,
    name: "Taste neighborhoods",
    role: "Discovery map",
    metric: "21.83",
    label: "Novelty bits",
    detail: "Clusters taste lanes for exploration and more readable profile insights.",
  },
  {
    icon: BarChart3,
    name: "Popularity baseline",
    role: "Cold-start guard",
    metric: "1.7%",
    label: "Hit Rate@10",
    detail: "Keeps the app useful before enough ratings exist for personalization.",
  },
];

const architectureFlow = [
  ["MovieLens 32M", "Movies, tags, links, and rating histories shape the offline training set."],
  ["Temporal tests", "Latest interactions are held out so models predict what a user likes next."],
  ["Model artifacts", "Validated weights, vectors, metrics, and checksums ship with the API."],
  ["FastAPI service", "Scores candidates, filters watched movies, explains picks, and saves feedback."],
  ["React product", "Turns every rating into a cleaner For You loop and personal taste view."],
];

const systemPlanes = [
  {
    icon: Database,
    title: "Data plane",
    copy: "MovieLens trains the models while Supabase stores each user's private ratings, watchlist, and feedback.",
  },
  {
    icon: Server,
    title: "Serving plane",
    copy: "FastAPI loads checked artifacts once, then serves recommendations, browsing, analytics, and health endpoints.",
  },
  {
    icon: LockKeyhole,
    title: "Trust plane",
    copy: "Supabase auth scopes every request to the signed-in user, with database policies as a second guard.",
  },
  {
    icon: BadgeCheck,
    title: "Release plane",
    copy: "Model metrics, checksums, export validation, API readiness, and frontend checks gate every release.",
  },
];

export function LandingPage() {
  return (
    <div className="landing">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark"><BrandMark /></span>
          <span>FilmFlicker</span>
        </Link>
        <nav>
          <Link to="/signin" className="nav-cta">Sign in</Link>
          <Link to="/register" className="nav-register">
            <UserPlus size={17} />
            Get started
          </Link>
        </nav>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Personal movie intelligence</p>
          <h1>FilmFlicker</h1>
          <p>
            A polished recommendation studio that compares five model strategies, learns from your ratings,
            and explains every pick in human language.
          </p>
          <div className="hero-actions">
            <Link className="primary-button" to="/register">Start matching <ArrowRight size={18} /></Link>
            <a className="secondary-button" href="#architecture">View architecture</a>
          </div>
        </div>
        <div className="signal-panel">
          <div className="signal-row"><Brain size={18} /> Five recommender models</div>
          <div className="signal-meter"><span style={{ width: "84%" }} /></div>
          <div className="signal-row"><Sparkles size={18} /> Taste profile gets smarter as you rate</div>
          <div className="signal-meter"><span style={{ width: "72%" }} /></div>
          <div className="signal-row"><ShieldCheck size={18} /> Supabase-backed user isolation</div>
          <div className="signal-meter"><span style={{ width: "91%" }} /></div>
        </div>
      </section>

      <section className="feature-band">
        {[
          ["Clear model choices", "Plain-English labels make advanced recommenders approachable."],
          ["Personal analytics", "Genres, ratings, decades, and discovery styles reveal what your watch history says."],
          ["Deployment confidence", "Exported metrics and artifact checks keep model releases reviewable."],
        ].map(([title, body]) => (
          <article key={title}>
            <BarChart3 size={22} />
            <h2>{title}</h2>
            <p>{body}</p>
          </article>
        ))}
      </section>

      <main className="landing-main">
        <section className="model-showcase" aria-labelledby="landing-model-title">
          <div className="model-intro">
            <div>
              <p className="eyebrow">Five recommenders</p>
              <h2 id="landing-model-title">A model stack with clear jobs, not one magic ranking trick.</h2>
              <p>
                FilmFlicker compares complementary recommenders, then uses the right signal for the moment:
                cold starts, similarity, collaborative taste, neural challengers, and discovery lanes.
              </p>
            </div>
            <div className="model-proof-strip" aria-label="Model release proof">
              <span><strong>5</strong> strategies</span>
              <span><strong>32M</strong> ratings</span>
              <span><strong>1K</strong> test users</span>
            </div>
          </div>
          <div className="model-showcase-grid landing-model-grid">
            {recommenderStack.map(({ icon: Icon, name, role, metric, label, detail }) => (
              <article key={name} tabIndex={0}>
                <div className="model-card-top">
                  <span><Icon size={20} /></span>
                  <small>{role}</small>
                </div>
                <h3>{name}</h3>
                <p>{detail}</p>
                <div className="model-metric">
                  <strong>{metric}</strong>
                  <span>{label}</span>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="system-flow" id="architecture" aria-labelledby="architecture-title">
          <div className="flow-intro">
            <p className="eyebrow">Architecture</p>
            <h2 id="architecture-title">How a rating becomes a recommendation.</h2>
            <p>
              Data becomes trained artifacts, artifacts become scored candidates, and the app turns those scores into
              a useful, explainable product loop.
            </p>
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

        <section className="system-planes" aria-labelledby="system-planes-title">
          <div className="section-title">
            <div>
              <p className="eyebrow">System shape</p>
              <h2 id="system-planes-title">Professional release mechanics around the models.</h2>
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
            <p className="eyebrow">Release posture</p>
            <h2>Validated, honest, and ready to demo.</h2>
            <p>
              The launch story is intentionally specific: content matching leads the top-10 gate, collaborative is
              strongest for rating prediction, neural taste embeddings are a real independent challenger, and every
              artifact is checked before serving.
            </p>
          </div>
          <div className="readiness-stack" aria-label="Release verification">
            <span><Workflow size={18} /> chronological evaluation</span>
            <span><Boxes size={18} /> artifact checksums</span>
            <span><Zap size={18} /> cached model serving</span>
            <span><Sparkles size={18} /> explainable product loop</span>
          </div>
        </section>
      </main>

      <footer className="landing-footer">
        <span>Copyright © 2026 Jeffrey Stanley</span>
        <div>
          <a href="https://github.com/jeffstanley2002" target="_blank" rel="noreferrer" aria-label="Jeffrey Stanley on GitHub">
            <Code2 size={22} />
          </a>
          <a href="https://www.linkedin.com/in/jeffrey-stanley-148119197" target="_blank" rel="noreferrer" aria-label="Jeffrey Stanley on LinkedIn">
            <BriefcaseBusiness size={22} />
          </a>
        </div>
      </footer>
    </div>
  );
}
