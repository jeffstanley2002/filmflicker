import { ArrowRight, Brain, Database, Film, GitBranch, LockKeyhole, Server, Sparkles, Star, Workflow } from "lucide-react";
import { Link } from "react-router-dom";

const pipeline = [
  ["Collect", "Movies, ratings, genres, posters."],
  ["Train", "Five recommenders compete."],
  ["Serve", "FastAPI returns picks fast."],
  ["Protect", "Supabase keeps taste private."],
  ["Delight", "React makes it feel instant."],
];

const capabilities = [
  {
    icon: Brain,
    title: "Machine learning depth",
    body: "Popularity, TF-IDF similarity, SVD collaborative filtering, KMeans neighborhoods, and latent taste embeddings run side by side with chronological ranking tests.",
  },
  {
    icon: Server,
    title: "Backend production shape",
    body: "FastAPI routers expose typed contracts, validate inputs, cache expensive model artifacts, and keep idempotent watched/rating APIs for reliable UI state.",
  },
  {
    icon: LockKeyhole,
    title: "Security model",
    body: "Supabase handles signup and session refresh. The API only accepts verified bearer tokens and uses the JWT subject as the database tenant boundary.",
  },
  {
    icon: Database,
    title: "Database clarity",
    body: "A dedicated cinematch_v2 schema separates React-era user data from older prototypes, with uniqueness constraints and cascading deletes from Supabase auth.users.",
  },
];

export function SystemDesignPage() {
  return (
    <div className="system-page">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark"><GitBranch size={22} /></span>
          <span>CineMatch</span>
        </Link>
        <nav>
          <Link to="/signin" className="nav-cta">Open app</Link>
        </nav>
      </header>

      <main className="page system-main">
        <section className="system-hero">
          <div>
            <p className="eyebrow">Architecture tour</p>
            <h1>A recommendation engine you can actually see.</h1>
            <p>Follow one rating as it becomes a smarter movie pick.</p>
          </div>
          <Link className="primary-button" to="/signin">Open app <ArrowRight size={18} /></Link>
        </section>

        <section className="architecture-orbit" aria-label="CineMatch architecture illustration">
          <div className="orbit-core">
            <Sparkles size={34} />
            <strong>CineMatch</strong>
            <span>taste in, movies out</span>
          </div>
          <article>
            <Film size={24} />
            <strong>Catalog</strong>
            <span>movie signals</span>
          </article>
          <article>
            <Brain size={24} />
            <strong>Models</strong>
            <span>ranking brain</span>
          </article>
          <article>
            <Server size={24} />
            <strong>API</strong>
            <span>quick answers</span>
          </article>
          <article>
            <Sparkles size={24} />
            <strong>App</strong>
            <span>fun feedback</span>
          </article>
        </section>

        <section className="decision-grid">
          <article>
            <strong>Current catalog</strong>
            <span>MovieLens 32M trained artifacts with catalog coverage through 2023.</span>
          </article>
          <article>
            <strong>Fresh releases</strong>
            <span>Add a modern catalog source first, then retrain/export before showing them in personalized recommendations.</span>
          </article>
          <article>
            <strong>Production rule</strong>
            <span>No model ships without metrics.json, artifact checks, and API health verification.</span>
          </article>
        </section>

        <section className="pipeline">
          {pipeline.map(([title, body], index) => (
            <article key={title}>
              <span>{index + 1}</span>
              <h2>{title}</h2>
              <p>{body}</p>
            </article>
          ))}
        </section>

        <section className="capability-grid">
          {capabilities.map(({ icon: Icon, title, body }) => (
            <article key={title}>
              <Icon size={24} />
              <h2>{title}</h2>
              <p>{body}</p>
            </article>
          ))}
        </section>

        <section className="review-panel">
          <Workflow size={26} />
          <div>
            <h2>How the loop feels</h2>
            <p>Users browse and rate movies, the API stores private taste signals, and the recommender refreshes picks with explanations that make each model approachable.</p>
          </div>
        </section>

        <section className="review-panel">
          <Star size={26} />
          <div>
            <h2>Why it stays trustworthy</h2>
            <p>New catalog data is added before retraining, metrics are checked before release, and the app only shows model outputs that the API can explain.</p>
          </div>
        </section>
      </main>
    </div>
  );
}
