import { ArrowRight, BarChart3, Brain, Clapperboard, ShieldCheck, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";

export function LandingPage() {
  return (
    <div className="landing">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-mark"><Clapperboard size={22} /></span>
          <span>CineMatch</span>
        </Link>
        <nav>
          <Link to="/auth" className="nav-cta">Sign in</Link>
        </nav>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Personal movie intelligence</p>
          <h1>CineMatch</h1>
          <p>
            A polished recommendation studio that compares five model strategies, learns from your ratings,
            and explains every pick in human language.
          </p>
          <div className="hero-actions">
            <Link className="primary-button" to="/auth">Open CineMatch <ArrowRight size={18} /></Link>
            <Link className="secondary-button" to="/system-design">View architecture</Link>
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
    </div>
  );
}
