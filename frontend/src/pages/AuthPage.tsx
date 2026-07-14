import { FormEvent, useState } from "react";
import { ArrowRight, Clapperboard, Film, Lock, Mail, Play, Sparkles, Star } from "lucide-react";
import { Link, Navigate } from "react-router-dom";
import { BackButton } from "../components/BackButton";
import { hasSupabaseConfig, supabase } from "../lib/supabase";
import type { Session } from "@supabase/supabase-js";

export function AuthPage({ session }: { session: Session | null }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (session) return <Navigate to="/app/browse" replace />;

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!supabase) {
      setMessage("Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to frontend/.env.local.");
      return;
    }
    setLoading(true);
    setMessage(null);
    const result =
      mode === "login"
        ? await supabase.auth.signInWithPassword({ email, password })
        : await supabase.auth.signUp({ email, password });
    setLoading(false);
    if (result.error) {
      setMessage(result.error.message);
      return;
    }
    if (mode === "register" && !result.data.session) {
      setMessage("Registration created. Check your email if confirmation is enabled in Supabase.");
    }
  }

  async function signInWithGoogle() {
    if (!supabase) {
      setMessage("Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to frontend/.env.local.");
      return;
    }
    setLoading(true);
    setMessage(null);
    const { error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${window.location.origin}/app/browse`,
      },
    });
    if (error) {
      setMessage(error.message);
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-art" aria-hidden="true">
        <div className="signin-preview">
          <div className="preview-header">
            <span><Sparkles size={17} /> Taste preview</span>
            <strong>Ready</strong>
          </div>
          <div className="preview-stage">
            <div className="preview-card main">
              <Film size={22} />
              <span>Drama</span>
            </div>
            <div className="preview-card side-a">
              <Play size={18} />
              <span>Sci-Fi</span>
            </div>
            <div className="preview-card side-b">
              <Star size={18} />
              <span>4.7</span>
            </div>
          </div>
          <div className="taste-meter-card">
            <div>
              <span>Recommendation signal</span>
              <strong>Warm, witty, cinematic</strong>
            </div>
            <div className="taste-bars">
              <span style={{ width: "86%" }} />
              <span style={{ width: "64%" }} />
              <span style={{ width: "74%" }} />
            </div>
          </div>
        </div>
      </div>
      <form className="auth-card" onSubmit={submit}>
        <BackButton fallback="/" />
        <div className="auth-brand"><Clapperboard size={28} /> CineMatch</div>
        <h1>{mode === "login" ? "Welcome back" : "Create your account"}</h1>
        <p>Save movies, tune your ratings, and keep your recommendation profile private to your account.</p>
        {!hasSupabaseConfig ? <div className="notice">Supabase frontend env vars are not configured yet.</div> : null}
        <div className="oauth-grid single">
          <button type="button" className="oauth-button" disabled={loading} onClick={signInWithGoogle}>
            <span className="google-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="20" height="20">
                <path fill="#4285F4" d="M22.6 12.2c0-.8-.1-1.6-.2-2.3H12v4.4h5.9c-.3 1.4-1 2.5-2.1 3.3v2.7h3.5c2.1-1.9 3.3-4.7 3.3-8.1Z" />
                <path fill="#34A853" d="M12 23c3 0 5.5-1 7.3-2.7l-3.5-2.7c-1 .7-2.2 1.1-3.8 1.1-2.9 0-5.3-1.9-6.1-4.5H2.3V17c1.8 3.6 5.5 6 9.7 6Z" />
                <path fill="#FBBC05" d="M5.9 14.2c-.2-.7-.4-1.4-.4-2.2s.1-1.5.4-2.2V7H2.3C1.5 8.5 1 10.2 1 12s.5 3.5 1.3 5l3.6-2.8Z" />
                <path fill="#EA4335" d="M12 5.3c1.6 0 3.1.6 4.2 1.7l3.1-3.1C17.5 2.1 15 1 12 1 7.8 1 4.1 3.4 2.3 7l3.6 2.8c.8-2.6 3.2-4.5 6.1-4.5Z" />
              </svg>
            </span>
            Continue with Google
          </button>
        </div>
        <div className="divider-label"><span>Email</span></div>
        <label>
          <Mail size={16} />
          <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Email" required />
        </label>
        <label>
          <Lock size={16} />
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" minLength={6} required />
        </label>
        {message ? <div className="form-error">{message}</div> : null}
        <button className="primary-button" disabled={loading}>{loading ? "Working..." : mode === "login" ? "Sign in" : "Create account"} <ArrowRight size={17} /></button>
        <button type="button" className="text-button" onClick={() => setMode(mode === "login" ? "register" : "login")}>
          {mode === "login" ? "Need an account? Register" : "Already have an account? Sign in"}
        </button>
        <p className="auth-footnote">Google sign-in uses Supabase Auth. Email remains available for fallback access. <Link to="/system-design">View architecture</Link></p>
      </form>
    </div>
  );
}
