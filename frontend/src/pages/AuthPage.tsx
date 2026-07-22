import { FormEvent, useState } from "react";
import { ArrowRight, Lock, Mail, Sparkles, UserPlus } from "lucide-react";
import { Link, Navigate } from "react-router-dom";
import { BackButton } from "../components/BackButton";
import { BrandMark } from "../components/BrandMark";
import { hasSupabaseConfig, supabase } from "../lib/supabase";
import { errorMessage } from "../lib/errors";
import type { Session } from "@supabase/supabase-js";

type AuthMode = "login" | "register";

const authCopy = {
  login: {
    title: "Welcome back",
    subtitle: "Pick up where your taste left off.",
    primary: "Sign in",
    switchKicker: "New here?",
    switchTitle: "Build a movie profile in a few ratings.",
    switchButton: "Get started",
    switchTo: "/register",
  },
  register: {
    title: "Start your movie profile",
    subtitle: "Rate a few favorites. FilmFlicker handles the picks.",
    primary: "Start matching",
    switchKicker: "Already have picks?",
    switchTitle: "Jump back into your movie queue.",
    switchButton: "Sign in",
    switchTo: "/signin",
  },
};

function AuthPage({ session, mode }: { session: Session | null; mode: AuthMode }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const copy = authCopy[mode];

  if (session) return <Navigate to="/app/browse" replace />;

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!supabase) {
      setMessage("Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to frontend/.env.local.");
      return;
    }
    setLoading(true);
    setMessage(null);
    try {
      const result =
        mode === "login"
          ? await supabase.auth.signInWithPassword({ email, password })
          : await supabase.auth.signUp({ email, password });
      if (result.error) {
        setMessage(result.error.message);
        return;
      }
      if (mode === "register" && !result.data.session) {
        setMessage("Registration created. Check your email if confirmation is enabled in Supabase.");
      }
    } catch (error) {
      setMessage(errorMessage(error, "Authentication is temporarily unavailable."));
    } finally {
      setLoading(false);
    }
  }

  async function signInWithGoogle() {
    if (!supabase) {
      setMessage("Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to frontend/.env.local.");
      return;
    }
    setLoading(true);
    setMessage(null);
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: {
          redirectTo: `${window.location.origin}/app/browse`,
        },
      });
      if (error) setMessage(error.message);
    } catch (error) {
      setMessage(errorMessage(error, "Google sign-in is temporarily unavailable."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <aside className="auth-side" aria-hidden="true">
        <p className="eyebrow">Movie night, tuned</p>
        <h2>Find the film that actually fits your mood.</h2>
        <div className="auth-taste-strip">
          <span><Sparkles size={15} /> Smart picks</span>
          <span>Private taste</span>
          <span>Five models</span>
        </div>
      </aside>
      <form className="auth-card" onSubmit={submit}>
        <BackButton fallback="/" />
        <div className="auth-brand">
          <span className="brand-mark"><BrandMark /></span>
          <span>FilmFlicker</span>
        </div>
        <p className="auth-kicker">{mode === "login" ? "Now showing" : "First picks"}</p>
        <h1>{copy.title}</h1>
        <p>{copy.subtitle}</p>
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
        <button className="primary-button" disabled={loading}>{loading ? "Working..." : copy.primary} <ArrowRight size={17} /></button>
        <div className="auth-switch-panel">
          <div>
            <span>{copy.switchKicker}</span>
            <strong>{copy.switchTitle}</strong>
          </div>
          <Link className="auth-switch-button" to={copy.switchTo}>
            {mode === "login" ? <UserPlus size={17} /> : <ArrowRight size={17} />}
            {copy.switchButton}
          </Link>
        </div>
      </form>
    </div>
  );
}

export function SignInPage({ session }: { session: Session | null }) {
  return <AuthPage session={session} mode="login" />;
}

export function RegisterPage({ session }: { session: Session | null }) {
  return <AuthPage session={session} mode="register" />;
}
