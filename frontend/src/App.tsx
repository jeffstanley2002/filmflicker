import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Session } from "@supabase/supabase-js";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { BackendWakeNotice, type BackendWakeState } from "./components/BackendWakeNotice";
import { Loading } from "./components/Loading";
import { warmApi } from "./lib/api";
import { supabase } from "./lib/supabase";

const AnalyticsPage = lazy(() => import("./pages/AnalyticsPage").then((module) => ({ default: module.AnalyticsPage })));
const RegisterPage = lazy(() => import("./pages/AuthPage").then((module) => ({ default: module.RegisterPage })));
const SignInPage = lazy(() => import("./pages/AuthPage").then((module) => ({ default: module.SignInPage })));
const BrowsePage = lazy(() => import("./pages/BrowsePage").then((module) => ({ default: module.BrowsePage })));
const LandingPage = lazy(() => import("./pages/LandingPage").then((module) => ({ default: module.LandingPage })));
const RecommendationsPage = lazy(() => import("./pages/RecommendationsPage").then((module) => ({ default: module.RecommendationsPage })));
const SystemDesignPage = lazy(() => import("./pages/SystemDesignPage").then((module) => ({ default: module.SystemDesignPage })));
const WatchlistPage = lazy(() => import("./pages/WatchlistPage").then((module) => ({ default: module.WatchlistPage })));
const WatchedPage = lazy(() => import("./pages/WatchedPage").then((module) => ({ default: module.WatchedPage })));
const CheckEmailPage = lazy(() => import("./pages/AuthPage").then((module) => ({ default: module.CheckEmailPage })));

function isConfirmedSession(nextSession: Session | null) {
  if (!nextSession) return false;
  const provider = nextSession.user.app_metadata?.provider;
  const providers = nextSession.user.app_metadata?.providers;
  const usesEmailPassword = provider === "email" || (Array.isArray(providers) && providers.includes("email"));
  return !usesEmailPassword || Boolean(nextSession.user.email_confirmed_at);
}

function displayNameFromSession(nextSession: Session | null) {
  const metadata = nextSession?.user.user_metadata ?? {};
  const metadataName = typeof metadata.display_name === "string" ? metadata.display_name.trim() : "";
  const metadataUsername = typeof metadata.username === "string" ? metadata.username.trim() : "";
  const emailPrefix = nextSession?.user.email?.split("@")[0] ?? "Movie friend";
  return metadataName || metadataUsername || emailPrefix || "Movie friend";
}

function isSignupConfirmationRedirect() {
  const params = new URLSearchParams(window.location.search);
  const hashParams = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  return window.location.pathname === "/signin" && (
    params.get("type") === "signup" || hashParams.get("type") === "signup"
  );
}

function RequireAuth({ session, children }: { session: Session | null; children: ReactNode }) {
  if (!session) return <Navigate to="/signin" replace />;
  return children;
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [backendWakeState, setBackendWakeState] = useState<BackendWakeState>("idle");
  const clearingInvalidSession = useRef(false);
  const lastWakePing = useRef(0);
  const navigate = useNavigate();

  const startBackendWarmup = useCallback((signal?: AbortSignal) => {
    const slowTimer = window.setTimeout(() => setBackendWakeState("slow"), 2_500);
    setBackendWakeState("warming");
    return warmApi(signal)
      .then(() => {
        if (!signal?.aborted) setBackendWakeState("ready");
      })
      .catch(() => {
        if (!signal?.aborted) setBackendWakeState("slow");
      })
      .finally(() => {
        window.clearTimeout(slowTimer);
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void startBackendWarmup(controller.signal);
    return () => {
      controller.abort();
    };
  }, [startBackendWarmup]);

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }
    const client = supabase;
    client.auth.getSession()
      .then(async ({ data }) => {
        if (data.session && !isConfirmedSession(data.session)) {
          await client.auth.signOut({ scope: "local" });
          setSession(null);
          navigate("/check-email", { replace: true, state: { email: data.session.user.email } });
          return;
        }
        if (data.session && isSignupConfirmationRedirect()) {
          await client.auth.signOut({ scope: "local" });
          setSession(null);
          navigate("/signin", { replace: true });
          return;
        }
        setSession(data.session);
      })
      .catch(() => setSession(null))
      .finally(() => setLoading(false));
    const { data } = client.auth.onAuthStateChange((_event, nextSession) => {
      if (nextSession && !isConfirmedSession(nextSession)) {
        setSession(null);
        void client.auth.signOut({ scope: "local" });
        navigate("/check-email", { replace: true, state: { email: nextSession.user.email } });
        return;
      }
      if (nextSession && isSignupConfirmationRedirect()) {
        setSession(null);
        void client.auth.signOut({ scope: "local" });
        navigate("/signin", { replace: true });
        return;
      }
      setSession(nextSession);
    });
    return () => data.subscription.unsubscribe();
  }, [navigate]);

  useEffect(() => {
    function unauthorized() {
      if (clearingInvalidSession.current) return;
      clearingInvalidSession.current = true;
      void (async () => {
        try {
          await supabase?.auth.signOut({ scope: "local" });
        } finally {
          setSession(null);
          navigate("/signin", { replace: true });
          clearingInvalidSession.current = false;
        }
      })();
    }
    window.addEventListener("filmflicker:unauthorized", unauthorized);
    return () => window.removeEventListener("filmflicker:unauthorized", unauthorized);
  }, [navigate]);

  useEffect(() => {
    if (!session) return;
    const controller = new AbortController();
    const wakeIfStale = () => {
      if (document.visibilityState === "hidden") return;
      const now = Date.now();
      if (now - lastWakePing.current < 60_000) return;
      lastWakePing.current = now;
      void supabase?.auth.getSession().then(({ data }) => {
        if (!controller.signal.aborted) setSession(isConfirmedSession(data.session) ? data.session : null);
      });
      void startBackendWarmup(controller.signal);
    };

    wakeIfStale();
    window.addEventListener("focus", wakeIfStale);
    document.addEventListener("visibilitychange", wakeIfStale);
    return () => {
      controller.abort();
      window.removeEventListener("focus", wakeIfStale);
      document.removeEventListener("visibilitychange", wakeIfStale);
    };
  }, [session, startBackendWarmup]);

  async function signOut() {
    const result = await supabase?.auth.signOut();
    if (result?.error) throw result.error;
    setSession(null);
    navigate("/");
  }

  if (loading) {
    return (
      <>
        <Loading label="Opening FilmFlicker" />
        <BackendWakeNotice state={backendWakeState} />
      </>
    );
  }

  const token = session?.access_token ?? "";
  const email = session?.user.email ?? "Signed in";
  const displayName = displayNameFromSession(session);

  async function updateDisplayName(nextName: string) {
    if (!supabase) return;
    const clean = nextName.trim();
    const { data, error } = await supabase.auth.updateUser({
      data: {
        display_name: clean || null,
        username: clean || null,
      },
    });
    if (error) throw error;
    if (session && data.user) setSession({ ...session, user: data.user });
  }

  return (
    <Suspense fallback={null}>
    <BackendWakeNotice state={backendWakeState} />
    <Routes>
      <Route path="/" element={session ? <Navigate to="/app/browse" replace /> : <LandingPage />} />
      <Route path="/auth" element={<Navigate to={session ? "/app/browse" : "/signin"} replace />} />
      <Route path="/signin" element={session ? <Navigate to="/app/browse" replace /> : <SignInPage session={session} />} />
      <Route path="/register" element={session ? <Navigate to="/app/browse" replace /> : <RegisterPage session={session} />} />
      <Route path="/check-email" element={<CheckEmailPage session={session} />} />
      <Route path="/system-design" element={session ? <Navigate to="/app/browse" replace /> : <SystemDesignPage />} />
      <Route
        path="/app"
        element={
          <RequireAuth session={session}>
            <AppShell email={email} displayName={displayName} token={token} onSignOut={signOut} onUpdateDisplayName={updateDisplayName} />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/app/browse" replace />} />
        <Route path="browse" element={<BrowsePage token={token} />} />
        <Route path="recommendations" element={<RecommendationsPage token={token} />} />
        <Route path="watchlist" element={<WatchlistPage token={token} />} />
        <Route path="watched" element={<WatchedPage token={token} />} />
        <Route path="analytics" element={<AnalyticsPage token={token} />} />
      </Route>
      <Route path="*" element={<Navigate to={session ? "/app/browse" : "/"} replace />} />
    </Routes>
    </Suspense>
  );
}
