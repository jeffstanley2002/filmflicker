import { lazy, Suspense, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Session } from "@supabase/supabase-js";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { AppShell } from "./components/AppShell";
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
const WatchedPage = lazy(() => import("./pages/WatchedPage").then((module) => ({ default: module.WatchedPage })));

function RequireAuth({ session, children }: { session: Session | null; children: ReactNode }) {
  if (!session) return <Navigate to="/signin" replace />;
  return children;
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const clearingInvalidSession = useRef(false);
  const lastWakePing = useRef(0);
  const navigate = useNavigate();

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }
    supabase.auth.getSession()
      .then(({ data }) => setSession(data.session))
      .catch(() => setSession(null))
      .finally(() => setLoading(false));
    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
    });
    return () => data.subscription.unsubscribe();
  }, []);

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
    window.addEventListener("cinematch:unauthorized", unauthorized);
    return () => window.removeEventListener("cinematch:unauthorized", unauthorized);
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
        if (!controller.signal.aborted) setSession(data.session);
      });
      void warmApi(controller.signal).catch(() => {
        // The next page request will show a user-facing message if Render is still waking.
      });
    };

    wakeIfStale();
    window.addEventListener("focus", wakeIfStale);
    document.addEventListener("visibilitychange", wakeIfStale);
    return () => {
      controller.abort();
      window.removeEventListener("focus", wakeIfStale);
      document.removeEventListener("visibilitychange", wakeIfStale);
    };
  }, [session]);

  async function signOut() {
    const result = await supabase?.auth.signOut();
    if (result?.error) throw result.error;
    setSession(null);
    navigate("/");
  }

  if (loading) return <Loading label="Opening CineMatch" />;

  const token = session?.access_token ?? "";
  const email = session?.user.email ?? "Signed in";

  return (
    <Suspense fallback={<Loading label="Loading page" />}>
    <Routes>
      <Route path="/" element={session ? <Navigate to="/app/browse" replace /> : <LandingPage />} />
      <Route path="/auth" element={<Navigate to={session ? "/app/browse" : "/signin"} replace />} />
      <Route path="/signin" element={session ? <Navigate to="/app/browse" replace /> : <SignInPage session={session} />} />
      <Route path="/register" element={session ? <Navigate to="/app/browse" replace /> : <RegisterPage session={session} />} />
      <Route path="/system-design" element={session ? <Navigate to="/app/browse" replace /> : <SystemDesignPage />} />
      <Route
        path="/app"
        element={
          <RequireAuth session={session}>
            <AppShell email={email} token={token} onSignOut={signOut} />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/app/browse" replace />} />
        <Route path="browse" element={<BrowsePage token={token} />} />
        <Route path="recommendations" element={<RecommendationsPage token={token} />} />
        <Route path="watched" element={<WatchedPage token={token} />} />
        <Route path="analytics" element={<AnalyticsPage token={token} />} />
      </Route>
      <Route path="*" element={<Navigate to={session ? "/app/browse" : "/"} replace />} />
    </Routes>
    </Suspense>
  );
}
