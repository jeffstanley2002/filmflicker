import { lazy, Suspense, useEffect, useState } from "react";
import type { ReactNode } from "react";
import type { Session } from "@supabase/supabase-js";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Loading } from "./components/Loading";
import { supabase } from "./lib/supabase";

const AnalyticsPage = lazy(() => import("./pages/AnalyticsPage").then((module) => ({ default: module.AnalyticsPage })));
const AuthPage = lazy(() => import("./pages/AuthPage").then((module) => ({ default: module.AuthPage })));
const BrowsePage = lazy(() => import("./pages/BrowsePage").then((module) => ({ default: module.BrowsePage })));
const LandingPage = lazy(() => import("./pages/LandingPage").then((module) => ({ default: module.LandingPage })));
const RecommendationsPage = lazy(() => import("./pages/RecommendationsPage").then((module) => ({ default: module.RecommendationsPage })));
const SystemDesignPage = lazy(() => import("./pages/SystemDesignPage").then((module) => ({ default: module.SystemDesignPage })));
const WatchedPage = lazy(() => import("./pages/WatchedPage").then((module) => ({ default: module.WatchedPage })));

function RequireAuth({ session, children }: { session: Session | null; children: ReactNode }) {
  if (!session) return <Navigate to="/auth" replace />;
  return children;
}

export function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });
    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
    });
    return () => data.subscription.unsubscribe();
  }, []);

  async function signOut() {
    await supabase?.auth.signOut();
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
      <Route path="/auth" element={session ? <Navigate to="/app/browse" replace /> : <AuthPage session={session} />} />
      <Route path="/system-design" element={session ? <Navigate to="/app/browse" replace /> : <SystemDesignPage />} />
      <Route
        path="/app"
        element={
          <RequireAuth session={session}>
            <AppShell email={email} onSignOut={signOut} />
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
