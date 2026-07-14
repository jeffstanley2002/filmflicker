import { BarChart3, ChevronLeft, ChevronRight, Film, Heart, LogOut, Radar, Search, Sparkles } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { ConfirmDialog } from "./ConfirmDialog";

export function AppShell({ email, onSignOut }: { email: string; onSignOut: () => void }) {
  const [collapsed, setCollapsed] = useState(false);
  const [peeking, setPeeking] = useState(false);
  const [confirmingSignOut, setConfirmingSignOut] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const name = email.split("@")[0] || "Movie friend";

  async function confirmSignOut() {
    setSigningOut(true);
    await onSignOut();
    setSigningOut(false);
  }

  return (
    <div className={`app-shell${collapsed ? " sidebar-collapsed" : ""}${peeking ? " sidebar-peeking" : ""}`}>
      <aside className="sidebar" aria-label="Primary navigation" onMouseLeave={() => setPeeking(false)}>
        <button type="button" className="sidebar-toggle" onClick={() => { setPeeking(false); setCollapsed((current) => !current); }} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
        </button>
        <NavLink to="/app/browse" className="brand">
          <span className="brand-mark"><Film size={22} /></span>
          <span>CineMatch</span>
        </NavLink>
        <nav>
          <NavLink to="/app/browse"><Search size={18} /><span>Browse</span></NavLink>
          <NavLink to="/app/recommendations"><Radar size={18} /><span>For You</span></NavLink>
          <NavLink to="/app/watched"><Heart size={18} /><span>Watched</span></NavLink>
          <NavLink to="/app/analytics"><BarChart3 size={18} /><span>Taste</span></NavLink>
        </nav>
        <div className="sidebar-user">
          <div className="profile-card">
            <div className="profile-avatar"><Sparkles size={16} /></div>
            <div>
              <strong>{name}</strong>
              <span>Level {Math.max(1, name.length)} taste scout</span>
            </div>
          </div>
          <button className="ghost-button" onClick={() => setConfirmingSignOut(true)}><LogOut size={16} /><span>Sign out</span></button>
        </div>
        {collapsed ? <div className="sidebar-reveal" onMouseEnter={() => setPeeking(true)} aria-hidden="true" /> : null}
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
      <ConfirmDialog
        open={confirmingSignOut}
        busy={signingOut}
        title="Sign out of CineMatch?"
        body="Your ratings and watched list are saved. You can jump back in anytime."
        confirmLabel="Sign out"
        onCancel={() => setConfirmingSignOut(false)}
        onConfirm={confirmSignOut}
      />
    </div>
  );
}
