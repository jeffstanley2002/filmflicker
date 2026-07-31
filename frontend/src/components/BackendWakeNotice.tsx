import { Server, Zap } from "lucide-react";

export type BackendWakeState = "idle" | "warming" | "slow" | "ready";

export function BackendWakeNotice({ state }: { state: BackendWakeState }) {
  if (state !== "slow") return null;

  return (
    <div className="backend-wake-notice" role="status" aria-live="polite">
      <span className="backend-wake-icon" aria-hidden="true">
        <Server size={18} />
      </span>
      <div>
        <strong>Portfolio backend is waking up</strong>
        <span>FilmFlicker runs on Render's free tier, so the first request after idle time can take about a minute.</span>
      </div>
      <Zap size={17} aria-hidden="true" />
    </div>
  );
}
