import { AlertTriangle, RefreshCw } from "lucide-react";
import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

type Props = { children: ReactNode };
type State = { failed: boolean };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("FilmFlicker render failure", error, info);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <main className="fatal-error" role="alert">
        <AlertTriangle size={30} />
        <h1>FilmFlicker could not open this view</h1>
        <p>Refresh the app to reconnect and try again.</p>
        <button className="primary-button" type="button" onClick={() => window.location.reload()}>
          <RefreshCw size={17} /> Refresh
        </button>
      </main>
    );
  }
}
