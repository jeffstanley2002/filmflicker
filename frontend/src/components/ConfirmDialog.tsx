import { LogOut, X } from "lucide-react";
import { useId, useRef } from "react";
import { useModalFocus } from "../lib/useModalFocus";

type Props = {
  open: boolean;
  busy?: boolean;
  title: string;
  body: string;
  confirmLabel: string;
  onCancel: () => void;
  onConfirm: () => void;
};

export function ConfirmDialog({ open, busy = false, title, body, confirmLabel, onCancel, onConfirm }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const bodyId = useId();
  useModalFocus(open, dialogRef, onCancel, busy);
  if (!open) return null;

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={() => { if (!busy) onCancel(); }}>
      <div ref={dialogRef} className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={bodyId} onMouseDown={(event) => event.stopPropagation()}>
        <button type="button" className="dialog-close" onClick={onCancel} aria-label="Close dialog" disabled={busy}>
          <X size={18} />
        </button>
        <div className="confirm-icon"><LogOut size={22} /></div>
        <h2 id={titleId}>{title}</h2>
        <p id={bodyId}>{body}</p>
        <div className="dialog-actions">
          <button type="button" className="secondary-button" onClick={onCancel} disabled={busy}>Stay here</button>
          <button type="button" className="danger-button" onClick={onConfirm} disabled={busy}>
            {busy ? "Signing out..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
