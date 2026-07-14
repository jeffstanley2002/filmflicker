import { LogOut, X } from "lucide-react";

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
  if (!open) return null;

  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <div className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title" onMouseDown={(event) => event.stopPropagation()}>
        <button type="button" className="dialog-close" onClick={onCancel} aria-label="Close dialog">
          <X size={18} />
        </button>
        <div className="confirm-icon"><LogOut size={22} /></div>
        <h2 id="confirm-title">{title}</h2>
        <p>{body}</p>
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
