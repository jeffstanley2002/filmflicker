import { useEffect } from "react";
import type { RefObject } from "react";

const focusableSelector = [
  "button:not([disabled])",
  "a[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export function useModalFocus(open: boolean, container: RefObject<HTMLElement | null>, onClose: () => void, busy = false) {
  useEffect(() => {
    if (!open || !container.current) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const modal = container.current;
    const focusable = () => Array.from(modal.querySelectorAll<HTMLElement>(focusableSelector));
    focusable()[0]?.focus();

    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape" && !busy) {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      previouslyFocused?.focus();
    };
  }, [busy, container, onClose, open]);
}
