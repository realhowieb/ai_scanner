"use client";

// Accessible modal: role="dialog" + aria-modal, focus moves in on open and back to
// the opener on close, Tab stays inside, Escape and the backdrop close it.
import { useEffect, useId, useRef } from "react";
import type { ReactNode } from "react";

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function Dialog({ open, title, onClose, children, footer }: {
  open: boolean; title: string; onClose: () => void; children: ReactNode; footer?: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const opener = useRef<Element | null>(null);
  const titleId = useId();
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  });

  useEffect(() => {
    if (!open) return;
    opener.current = document.activeElement;
    const box = ref.current;
    const first = box?.querySelector<HTMLElement>("[data-autofocus]") || box?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab" || !box) return;
      const items = Array.from(box.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (!items.length) return;
      const firstEl = items[0]!;
      const lastEl = items[items.length - 1]!;
      if (e.shiftKey && document.activeElement === firstEl) {
        e.preventDefault();
        lastEl.focus();
      } else if (!e.shiftKey && document.activeElement === lastEl) {
        e.preventDefault();
        firstEl.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      (opener.current as HTMLElement | null)?.focus?.();
    };
  }, [open]);

  if (!open) return null;
  return (
    <div className="backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} ref={ref}>
        <div className="dialog-head">
          <h2 className="h2" id={titleId}>{title}</h2>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
          </button>
        </div>
        <div className="dialog-body">{children}</div>
        {footer && <div className="dialog-foot">{footer}</div>}
      </div>
    </div>
  );
}

/** "Are you sure?" for destructive actions. Nothing changes until onConfirm succeeds. */
export function ConfirmDialog({ open, title, body, confirmLabel, busy, error, onConfirm, onClose }: {
  open: boolean; title: string; body: ReactNode; confirmLabel: string; busy: boolean;
  error?: ReactNode; onConfirm: () => void; onClose: () => void;
}) {
  return (
    <Dialog open={open} title={title} onClose={onClose}
      footer={<>
        <button type="button" className="btn" onClick={onClose} data-autofocus>Keep it</button>
        <button type="button" className="btn btn-danger" onClick={onConfirm} disabled={busy}>{busy ? "Working…" : confirmLabel}</button>
      </>}>
      <div className="stack-sm">{body}{error}</div>
    </Dialog>
  );
}
