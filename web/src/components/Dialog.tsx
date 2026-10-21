"use client";

// Accessible modal built on Radix Dialog, styled with our own .backdrop/.dialog classes.
// Radix moves focus in, keeps Tab inside, hides the page behind from screen readers,
// stops it scrolling, and closes on Escape or a click outside. Focus goes back to
// whatever opened it (we open dialogs from plain buttons, not Radix triggers).
import { Dialog as Radix } from "radix-ui";
import { useRef } from "react";
import type { ReactNode } from "react";

export function Dialog({ open, title, onClose, children, footer }: {
  open: boolean; title: string; onClose: () => void; children: ReactNode; footer?: ReactNode;
}) {
  const opener = useRef<HTMLElement | null>(null);
  return (
    <Radix.Root open={open} onOpenChange={(next) => { if (!next) onClose(); }}>
      <Radix.Portal>
        <Radix.Overlay className="backdrop">
          <Radix.Content className="dialog" aria-describedby={undefined}
            onOpenAutoFocus={(e) => {
              opener.current = document.activeElement as HTMLElement | null;
              const preferred = (e.currentTarget as HTMLElement | null)?.querySelector<HTMLElement>("[data-autofocus]");
              if (preferred) { e.preventDefault(); preferred.focus(); }
            }}
            onCloseAutoFocus={(e) => {
              const back = opener.current;
              if (back?.isConnected) { e.preventDefault(); back.focus(); }
            }}>
            <div className="dialog-head">
              <Radix.Title className="h2">{title}</Radix.Title>
              <Radix.Close className="icon-btn" aria-label="Close">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
              </Radix.Close>
            </div>
            <div className="dialog-body">{children}</div>
            {footer && <div className="dialog-foot">{footer}</div>}
          </Radix.Content>
        </Radix.Overlay>
      </Radix.Portal>
    </Radix.Root>
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
