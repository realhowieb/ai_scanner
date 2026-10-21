"use client";

// A small "?" button that opens a short explanation. Unlike a title tooltip it works on
// touch screens and from the keyboard, and screen readers announce it (Radix Popover).
import { Popover } from "radix-ui";
import type { ReactNode } from "react";

export function Hint({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Popover.Root>
      <Popover.Trigger className="hint-btn" aria-label={`About ${label}`}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10" /><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14" /><path d="M12 17.5h.01" />
        </svg>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="hint-pop" side="top" align="center" sideOffset={6} collisionPadding={12}>
          <p className="hint-title">{label}</p>
          <div className="body-sm">{children}</div>
          <Popover.Arrow className="hint-arrow" width={12} height={6} />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
