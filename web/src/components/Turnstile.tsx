"use client";

// Cloudflare Turnstile bot check for sign-up and password reset. Shown only when
// NEXT_PUBLIC_TURNSTILE_SITE_KEY is set at build time; the API checks the token when it has
// TURNSTILE_SECRET_KEY. Usually invisible: Cloudflare only asks for a click when unsure.
import { useEffect, useRef, useState } from "react";

type TurnstileApi = {
  render: (el: HTMLElement, options: Record<string, unknown>) => string;
  reset: (id?: string) => void;
  remove: (id: string) => void;
};
declare global {
  interface Window { turnstile?: TurnstileApi }
}

export const TURNSTILE_SITE_KEY = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY?.trim() || "";
const SRC = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
let loading: Promise<TurnstileApi> | null = null;

function loadTurnstile(): Promise<TurnstileApi> {
  if (window.turnstile) return Promise.resolve(window.turnstile);
  loading ??= new Promise<TurnstileApi>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = SRC;
    script.async = true;
    script.onload = () => (window.turnstile ? resolve(window.turnstile) : reject(new Error("turnstile missing")));
    script.onerror = () => { loading = null; reject(new Error("turnstile blocked")); };
    document.head.appendChild(script);
  });
  return loading;
}

/** Calls onToken with a one-time token (or null when it expires). Change `resetKey` after a
 * failed submit: each token works once, so the widget must issue a new one. */
export function Turnstile({ action, onToken, resetKey = 0 }: { action: string; onToken: (token: string | null) => void; resetKey?: number }) {
  const box = useRef<HTMLDivElement>(null);
  const widget = useRef<string | null>(null);
  const callback = useRef(onToken);
  const [failed, setFailed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { callback.current = onToken; }, [onToken]);

  useEffect(() => {
    if (!TURNSTILE_SITE_KEY || !box.current) return;
    let cancelled = false;
    loadTurnstile().then((ts) => {
      if (cancelled || !box.current) return;
      widget.current = ts.render(box.current, {
        sitekey: TURNSTILE_SITE_KEY, action, appearance: "interaction-only",
        callback: (t: string) => { setError(null); callback.current(t); },
        "expired-callback": () => callback.current(null),
        // With interaction-only the widget stays hidden when it fails (wrong hostname or key,
        // blocked iframe), so say so instead of leaving a dead submit button.
        "error-callback": (code?: string | number) => { callback.current(null); setError(String(code ?? "unknown")); },
        "unsupported-callback": () => { callback.current(null); setError("unsupported browser"); },
      });
    }).catch(() => { if (!cancelled) setFailed(true); });
    return () => {
      cancelled = true;
      if (widget.current && window.turnstile) window.turnstile.remove(widget.current);
      widget.current = null;
    };
  }, [action]);

  useEffect(() => {
    if (resetKey && widget.current && window.turnstile) {
      callback.current(null);
      window.turnstile.reset(widget.current);
    }
  }, [resetKey]);

  if (!TURNSTILE_SITE_KEY) return null;
  return <>
    <div ref={box} className="turnstile" />
    {failed && <p className="form-error" role="alert">The security check couldn&apos;t load. Turn off content blockers for this site, then reload.</p>}
    {!failed && error && <p className="form-error" role="alert">The security check didn&apos;t work on this page (Cloudflare error {error}). Reload and try again; if it keeps happening, contact support.</p>}
  </>;
}
