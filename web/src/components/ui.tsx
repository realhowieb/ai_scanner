"use client";

import Link from "next/link";
import { useState } from "react";
import type { ReactNode } from "react";

import { ApiError, api, unwrap } from "@/api/client";
import { freshness, etTime } from "@/lib/format";

export function Card({ title, aside, children, id, className = "" }: { title?: ReactNode; aside?: ReactNode; children: ReactNode; id?: string; className?: string }) {
  const hid = id ? `${id}-h` : undefined;
  return (
    <section className={`card ${className}`} aria-labelledby={hid}>
      {(title || aside) && (
        <div className="card-head">
          {title && <h2 className="h2" id={hid}>{title}</h2>}
          {aside && <div className="cap">{aside}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function ScoreBadge({ score }: { score: number | null | undefined }) {
  return <span className="score" aria-label={`HSF Score ${score ?? "unavailable"}`}>{score ?? "--"}</span>;
}

export function ScoreBar({ score }: { score: number | null | undefined }) {
  const s = Math.max(0, Math.min(100, score ?? 0));
  return (
    <div className="scorebar">
      <div className="track" aria-hidden="true"><div className="fill" style={{ width: `${s}%` }} /></div>
      <span className="mono strong">{score ?? "--"}</span>
    </div>
  );
}

export function Pill({ children, tone }: { children: ReactNode; tone?: "warn" | "gold" | "up" | "down" }) {
  return <span className={`pill${tone ? ` pill-${tone}` : ""}`}>{children}</span>;
}

/** `staleCheck` is off for session scans (last evening's after-hours scan is meant to be hours old). */
export function Freshness({ at, label = "Scan", staleCheck = true }: { at: string | null | undefined; label?: string; staleCheck?: boolean }) {
  const raw = freshness(at);
  const f = staleCheck ? raw : { ...raw, stale: false };
  return (
    <span className={`fresh${f.stale ? " stale" : ""}`} title={at ? new Date(at).toISOString() : undefined}>
      {label} {etTime(at)} · {f.label}{f.stale ? " · Stale" : ""}
    </span>
  );
}

export function Skeleton({ rows = 4, label = "Loading" }: { rows?: number; label?: string }) {
  return (
    <div className="skeleton" role="status" aria-live="polite">
      <span className="sr-only">{label}…</span>
      {Array.from({ length: rows }, (_, i) => <div key={i} className="sk-row" />)}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <p className="strong">{title}</p>
      {children && <div className="cap">{children}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry, what = "this" }: { error: ApiError; onRetry?: () => void; what?: string }) {
  const outage = error.status === 503 || error.status === 502 || error.status === 504 || error.status === 0;
  return (
    <div className="error" role="alert">
      <p className="strong">{outage ? `Couldn't load ${what} right now` : error.message}</p>
      {outage && <p className="cap">{error.message}{error.retryAfterS ? ` Try again in ${error.retryAfterS}s.` : ""}</p>}
      <div className="row-actions">
        {onRetry && <button type="button" className="btn" onClick={onRetry}>Try again</button>}
        {error.requestId && <span className="cap mono">Support code: {error.requestId}</span>}
      </div>
    </div>
  );
}

export function Locked({ title, plan, children }: { title: string; plan: "pro" | "premium"; children?: ReactNode }) {
  return (
    <div className="locked">
      <div className="locked-head">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></svg>
        <p className="strong">{title}</p>
        <Pill tone="gold">{plan === "pro" ? "Pro" : "Premium"}</Pill>
      </div>
      {children && <p className="cap">{children}</p>}
      <UpgradeButton plan={plan} />
    </div>
  );
}

export function UpgradeButton({ plan }: { plan: "pro" | "premium" }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const go = async () => {
    setBusy(true);
    setErr(null);
    try {
      const link = await unwrap(api.POST("/v1/billing/checkout", { body: { plan, interval: "month" } }));
      window.location.assign(link.url);
    } catch (e) {
      setErr(e instanceof ApiError ? e : null);
      setBusy(false);
    }
  };
  return (
    <div className="upgrade">
      <button type="button" className="btn btn-primary" onClick={go} disabled={busy}>
        {busy ? "Opening checkout…" : `Upgrade to ${plan === "pro" ? "Pro" : "Premium"}`}
      </button>
      {err && <p className="cap" role="alert">{err.message}{err.requestId ? ` (support code ${err.requestId})` : ""}</p>}
    </div>
  );
}

export function TickerLink({ ticker }: { ticker: string }) {
  return <Link href={`/stocks/${encodeURIComponent(ticker)}`} className="tk">{ticker}</Link>;
}

export function Disclaimer() {
  return <p className="cap">HSF Score is an opportunity ranking, not a probability of profit. Educational research only, not financial advice.</p>;
}

/** A failed action's message, with the support code when the server gave one. */
export function ErrorLine({ error }: { error: ApiError | null }) {
  if (!error) return null;
  return (
    <p className="form-error" role="alert">
      {error.message}
      {error.requestId && <span className="cap mono"> · Support code: {error.requestId}</span>}
    </p>
  );
}
