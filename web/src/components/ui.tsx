"use client";

import Link from "@/components/Link";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { useBillingInterval } from "@/components/BillingInterval";

import { ApiError } from "@/api/client";
import { freshness, etTime } from "@/lib/format";
import { beginBilling } from "@/lib/billingReturn";
import { captureUsage } from "@/lib/usageAnalytics";
import { readCachedMe } from "@/session/meCache";
import { checkout } from "@/lib/checkout";

export function Card({ title, aside, children, id, className = "" }: { title?: ReactNode; aside?: ReactNode; children: ReactNode; id?: string; className?: string }) {
  const hid = id ? `${id}-h` : undefined;
  return (
    <section id={id} className={`card ${className}`} aria-labelledby={hid}>
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

/** The ranked list's floor: names below it aren't ranked setups, so their score is shown muted. */
export const RANKED_MIN = 40;

export function ScoreBadge({ score }: { score: number | null | undefined }) {
  const weak = score === null || score === undefined || score < RANKED_MIN;
  return <span className={`score${weak ? " score-weak" : ""}`} aria-label={`HSF Score ${score ?? "unavailable"}`}>{score ?? "--"}</span>;
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
/** Only an explicit server missed-slot flag labels a scan delayed. */
export function Freshness({ at, label = "Scan", staleCheck = true, stale }: { at: string | null | undefined; label?: string; staleCheck?: boolean; stale?: boolean | null }) {
  const raw = freshness(at);
  // Age describes a snapshot; only a missed-slot flag establishes a delayed scan.
  const f = { ...raw, stale: staleCheck && stale === true };
  return (
    <span className={`fresh${f.stale ? " stale" : ""}`} title={at ? new Date(at).toISOString() : undefined}>
      {label} {etTime(at)} · {f.label} · {f.stale ? "Scan delayed" : stale === false ? "On schedule" : "Saved snapshot"}
    </span>
  );
}

/** After this long, say why: the API's host sleeps when idle and takes up to a minute to wake. */
export const SLOW_AFTER_MS = 8000;
export const WAKING_UP = "Still loading. The HSF service may be waking up, which can take up to a minute.";

export function useSlow(active: boolean, afterMs = SLOW_AFTER_MS): boolean {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    if (!active) return;
    const t = setTimeout(() => setSlow(true), afterMs);
    return () => { clearTimeout(t); setSlow(false); };
  }, [active, afterMs]);
  return active && slow;
}

export function Skeleton({ rows = 4, label = "Loading" }: { rows?: number; label?: string }) {
  const slow = useSlow(true);
  return (
    <div className="skeleton" role="status" aria-live="polite">
      <span className="sr-only">{label}…</span>
      {slow && <p className="cap">{WAKING_UP}</p>}
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

function RetryButton({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  const [remaining, setRemaining] = useState(error.retryAfterS ?? 0);
  useEffect(() => {
    if (!error.retryAfterS) return;
    const deadline = error.receivedAt + error.retryAfterS * 1000;
    const timer = setInterval(() => setRemaining(Math.max(0, Math.ceil((deadline - Date.now()) / 1000))), 1000);
    return () => clearInterval(timer);
  }, [error]);
  return <button type="button" className="btn" onClick={onRetry} disabled={remaining > 0}>
    {remaining > 0 ? `Try again in ${remaining}s` : "Try again"}
  </button>;
}

export function ErrorState({ error, onRetry, what = "this" }: { error: ApiError; onRetry?: () => void; what?: string }) {
  const outage = error.status === 503 || error.status === 502 || error.status === 504 || error.status === 0;
  return (
    <div className="error" role="alert">
      <p className="strong">{outage ? `Couldn't load ${what} right now` : error.message}</p>
      {outage && <p className="cap">{error.message}{error.retryAfterS ? ` Try again in ${error.retryAfterS}s.` : ""}</p>}
      <div className="row-actions">
        {onRetry && <RetryButton key={error.receivedAt} error={error} onRetry={onRetry} />}
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

export function UpgradeButton({ plan, label }: { plan: "pro" | "premium"; label?: string }) {
  const [interval] = useBillingInterval();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<ApiError | null>(null);
  const go = async () => {
    setBusy(true);
    setErr(null);
    try {
      const link = await checkout(plan, interval);
      captureUsage("checkout_started", { plan, interval });
      beginBilling(readCachedMe()?.plan ?? null, plan);
      window.location.assign(link.url);
    } catch (e) {
      setErr(e instanceof ApiError ? e : null);
      setBusy(false);
    }
  };
  return (
    <div className="upgrade">
      <button type="button" className="btn btn-primary" onClick={go} disabled={busy}>
        {busy ? "Opening checkout…" : label ?? `Upgrade to ${plan === "pro" ? "Pro" : "Premium"}`}
      </button>
      <p className="cap">{interval === "year" ? "Yearly billing" : "Monthly billing"}</p>
      {err && <p className="cap" role="alert">{err.message}{err.requestId ? ` (support code ${err.requestId})` : ""}</p>}
    </div>
  );
}

export function TickerLink({ ticker }: { ticker: string }) {
  return <Link href={`/stocks/${encodeURIComponent(ticker)}`} className="tk">{ticker}</Link>;
}

export function ResearchNotice({ compact = false }: { compact?: boolean }) {
  return (
    <p className={compact ? "cap" : "notice"} role={compact ? undefined : "note"}>
      HSF highlights research signals from the available scan data. It does not predict profit, recommend entries or exits, or replace your own risk review.
    </p>
  );
}

/** A failed action's message, with the support code when the server gave one. */
export function ErrorLine({ error }: { error: ApiError | null }) {
  if (!error) return null;
  return (
    <p className="form-error" role="alert">
      {error.message}
      {Object.entries(error.fields).map(([field, message]) => <span className="field-error" key={field}> {field.replaceAll("_", " ")}: {message}</span>)}
      {error.retryAfterS ? <span> Wait {error.retryAfterS}s before retrying.</span> : null}
      {error.requestId && <span className="cap mono"> · Support code: {error.requestId}</span>}
    </p>
  );
}

/** Request freshness is separate from the API's market-data timestamps. */
export function RefreshStatus({ updatedAt, refreshing, error, onRetry, sourceNote = "Market prices may be older." }: {
  updatedAt: number | null; refreshing: boolean; error: ApiError | null; onRetry: () => void;
  sourceNote?: string;
}) {
  return <div className="cap" aria-live="polite">
    {refreshing ? <p role="status">Refreshing… Showing the last available data.</p>
      : updatedAt ? <p>Last checked {etTime(new Date(updatedAt).toISOString())}. {sourceNote}</p> : null}
    {error && <div className="banner">
      <p>Couldn&apos;t refresh. Showing the last available data.</p>
      <ErrorLine error={error} />
      <RetryButton key={error.receivedAt} error={error} onRetry={onRetry} />
    </div>}
  </div>;
}
