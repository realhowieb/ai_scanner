"use client";

// Create an alert. The fields, minimums, directions and defaults come from
// GET /v1/alerts/types (the rules POST /v1/alerts enforces); the server's own answer
// (403 plan limit, 409 duplicate, 422 invalid) is shown as-is.
import { useMemo, useState } from "react";
import type { FormEvent } from "react";

import { alerts } from "@/api/userData";
import type { Alert, AlertCreate, AlertType } from "@/api/userData";
import { TICKER_RE } from "@/components/AppShell";
import { ErrorLine } from "@/components/ui";
import { useAction } from "@/hooks/useAction";

export const DIRECTION_LABELS: Record<string, string> = {
  above: "Rises above", below: "Falls below", bullish: "Bullish (golden cross)", bearish: "Bearish (death cross)",
  up: "Crosses up", down: "Crosses down",
};

const money = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });

export function describeAlert(a: Pick<Alert, "type" | "ticker" | "threshold" | "direction" | "watchlist_only">): string {
  const t = a.ticker ?? "";
  const n = a.threshold ?? 0;
  switch (a.type) {
    case "breakout": return `Breakout Score ≥ ${n} (${a.watchlist_only ? "watchlist tickers only" : "all tickers"})`;
    case "watchlist": return "Any watchlist ticker appears in a scan";
    case "price": return `${t} price ${a.direction === "below" ? "falls below" : "rises above"} $${money(n)}`;
    case "move": return `${t} moves ±${n}% in a day`;
    case "rvol": return `${t} trades at ${n}× its average volume`;
    case "ema_cross": return `${t} EMA 9/21 ${a.direction === "bearish" ? "bearish" : "bullish"} cross`;
    case "ewo_cross": return `${t} EWO crosses ${a.direction === "down" ? "down" : "up"} through zero`;
    default: return a.type;
  }
}

/** The API serving this site predates GET /v1/alerts/types (it ships with the same release). */
export function typesUnavailable(status: number | undefined): boolean {
  return status === 404 || status === 405;
}

export const TYPES_UNAVAILABLE = "Creating alerts here needs the latest HSF API, which isn't deployed yet. Your existing alerts are listed and can be turned on, off or deleted; create new ones in the classic app for now.";

export type AlertPrefill = Partial<Pick<AlertCreate, "type" | "ticker" | "threshold" | "direction">>;

export function validate(spec: AlertType, ticker: string, threshold: string, direction: string): string | null {
  if (spec.needs_ticker && !TICKER_RE.test(ticker.trim())) return "Enter a ticker symbol like AAPL.";
  if (spec.threshold) {
    const v = Number(threshold);
    if (threshold.trim() === "" || !Number.isFinite(v)) return `Enter ${spec.threshold.label.toLowerCase()}.`;
    const { min, min_exclusive: excl, max } = spec.threshold;
    if (excl ? v <= min : v < min) return `${spec.threshold.label} must be ${excl ? "greater than" : "at least"} ${min}.`;
    if (v > max) return `${spec.threshold.label} must be at most ${max.toLocaleString()}.`;
  }
  if (spec.directions.length && !spec.directions.includes(direction)) return "Choose a direction.";
  return null;
}

export function AlertForm({ types, prefill, lockType = false, onCreated, submitLabel = "Create alert" }: {
  types: AlertType[]; prefill?: AlertPrefill; lockType?: boolean; onCreated: (a: Alert) => void; submitLabel?: string;
}) {
  const [type, setType] = useState<string>(prefill?.type ?? types[0]?.type ?? "price");
  const spec = useMemo(() => types.find((t) => t.type === type) ?? types[0], [types, type]);
  const [ticker, setTicker] = useState(prefill?.ticker ?? "");
  const initialSpec = types.find((t) => t.type === (prefill?.type ?? types[0]?.type));
  const [threshold, setThreshold] = useState(prefill?.threshold != null ? String(prefill.threshold)
    : initialSpec?.threshold?.default != null ? String(initialSpec.threshold.default) : "");
  const [direction, setDirection] = useState(prefill?.direction ?? "");
  const [watchlistOnly, setWatchlistOnly] = useState(false);
  const [invalid, setInvalid] = useState<string | null>(null);
  const act = useAction();
  if (!spec) return null;

  const pickType = (t: string) => {
    setType(t);
    const next = types.find((x) => x.type === t);
    setThreshold(next?.threshold?.default != null ? String(next.threshold.default) : "");
    setDirection(next?.directions[0] ?? "");
    setInvalid(null);
    act.clear();
  };
  const dir = direction || spec.directions[0] || "";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    const problem = validate(spec, ticker, threshold, dir);
    setInvalid(problem);
    if (problem) return;
    const body: AlertCreate = {
      type: spec.type,
      ...(spec.needs_ticker ? { ticker: ticker.trim().toUpperCase() } : {}),
      ...(spec.threshold ? { threshold: Number(threshold) } : {}),
      ...(spec.directions.length ? { direction: dir } : {}),
      ...(spec.watchlist_only_option ? { watchlist_only: watchlistOnly } : {}),
    };
    const created = await act.run(() => alerts.create(body));
    if (created) onCreated(created);
  };

  return (
    <form className="stack-sm alert-form" onSubmit={submit} noValidate>
      {!lockType && (
        <label className="field"><span>Alert type</span>
          <select value={spec.type} onChange={(e) => pickType(e.target.value)}>
            {types.map((t) => <option key={t.type} value={t.type}>{t.label}</option>)}
          </select>
        </label>
      )}
      <p className="cap">{spec.description}</p>
      <div className="grid2">
        {spec.needs_ticker && (
          <label className="field"><span>Ticker</span>
            <input value={ticker} maxLength={10} autoComplete="off" onChange={(e) => setTicker(e.target.value)} placeholder="AAPL" />
          </label>
        )}
        {spec.directions.length > 0 && (
          <label className="field"><span>Direction</span>
            <select value={dir} onChange={(e) => setDirection(e.target.value)}>
              {spec.directions.map((d) => <option key={d} value={d}>{DIRECTION_LABELS[d] ?? d}</option>)}
            </select>
          </label>
        )}
        {spec.threshold && (
          <label className="field"><span>{spec.threshold.label}</span>
            <input inputMode="decimal" value={threshold} onChange={(e) => setThreshold(e.target.value)}
              placeholder={spec.threshold.default != null ? String(spec.threshold.default) : ""} />
          </label>
        )}
      </div>
      {spec.watchlist_only_option && (
        <label className="check"><input type="checkbox" checked={watchlistOnly} onChange={(e) => setWatchlistOnly(e.target.checked)} />
          <span>Only tickers on my watchlists</span></label>
      )}
      {invalid && <p className="form-error" role="alert">{invalid}</p>}
      <ErrorLine error={act.error} />
      <div className="row-actions">
        <button type="submit" className="btn btn-primary" disabled={act.busy}>{act.busy ? "Creating…" : submitLabel}</button>
      </div>
    </form>
  );
}
