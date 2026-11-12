"use client";

// Admin > Models: AI Confidence (admins only). Brings back what the old admin showed under
// Scanner results: the model card, the latest scan's scores, calibration buckets, and a
// Claude "why this score" explainer that runs only when asked. All reads are small SQL
// queries on the API; the model itself is never loaded there.
import { Fragment, useMemo, useState } from "react";

import { api, unwrap } from "@/api/client";
import { Card, Empty, ErrorLine, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";

type Tier = "High" | "Medium" | "Low" | "Unknown";
type TrainBucket = { bucket: string; signals: number; mean_confidence: number | null; hit_rate: number | null };
type LiveBucket = { bucket: string; signals: number; mean_confidence: number | null; hits: number; misses: number;
  unclear: number; hit_rate_low: number; hit_rate_high: number };
type Score = { ticker: string; ai_confidence: number | null; tier: Tier; breakout_score: number | null;
  hsf_score: number | null; setup: string | null; chg_pct: number | null };
export type AiConfidenceSummary = {
  model: { model_version: string; trained_at: string | null; saved_at: string | null; dataset_version: string | null;
    target_rule: string | null; features: number | null; calibration_method: string | null; calibration_points: number;
    training_buckets: TrainBucket[] } | null;
  warnings: string[];
  latest: { scanned_at: string | null; rows: Score[]; tiers: Record<"High" | "Medium" | "Low", number> };
  live: { days: number; matured: number; pending: number; buckets: LiveBucket[] };
  note: string;
};
type Explain = { ticker: string; ai_confidence: number | null; tier: string; earnings_in_days: number | null;
  features: Record<string, number>; explanation: string | null; error: string | null };

const pct = (v: number | null | undefined, digits = 1) => (v == null ? "—" : `${v.toFixed(digits)}%`);
const share = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const fixed = (v: number | null | undefined, digits = 0) => (v == null ? "—" : v.toFixed(digits));
const tone = (t: Tier) => (t === "High" ? "up" : t === "Medium" ? "gold" : undefined);
const TARGET_FALLBACK = "+4% before −2% within 5 trading days";
type SortKey = "ai_confidence" | "ticker" | "hsf_score" | "breakout_score";

export function AiConfidenceCard() {
  const v = useApi("admin-ai-confidence", (signal) =>
    unwrap(api.GET("/v1/admin/ml/ai-confidence", { signal })) as unknown as Promise<AiConfidenceSummary>);
  return (
    <Card title="AI Confidence" aside={<button className="btn btn-sm" onClick={v.reload} disabled={v.refreshing}>Refresh</button>}>
      {v.error ? <ErrorState error={v.error} onRetry={v.reload} what="AI Confidence" />
        : !v.data ? <Skeleton rows={4} label="Loading AI Confidence" />
          : <AiConfidenceBody d={v.data} />}
    </Card>
  );
}

export function AiConfidenceBody({ d }: { d: AiConfidenceSummary }) {
  const m = d.model;
  return (
    <div className="stack">
      <section aria-label="Model status" className="stack-sm">
        <p className="cap">Admin only: customers don&apos;t see this score. It estimates the chance a setup goes {TARGET_FALLBACK}.</p>
        {m ? <>
          <p>Active model: <strong className="mono">{m.model_version}</strong> {m.calibration_method ? <Pill tone="up">Calibrated ({m.calibration_method})</Pill> : <Pill tone="warn">Not calibrated</Pill>}</p>
          <p className="cap">Trained {etTime(m.trained_at)} · saved {etTime(m.saved_at)} · dataset {m.dataset_version ?? "not recorded"} · {m.features ?? "?"} inputs</p>
          {m.target_rule && <p className="cap">Target: <span className="mono">{m.target_rule}</span></p>}
        </> : <p><Pill tone="warn">No database model</Pill></p>}
        {d.warnings.length > 0 && <ul className="stack-sm" aria-label="AI Confidence warnings">
          {d.warnings.map((w) => <li key={w} className="notice">{w}</li>)}
        </ul>}
      </section>
      <LatestScores latest={d.latest} />
      <Calibration d={d} />
      <p className="cap">{d.note}</p>
    </div>
  );
}

function LatestScores({ latest }: { latest: AiConfidenceSummary["latest"] }) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "ai_confidence", desc: true });
  const [open, setOpen] = useState<string | null>(null);
  const rows = useMemo(() => [...latest.rows].sort((a, b) => {
    const dir = sort.desc ? -1 : 1;
    if (sort.key === "ticker") return dir * a.ticker.localeCompare(b.ticker);
    return dir * ((a[sort.key] ?? -Infinity) - (b[sort.key] ?? -Infinity));
  }), [latest.rows, sort]);
  const by = (key: SortKey) => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "ticker" }));
  const head = (key: SortKey, label: string, num = true) => (
    <th scope="col" className={num ? "num" : undefined} aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}>
      <button type="button" className="th-sort" onClick={() => by(key)}>{label}</button>
    </th>
  );
  return (
    <section aria-label="Latest scan scores" className="stack-sm">
      <h3 className="h3">Latest scan scores</h3>
      {!latest.scanned_at || !rows.length ? <Empty title="No saved AI Confidence scores yet" /> : <>
        <p className="cap">Scanned {etTime(latest.scanned_at)} · {rows.length} stocks · High {latest.tiers.High} · Medium {latest.tiers.Medium} · Low {latest.tiers.Low}. High is 40% or more, Medium 25–40%.</p>
        <div className="table-wrap"><table className="table">
          <thead><tr>{head("ticker", "Stock", false)}{head("ai_confidence", "AI Confidence")}<th scope="col">Tier</th>{head("hsf_score", "HSF Score")}{head("breakout_score", "Breakout")}<th scope="col">Setup</th><th scope="col"><span className="sr-only">Explain</span></th></tr></thead>
          <tbody>
            {rows.map((r) => <Fragment key={r.ticker}>
              <tr>
                <th scope="row" className="mono">{r.ticker}</th>
                <td className="num mono">{pct(r.ai_confidence)}</td>
                <td><Pill tone={tone(r.tier)}>{r.tier}</Pill></td>
                <td className="num mono">{fixed(r.hsf_score)}</td>
                <td className="num mono">{fixed(r.breakout_score, 1)}</td>
                <td>{r.setup ?? "—"}</td>
                <td><button type="button" className="btn btn-sm" aria-expanded={open === r.ticker} onClick={() => setOpen(open === r.ticker ? null : r.ticker)}>Why this score?</button></td>
              </tr>
              {open === r.ticker && <tr><td colSpan={7}><Explainer ticker={r.ticker} /></td></tr>}
            </Fragment>)}
          </tbody>
        </table></div>
      </>}
    </section>
  );
}

function Explainer({ ticker }: { ticker: string }) {
  const { busy, error, run } = useAction();
  const [res, setRes] = useState<Explain | null>(null);
  const go = async () => {
    const out = await run(() => unwrap(api.POST("/v1/admin/ml/ai-confidence/explain", { body: { ticker } })) as unknown as Promise<Explain>);
    if (out) setRes(out);
  };
  return (
    <div className="stack-sm" aria-live="polite">
      {!res && <div><button type="button" className="btn btn-sm btn-primary" onClick={go} disabled={busy}>{busy ? "Asking Claude…" : `Explain ${ticker} with Claude`}</button> <span className="cap">Uses one AI call.</span></div>}
      <ErrorLine error={error} />
      {res && <>
        {res.explanation ? <p>{res.explanation}</p> : <p className="form-error">{res.error}</p>}
        <p className="cap">Model inputs: {Object.entries(res.features).map(([k, n]) => `${k} ${n.toFixed(2)}`).join(" · ") || "not captured"}{res.earnings_in_days != null ? ` · earnings in ${res.earnings_in_days} day(s)` : ""}</p>
      </>}
    </div>
  );
}

function Calibration({ d }: { d: AiConfidenceSummary }) {
  const train = d.model?.training_buckets ?? [];
  const live = d.live;
  return (
    <section aria-label="Calibration" className="stack-sm">
      <h3 className="h3">Does the percentage hold up?</h3>
      <p className="cap">Last {live.days} days: {live.matured} saved signals have played out, {live.pending} are still inside their 5-day window. Only the 5-day high and low are stored, so a stock that touched both +4% and −2% counts as unclear and the hit rate shows as a range.</p>
      {!live.buckets.length ? <Empty title="No matured signals with AI Confidence yet" /> :
        <div className="table-wrap"><table className="table" aria-label="Live calibration">
          <thead><tr><th scope="col">Confidence</th><th scope="col" className="num">Signals</th><th scope="col" className="num">Avg confidence</th><th scope="col" className="num">Hit rate</th><th scope="col" className="num">Hits / misses / unclear</th></tr></thead>
          <tbody>{live.buckets.map((b) => <tr key={b.bucket}>
            <th scope="row">{b.bucket}</th><td className="num mono">{b.signals}</td><td className="num mono">{pct(b.mean_confidence)}</td>
            <td className="num mono">{b.hit_rate_low === b.hit_rate_high ? pct(b.hit_rate_low) : `${pct(b.hit_rate_low)} – ${pct(b.hit_rate_high)}`}</td>
            <td className="num mono">{b.hits} / {b.misses} / {b.unclear}</td>
          </tr>)}</tbody>
        </table></div>}
      {train.length > 0 && <details>
        <summary>At training time ({train.reduce((n, b) => n + b.signals, 0)} validation signals)</summary>
        <div className="table-wrap"><table className="table" aria-label="Training calibration">
          <thead><tr><th scope="col">Raw score</th><th scope="col" className="num">Signals</th><th scope="col" className="num">Avg score</th><th scope="col" className="num">Hit rate</th></tr></thead>
          <tbody>{train.map((b) => <tr key={b.bucket}><th scope="row">{b.bucket}</th><td className="num mono">{b.signals}</td><td className="num mono">{share(b.mean_confidence)}</td><td className="num mono">{share(b.hit_rate)}</td></tr>)}</tbody>
        </table></div>
      </details>}
    </section>
  );
}
