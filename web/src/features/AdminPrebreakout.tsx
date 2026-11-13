"use client";

import { Fragment, useMemo, useState } from "react";

import { api, unwrap } from "@/api/client";
import { Card, Empty, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";

type Bucket = { bucket: string; signals: number; mean_confidence: number | null; hit_rate: number | null };
type Score = { ticker: string; probability: number | null; hsf_score: number | null; breakout_score: number | null;
  setup: string | null; model_version: string | null };
export type PrebreakoutSummary = {
  model: { model_version: string; trained_at: string | null; saved_at: string | null; dataset_version: string | null;
    target_rule: string | null; features: number | null; calibration_method: string | null; validation_auc: number | null;
    training_buckets: Bucket[] } | null;
  warnings: string[]; latest: { scanned_at: string | null; rows: Score[] }; live_validation: string;
};
type Inputs = { ticker: string; scanned_at: string; model_version: string | null; note: string;
  inputs: { name: string; value: number | null; defaulted: boolean | null }[] };
const pct = (v: number | null | undefined) => v == null ? "—" : `${v.toFixed(1)}%`;

export function PrebreakoutCard() {
  const v = useApi("admin-prebreakout", (signal) =>
    unwrap(api.GET("/v1/admin/ml/prebreakout", { signal })) as unknown as Promise<PrebreakoutSummary>);
  return <Card title="PreBreakout" aside={<button className="btn btn-sm" onClick={v.reload} disabled={v.refreshing}>Refresh</button>}>
    {v.error ? <ErrorState error={v.error} onRetry={v.reload} what="PreBreakout" />
      : !v.data ? <Skeleton rows={4} label="Loading PreBreakout" /> : <PrebreakoutBody d={v.data} />}
  </Card>;
}

export function PrebreakoutBody({ d }: { d: PrebreakoutSummary }) {
  const m = d.model;
  return <div className="stack">
    <section aria-label="PreBreakout model status" className="stack-sm">
      <p className="cap">The production scorer behind Premium PreBreakout. Separate from AI Confidence and ML v4 challengers.</p>
      {m ? <>
        <p>Active model: <strong className="mono">{m.model_version}</strong> {m.calibration_method
          ? <Pill tone="up">Calibration map: {m.calibration_method}</Pill> : <Pill tone="warn">No calibration map recorded</Pill>}</p>
        <p className="cap">Trained {etTime(m.trained_at)} · saved {etTime(m.saved_at)} · dataset {m.dataset_version ?? "not recorded"} · {m.features ?? "?"} inputs</p>
        <p>Target: {m.target_rule ?? "Not recorded for this model. Do not assume it shares AI Confidence’s target."}</p>
        <p className="cap">Training validation AUC: {m.validation_auc?.toFixed(3) ?? "not recorded"}. Historical evidence, not current live accuracy.</p>
      </> : <Empty title="No active database model recorded" />}
      {(d.warnings ?? []).map((w) => <p className="notice" key={w}>{w}</p>)}
    </section>
    <Scores latest={d.latest ?? { scanned_at: null, rows: [] }} />
    <section aria-label="PreBreakout calibration" className="stack-sm">
      <h3 className="h3">Does the percentage hold up?</h3>
      <p>{d.live_validation}</p>
      {m?.training_buckets?.length ? <details><summary>Training-time validation calibration</summary>
        <div className="table-wrap"><table className="table" aria-label="PreBreakout training calibration">
          <thead><tr><th scope="col">Raw score bucket</th><th scope="col">Signals</th><th scope="col">Average raw score</th><th scope="col">Target hit rate</th></tr></thead>
          <tbody>{m.training_buckets.map((b) => <tr key={b.bucket}><th scope="row">{b.bucket}</th><td>{b.signals}</td>
            <td>{pct(b.mean_confidence == null ? null : b.mean_confidence * 100)}</td><td>{pct(b.hit_rate == null ? null : b.hit_rate * 100)}</td></tr>)}</tbody>
        </table></div></details> : <p className="cap">No training calibration buckets recorded.</p>}
    </section>
  </div>;
}

function Scores({ latest }: { latest: PrebreakoutSummary["latest"] }) {
  const [sort, setSort] = useState<"ticker" | "probability">("probability");
  const [descending, setDescending] = useState(true);
  const [open, setOpen] = useState<string | null>(null);
  const rows = useMemo(() => [...latest.rows].sort((a, b) => {
    if (sort === "ticker") return (descending ? -1 : 1) * a.ticker.localeCompare(b.ticker);
    if (a.probability == null) return b.probability == null ? 0 : 1;
    if (b.probability == null) return -1;
    return (descending ? -1 : 1) * (a.probability - b.probability);
  }), [latest.rows, sort, descending]);
  const by = (key: typeof sort) => { setDescending(sort === key ? !descending : key !== "ticker"); setSort(key); };
  return <section aria-label="PreBreakout latest scan scores" className="stack-sm">
    <h3 className="h3">Latest saved PreBreakout scores</h3>
    <p className="cap">Scan {etTime(latest.scanned_at)}. Saved prediction-time percentages; no rescoring. Missing captures stay unavailable. No High/Medium/Low cutoffs are assumed.</p>
    {!rows.length ? <Empty title="No saved PreBreakout scores yet" /> : <div className="table-wrap"><table className="table">
      <thead><tr><th scope="col" aria-sort={sort === "ticker" ? descending ? "descending" : "ascending" : "none"}><button className="th-sort" onClick={() => by("ticker")}>Stock</button></th>
        <th scope="col" aria-sort={sort === "probability" ? descending ? "descending" : "ascending" : "none"}><button className="th-sort" onClick={() => by("probability")}>PreBreakout</button></th>
        <th scope="col">HSF Score</th><th scope="col">Breakout</th><th scope="col">Setup</th><th scope="col">Scored by</th><th scope="col">Inputs</th></tr></thead>
      <tbody>{rows.map((r) => <Fragment key={r.ticker}><tr><th scope="row">{r.ticker}</th><td>{r.probability == null ? "Unavailable" : pct(r.probability)}</td>
        <td>{r.hsf_score ?? "—"}</td><td>{r.breakout_score ?? "—"}</td><td>{r.setup ?? "—"}</td><td>{r.model_version ?? "Unknown"}</td>
        <td>{r.probability != null && <button className="btn btn-sm" aria-expanded={open === r.ticker} onClick={() => setOpen(open === r.ticker ? null : r.ticker)}>View saved inputs</button>}</td></tr>
        {open === r.ticker && latest.scanned_at && <tr><td colSpan={7}><SavedInputs ticker={r.ticker} at={latest.scanned_at} /></td></tr>}</Fragment>)}</tbody>
    </table></div>}
  </section>;
}

function SavedInputs({ ticker, at }: { ticker: string; at: string }) {
  const v = useApi(`admin-prebreakout-inputs:${ticker}:${at}`, (signal) => unwrap(api.GET("/v1/admin/ml/prebreakout/inputs/{ticker}", {
    params: { path: { ticker }, query: { scanned_at: at } }, signal })) as unknown as Promise<Inputs>);
  if (v.error) return <ErrorState error={v.error} onRetry={v.reload} what="saved PreBreakout inputs" />;
  if (!v.data) return <Skeleton rows={2} label="Loading saved inputs" />;
  return <div className="stack-sm"><p className="cap">{v.data.note}</p><p className="cap">Captured {etTime(v.data.scanned_at)} · model {v.data.model_version ?? "unknown"}</p>
    {!v.data.inputs.length ? <Empty title="Inputs were not recorded for this saved prediction" /> : <div className="table-wrap"><table className="table" aria-label="Captured PreBreakout inputs">
      <thead><tr><th scope="col">Input</th><th scope="col">Value</th><th scope="col">Source</th></tr></thead>
      <tbody>{v.data.inputs.map((f) => <tr key={f.name}><th scope="row">{f.name}</th><td>{f.value?.toFixed(3) ?? "Unavailable"}</td>
        <td>{f.defaulted == null ? "Unknown" : f.defaulted ? "Defaulted" : "Observed"}</td></tr>)}</tbody>
    </table></div>}</div>;
}
