"use client";

import { useState } from "react";
import { useSearchParams } from "next/navigation";
import type { Schemas } from "@/api/client";
import { Card, ErrorLine, Pill } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { defaultScanApi } from "@/hooks/useScanJob";
import { etTime, num, pct, price } from "@/lib/format";

type Evidence = Schemas["IndicatorEvidence"];
type Stock = Schemas["StockDetail"];
type Selection = Pick<Schemas["ScanParams"], "rebound" | "momentum_recovery" | "uptrend_pullback" | "macd_confirmation">;

type Check = [string, string, string];
function EvidenceTable({ rows, label }: { rows: Check[]; label: string }) {
  return <div className="table-wrap" role="region" aria-label={label} tabIndex={0}>
    <table className="table indicator-checks-table"><thead><tr><th scope="col">Condition</th><th scope="col">Reading</th><th scope="col">Check</th></tr></thead><tbody>
      {rows.map(([name, value, result]) => <tr key={name}><th scope="row">{name}</th><td>{value}</td><td>{result}</td></tr>)}
    </tbody></table>
  </div>;
}
function HistoryBasis({ e }: { e: Evidence }) {
  return <p className="cap">Calculation history: {e.history_bars != null && e.history_start ? `${e.history_bars} daily bars, ${e.history_start} through ${e.bar_date ?? "unknown date"}.` : "not recorded for this saved reading."}</p>;
}

function flag(available: boolean, passed: boolean) { return !available ? "Needs data" : passed ? "Pass" : "Not met"; }

export function IndicatorFacts({ e, label = "RSI and MACD evidence", selection }: { e: Evidence; label?: string; selection?: Selection }) {
  const rsi = e.rsi_14;
  const previous = e.rsi_previous;
  const checks: Check[] = [
    ["Oversold in prior five daily bars", e.prior_5_rsi_low == null ? "—" : `${num(e.prior_5_rsi_low)} on ${e.prior_5_rsi_low_date ?? "unknown date"}`, flag(e.prior_5_rsi_low != null, (e.prior_5_rsi_low ?? 100) <= 30)],
    ["Rebound RSI: above 30, at most 50", num(rsi), flag(rsi != null, rsi != null && rsi > 30 && rsi <= 50)],
    ["Up close", `${price(e.previous_close)} → ${price(e.close)} (${pct(e.change_pct)})`, flag(e.close != null && e.previous_close != null, (e.close ?? 0) > (e.previous_close ?? Infinity))],
    ["Relative volume: at least 1.0", e.relative_volume == null ? "—" : `${num(e.relative_volume)}×`, flag(e.relative_volume != null, (e.relative_volume ?? 0) >= 1)],
    ["Momentum recovery: RSI crosses above 50", `${num(previous)} → ${num(rsi)}`, flag(previous != null && rsi != null, previous != null && rsi != null && previous <= 50 && rsi > 50)],
    ["Pullback: RSI turns up within 40–50", `${num(previous)} → ${num(rsi)}`, flag(previous != null && rsi != null, previous != null && rsi != null && previous >= 40 && previous < rsi && rsi <= 50)],
    ["Price above 50-day average", `${price(e.close)} / SMA ${price(e.sma50)}`, flag(e.close != null && e.sma50 != null, (e.close ?? 0) > (e.sma50 ?? Infinity))],
    ["MACD above signal line", `${num(e.macd_line, 4)} / signal ${num(e.macd_signal, 4)}`, flag(e.macd_histogram != null, e.macd_above_signal)],
    ["Fresh MACD bullish crossover", `Histogram ${num(e.macd_previous_histogram, 4)} → ${num(e.macd_histogram, 4)}`, flag(e.macd_histogram != null && e.macd_previous_histogram != null, e.macd_bullish_cross)],
  ];
  const statuses = [
    { name: "Rebound candidate", passed: e.rebound_candidate, selected: selection?.rebound },
    { name: "Momentum recovery", passed: e.momentum_recovery, selected: selection?.momentum_recovery },
    { name: "Uptrend pullback", passed: e.uptrend_pullback, selected: selection?.uptrend_pullback },
    { name: "Fresh MACD bullish crossover", passed: e.macd_bullish_cross, selected: selection?.macd_confirmation === "bullish_cross" },
    { name: "MACD above signal line", passed: e.macd_above_signal, selected: selection?.macd_confirmation === "above_signal" },
  ];
  const highlighted = statuses.filter((s) => selection ? s.selected : s.passed);
  const required = new Set<number>();
  if (selection?.rebound) [0, 1, 2, 3].forEach((i) => required.add(i));
  if (selection?.momentum_recovery) [2, 3, 4].forEach((i) => required.add(i));
  if (selection?.uptrend_pullback) [2, 3, 5, 6].forEach((i) => required.add(i));
  if (selection?.macd_confirmation === "above_signal") required.add(7);
  if (selection?.macd_confirmation === "bullish_cross") required.add(8);
  const focused = checks.filter((_, i) => required.has(i));
  const other = checks.filter((_, i) => !required.has(i));
  return <>
    <div className="row-actions">
      {highlighted.map((s) => <Pill key={s.name} tone={s.passed ? "up" : "warn"}>{selection ? `${s.name}: ${s.passed ? "passed" : "not confirmed"}` : s.name}</Pill>)}
      {!selection && highlighted.length === 0 && <span className="cap">No indicator setup confirmed.</span>}
    </div>
    <p className="body-sm">RSI {num(rsi)} · Prior five-bar low {num(e.prior_5_rsi_low)}{e.prior_5_rsi_low_date ? ` (${e.prior_5_rsi_low_date})` : ""} · Change {pct(e.change_pct)} · RVOL {num(e.relative_volume)}×</p>
    <details><summary>View rule-by-rule checks</summary>
      {focused.length > 0 ? <>
        <p className="strong">Required for the selected setup</p>
        {!selection?.macd_confirmation && <p className="cap">MACD was not required for this scan.</p>}
        <EvidenceTable rows={focused} label={label} />
        <details><summary>Other indicator checks (not required for this scan)</summary>
          <div className="row-actions">{statuses.filter((s) => !s.selected).map((s) => <Pill key={s.name}>{s.name}: {s.passed ? "passed" : "not confirmed"}</Pill>)}</div>
          <EvidenceTable rows={other} label={`${label} · other indicators`} />
        </details>
      </> : <>
        <div className="row-actions">{statuses.map((s) => <Pill key={s.name}>{s.name}: {s.passed ? "passed" : "not confirmed"}</Pill>)}</div>
        <EvidenceTable rows={checks} label={label} />
      </>}
    </details>
    <details><summary>Calculation details</summary>
      <HistoryBasis e={e} />
      <p className="cap">Each RSI setup also needs an up close and relative volume ≥1.0. Daily Wilder RSI(14); MACD(12,26,9). MACD above its signal line can still be below zero. Missing history is not a passing check.</p>
    </details>
  </>;
}

function LatestReadings({ s }: { s: Stock }) {
  if (!s.indicator_evidence) return null;
  return <>
    <p className="cap">Daily bar {s.indicator_evidence.bar_date ?? "date unavailable"} · {s.volume_feed ?? "unknown"} volume feed. The latest daily bar can change during the session.</p>
    <IndicatorFacts e={s.indicator_evidence} label="Latest chart RSI and MACD evidence" />
  </>;
}

/** Mount comparison readings only on request; each stock/scan starts collapsed. */
function LatestComparison({ s }: { s: Stock }) {
  const [expanded, setExpanded] = useState(false);
  return <details onToggle={(event) => setExpanded(event.currentTarget.open)}>
    <summary>Compare with latest</summary>
    {expanded && <>
      <p className="cap">These are separate chart-bar readings; they do not replace your saved scan. RSI and MACD can differ when the calculation history or bars differ, even if the final close matches.</p>
      <LatestReadings s={s} />
    </>}
  </details>;
}

/** Custom links retain the account-owned scan; chart readings remain a separate snapshot. */
export function StockIndicatorEvidence({ s }: { s: Stock }) {
  const params = useSearchParams();
  const requested = params.get("scan");
  const scanId = requested && /^[0-9a-f]{32}$/.test(requested) ? requested : null;
  const saved = useApi(scanId ? `scan-evidence:${scanId}` : null, (signal) => defaultScanApi.get(scanId!, signal));
  const job = saved.data;
  const row = job?.status === "complete" ? job.result?.setups.find((r) => r.ticker === s.ticker) : null;
  const snapshot = row?.indicator_evidence;
  const latest = s.indicator_evidence;
  if (!requested && !latest) return null;
  const selected = job ? [job.params.rebound && "Rebound", job.params.momentum_recovery && "Momentum recovery", job.params.uptrend_pullback && "Uptrend pullback",
    job.params.macd_confirmation === "bullish_cross" && "Fresh MACD bullish crossover", job.params.macd_confirmation === "above_signal" && "MACD above signal"].filter(Boolean).join(" · ") : "";
  return <Card title="RSI and MACD checks" id="indicators">
    <div><Pill>Experimental research</Pill></div>
    {requested && <>
      <h3 className="h3">Saved scan evidence</h3>
      {scanId && saved.loading && <p className="cap" role="status">Loading the saved scan evidence…</p>}
      <ErrorLine error={saved.error} />
      {snapshot ? <>
        <p className="cap">{selected || "Custom indicator filters"} · Scan {job?.started_at ? etTime(job.started_at) : "time unavailable"} · Daily bar {snapshot.bar_date ?? "date unavailable"}.</p>
        <IndicatorFacts e={snapshot} selection={job?.params} label="Selected scan RSI and MACD evidence" />
        <p className="cap">These readings are saved from your selected scan. {latest ? "Use the comparison below to review newer chart data." : "Current chart readings are unavailable."}</p>
      </> : (!scanId || (!saved.loading && !saved.error)) && <p className="cap">Saved evidence is unavailable for this ticker. Older scans may not contain it; no past qualification is inferred from newer bars.</p>}
    </>}
    {latest && (requested ? <LatestComparison key={`${s.ticker}:${requested}`} s={s} /> : <>
      <h3 className="h3">Latest indicator checks</h3>
      <LatestReadings s={s} />
    </>)}
    <p className="cap">These checks explain screening rules, not a prediction of profit. HSF score and model probabilities are separate and unchanged.</p>
  </Card>;
}
