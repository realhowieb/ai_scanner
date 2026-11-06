"use client";

import { useSearchParams } from "next/navigation";
import type { Schemas } from "@/api/client";
import { Card, ErrorLine, Pill } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { defaultScanApi } from "@/hooks/useScanJob";
import { etTime, num, pct, price } from "@/lib/format";

type Evidence = Schemas["IndicatorEvidence"];
type Stock = Schemas["StockDetail"];

function flag(available: boolean, passed: boolean) { return !available ? "Needs data" : passed ? "Pass" : "Not met"; }

export function IndicatorFacts({ e, label = "RSI and MACD evidence" }: { e: Evidence; label?: string }) {
  const rsi = e.rsi_14;
  const previous = e.rsi_previous;
  const checks: [string, string, string][] = [
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
  return <>
    <div className="row-actions">
      <Pill>{e.rebound_candidate ? "Rebound candidate" : "Rebound not confirmed"}</Pill>
      <Pill>{e.momentum_recovery ? "Momentum recovery" : "Momentum recovery not confirmed"}</Pill>
      <Pill>{e.uptrend_pullback ? "Uptrend pullback" : "Uptrend pullback not confirmed"}</Pill>
    </div>
    <p className="body-sm">RSI {num(rsi)} · Prior five-bar low {num(e.prior_5_rsi_low)}{e.prior_5_rsi_low_date ? ` (${e.prior_5_rsi_low_date})` : ""} · Change {pct(e.change_pct)} · RVOL {num(e.relative_volume)}×</p>
    <details><summary>View rule-by-rule checks</summary>
    <div className="table-scroll" role="region" aria-label={label} tabIndex={0}>
      <table><thead><tr><th>Condition</th><th>Reading</th><th>Check</th></tr></thead><tbody>
        {checks.map(([label, value, result]) => <tr key={label}><td>{label}</td><td>{value}</td><td>{result}</td></tr>)}
      </tbody></table>
    </div>
    </details>
    <p className="cap">Each RSI setup also needs an up close and relative volume ≥1.0. Daily Wilder RSI(14); MACD(12,26,9). MACD above its signal line can still be below zero. Missing history is not a passing check.</p>
  </>;
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
    <Pill>Experimental research</Pill>
    {requested && <>
      <h3 className="h3">Selected custom scan</h3>
      {scanId && saved.loading && <p className="cap" role="status">Loading the saved scan evidence…</p>}
      <ErrorLine error={saved.error} />
      {snapshot ? <>
        <p className="cap">{selected || "Custom indicator filters"} · Scan {job?.started_at ? etTime(job.started_at) : "time unavailable"} · Daily bar {snapshot.bar_date ?? "date unavailable"}.</p>
        <IndicatorFacts e={snapshot} label="Selected scan RSI and MACD evidence" />
        <p className="cap">These readings are saved from your selected scan. The chart and HSF score use their own snapshots and may differ.</p>
      </> : (!scanId || (!saved.loading && !saved.error)) && <p className="cap">Saved evidence is unavailable for this ticker. Older scans may not contain it; no past qualification is inferred from newer bars.</p>}
    </>}
    {latest && <details open={!requested}><summary>Latest chart-bar checks · {latest.bar_date ?? "date unavailable"}</summary>
      <p className="cap">Calculated separately from the displayed daily bars ({s.volume_feed ?? "unknown"} volume feed). These readings may differ from the selected scan. The latest bar can change during the session.</p>
      <IndicatorFacts e={latest} label="Latest chart RSI and MACD evidence" />
    </details>}
    <p className="cap">These checks explain screening rules, not a prediction of profit. HSF score and model probabilities are separate and unchanged.</p>
  </Card>;
}
