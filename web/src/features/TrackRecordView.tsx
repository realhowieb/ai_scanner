"use client";

// Historical research (Pro): how saved scan picks did against SPY, by ranking and
// horizon, as computed daily by the scheduler. Descriptive only. Summaries below the
// API's minimum sample size are shown as "still building" with no rates, so a handful
// of picks never reads as evidence.
import { useState } from "react";

import { research } from "@/api/account";
import type { TrackRecordSummary } from "@/api/account";
import type { Schemas } from "@/api/client";
import { Card, Empty, ErrorState, Locked, Pill, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etDate, etTime } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

import { OutcomeEvidence } from "./OutcomeEvidence";
import { csvFile, ResearchCsvButton, round } from "./ResearchCsv";
import type { CsvTable } from "./ResearchCsv";

type Ranking = "breakout" | "prebreakout";
type Day = Schemas["TrackRecordDay"];

const RANKINGS: { id: Ranking; label: string; note: string }[] = [
  { id: "breakout", label: "Breakout score", note: "Each saved scan's top names by Breakout score" },
  { id: "prebreakout", label: "PreBreakout", note: "Top names by the PreBreakout probabilities saved with each scan. Scans without saved predictions are excluded." },
];

/** 0.012 -> "+1.20%" (the API sends fractions). */
export function signedPct(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const p = v * 100;
  return `${p > 0 ? "+" : ""}${p.toFixed(2)}%`;
}

function share(v: number | null | undefined): string {
  return v === null || v === undefined || !Number.isFinite(v) ? "—" : `${Math.round(v * 100)}%`;
}


/** `runningSum` overlays the running sum of the daily averages: only for 1-day holds, where days don't overlap. */
function DailyChart({ days, runningSum = false }: { days: Day[]; runningSum?: boolean }) {
  const pts = days.filter((d) => d.avg_excess_return !== null && d.avg_excess_return !== undefined && Number.isFinite(d.avg_excess_return));
  if (pts.length < 2) return <Empty title="Not enough matured days to chart yet." />;
  const W = 720, H = 200, PAD = 12;
  const vals = pts.map((d) => d.avg_excess_return as number);
  const lim = Math.max(0.001, ...vals.map(Math.abs));
  const mid = H / 2;
  const step = (W - PAD * 2) / pts.length;
  const bw = Math.max(1, step * 0.7);
  const pos = vals.filter((v) => v > 0).length;
  // Running sum of the daily averages, on its own scale (right-hand labels).
  const run: number[] = [];
  vals.forEach((v, i) => run.push((run[i - 1] ?? 0) + v));
  const rlim = Math.max(0.001, ...run.map(Math.abs));
  const ry = (v: number) => mid - (v / rlim) * (mid - PAD);
  const runPath = run.map((v, i) => `${i ? "L" : "M"}${(PAD + i * step + step / 2).toFixed(1)},${ry(v).toFixed(1)}`).join("");
  const total = run[run.length - 1]!;
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`Daily average excess return vs SPY over ${pts.length} days from ${etDate(pts[0]!.day)} to ${etDate(pts[pts.length - 1]!.day)}; ${pos} days above SPY`}>
        <line x1={PAD} x2={W - PAD} y1={mid} y2={mid} className="grid" />
        {pts.map((d, i) => {
          const v = d.avg_excess_return as number;
          const h = (Math.abs(v) / lim) * (mid - PAD);
          return <rect key={d.day} className={v >= 0 ? "bar-up" : "bar-down"} x={PAD + i * step + (step - bw) / 2} width={bw} y={v >= 0 ? mid - h : mid} height={Math.max(1, h)} />;
        })}
        {runningSum && <path d={runPath} className="run-line" />}
        {runningSum && <circle cx={PAD + (run.length - 1) * step + step / 2} cy={ry(total)} r={3.5} className="run-dot" />}
        <text x={W - PAD} y={PAD + 10} className="axis" textAnchor="end">{signedPct(lim)}</text>
        <text x={W - PAD} y={H - PAD} className="axis" textAnchor="end">{signedPct(-lim)}</text>
      </svg>
      <figcaption className="cap">
        {pts.length} days, {etDate(pts[0]!.day)} to {etDate(pts[pts.length - 1]!.day)}. Bars are each day&apos;s average excess return of that day&apos;s picks vs SPY. {pos} of {pts.length} days were above SPY.
        {runningSum && <>{" "}<span className="run-key">Line</span>: running sum of those daily averages, {signedPct(total)} at the end (its own scale; no costs, so not a portfolio return).</>}
      </figcaption>
    </figure>
  );
}

/** Premium CSV: the ranking study by horizon, then the chosen horizon's day-by-day series.
 * Rates for horizons still building are left blank, as on the page. */
export function trackRecordCsv(rows: TrackRecordSummary[], days: Day[], horizon: number): CsvTable {
  const p = (v: number | null | undefined) => (v == null ? null : round(v * 100, 3));
  const out: unknown[][] = rows.map((r) => ["Summary", r.ranking_label, r.horizon_days, "", r.sufficient ? p(r.avg_excess_return) : null,
    r.sufficient ? p(r.median_excess_return) : null, r.sufficient ? p(r.win_rate) : null, r.sample_size ?? null, r.runs_used ?? null, r.benchmark, r.computed_at ?? "", r.methodology, r.entry_policy, r.sufficient ? p(r.avg_absolute_return) : null,
    r.sufficient ? p(r.median_absolute_return) : null, r.sufficient ? p(r.positive_return_rate) : null,
    r.selected_count, r.evaluated_count, r.excluded_count, r.coverage_pct, r.unique_ticker_count, r.signal_date_count,
    JSON.stringify(r.exclusions ?? {})]);
  for (const d of days) out.push(["Daily", rows[0]?.ranking_label ?? "", horizon, d.day, p(d.avg_excess_return), null, null, null, null, rows[0]?.benchmark ?? "SPY", "", d.methodology, "", null, null, null, null, null, null, null, null, null, ""]);
  return {
    header: ["Section", "Ranking", "Horizon (trading days)", "Day", "Avg vs benchmark %", "Median vs benchmark %", "Beat benchmark %", "Matured picks", "Scans", "Benchmark", "Computed (UTC)", "Methodology", "Entry policy", "Avg absolute %", "Median absolute %", "Positive absolute %", "Selected", "Evaluated", "Excluded", "Coverage %", "Unique tickers", "Signal dates", "Exclusions"],
    rows: out,
  };
}

function SummaryTable({ rows, minSample }: { rows: TrackRecordSummary[]; minSample: number }) {
  return <div className="table-wrap" tabIndex={0} role="region" aria-label="Ranking study table; scroll sideways for more columns">
    <table className="table"><thead><tr>
      <th>Horizon</th><th>Avg stock</th><th>Median stock</th><th>Avg vs SPY</th><th>Median vs SPY</th>
      <th>Positive stock return</th><th>Beat SPY</th><th>Evaluated / selected</th><th>Coverage</th><th>Unique tickers</th><th>Signal dates</th>
    </tr></thead><tbody>{rows.map((r) => <tr key={r.horizon_days}>
      <th scope="row">{r.horizon_days} trading day{r.horizon_days === 1 ? "" : "s"}</th>
      {r.sufficient ? <>
        <td className="num mono">{signedPct(r.avg_absolute_return)}</td><td className="num mono">{signedPct(r.median_absolute_return)}</td>
        <td className="num mono">{signedPct(r.avg_excess_return)}</td><td className="num mono">{signedPct(r.median_excess_return)}</td>
        <td className="num mono">{share(r.positive_return_rate)}</td><td className="num mono">{share(r.benchmark_outperformance_rate)}</td>
      </> : <td colSpan={6} className="cap">Still building ({r.sample_size ?? 0} of {minSample} needed)</td>}
      <td>{r.evaluated_count ?? 0} / {r.selected_count ?? 0}</td><td>{r.coverage_pct == null ? "—" : `${r.coverage_pct.toFixed(1)}%`}</td>
      <td>{r.unique_ticker_count ?? 0}</td><td>{r.signal_date_count ?? 0}</td>
    </tr>)}</tbody></table>
    <details><summary>Coverage and exclusions</summary>{rows.map((r) => <p className="cap" key={r.horizon_days}>
      {r.horizon_days}d: {r.excluded_count ?? 0} excluded. {Object.entries(r.exclusions ?? {}).map(([reason, count]) => `${reason}: ${count}`).join("; ") || "No selected-pick exclusions."}
      {" "}{r.ranking_unavailable_snapshots ?? 0} snapshots had no usable ranking evidence; no replacement picks were invented.
    </p>)}</details>
  </div>;
}

function Research() {
  const [ranking, setRanking] = useState<Ranking>("breakout");
  const rec = useApi("track-record:track-v2-next-open", (signal) => research.trackRecord(signal));
  const rows = (rec.data?.methodology === "track-v2-next-open" ? rec.data.summaries : []).filter((r) => r.methodology === "track-v2-next-open" && r.ranking === ranking).sort((a, b) => a.horizon_days - b.horizon_days);
  const sufficient = rows.filter((r) => r.sufficient);
  const [horizon, setHorizon] = useState<number | null>(null);
  const chosen = horizon !== null && rows.some((r) => r.horizon_days === horizon) ? horizon
    : (sufficient.find((r) => r.horizon_days === 5) ?? sufficient[0] ?? rows[0])?.horizon_days ?? null;
  const daily = useApi(rec.data && chosen !== null ? `tr-daily:track-v2-next-open:${ranking}:${chosen}` : null,
    (signal) => research.daily(ranking, chosen!, 120, signal));

  if (rec.error && !rec.data) {
    if (rec.error.status === 403) return <Card><Locked title="Historical research is part of Pro" plan="pro">How saved scan picks did against SPY, by horizon.</Locked></Card>;
    return <ErrorState error={rec.error} onRetry={rec.reload} what="the track record" />;
  }
  if (!rec.data) return <Skeleton rows={8} label="Loading the track record" />;
  const d = rec.data;
  const meta = rows.find((r) => r.computed_at) ?? d.summaries[0];
  const rk = RANKINGS.find((r) => r.id === ranking)!;

  return (
    <>
      <p className="banner" role="note">{d.disclaimer}</p>
      <section aria-label="Ranking" className="filters">
        <div className="chips" role="group" aria-label="Ranking">
          {RANKINGS.map((r) => (
            <button key={r.id} type="button" className="chip" aria-pressed={ranking === r.id} onClick={() => { setRanking(r.id); setHorizon(null); }}>{r.label}</button>
          ))}
        </div>
      </section>

      {rows.length === 0 ? (
        <Card><Empty title="No track record computed for this ranking yet.">It&apos;s computed once a day from saved scans whose outcomes have matured.</Empty></Card>
      ) : (
        <>
          <Card title={`Ranking study: top ${rk.label} picks vs ${meta?.benchmark ?? "SPY"}`} id="summary"
            aside={meta?.computed_at ? `Computed ${etTime(meta.computed_at)}` : undefined}>
            <p className="cap">
              {rk.note}{meta?.top_n ? ` (top ${meta.top_n} per scan)` : ""}, held for each horizon. Stock returns and excess returns over {meta?.benchmark ?? "SPY"} are shown separately. Beating SPY can still mean losing money. Zero stock returns are not positive; benchmark ties do not count as outperformance.
              A horizon needs at least {d.min_sample_size} matured picks before rates are shown.
            </p>
            <SummaryTable rows={rows} minSample={d.min_sample_size} />
            <div className="row-actions end"><ResearchCsvButton what="the Track record" filename={csvFile("track-record", rk.label)}
              build={() => trackRecordCsv(rows, (daily.data ?? []).filter((row) => row.methodology === "track-v2-next-open"), chosen ?? rows[0]!.horizon_days)} /></div>
          </Card>

          <Card title="Day by day" id="daily" aside={
            <div className="seg-group" role="radiogroup" aria-label="Horizon">
              {rows.map((r) => (
                <label key={r.horizon_days} className={`seg${chosen === r.horizon_days ? " on" : ""}`}>
                  <input type="radio" className="sr-only" name="horizon" checked={chosen === r.horizon_days} onChange={() => setHorizon(r.horizon_days)} />
                  {r.horizon_days}d
                </label>
              ))}
            </div>
          }>
            {daily.error && !daily.data ? <ErrorState error={daily.error} onRetry={daily.reload} what="the daily series" />
              : !daily.data || daily.loading ? <Skeleton rows={4} label="Loading the daily series" />
              : <DailyChart days={daily.data.filter((row) => row.methodology === "track-v2-next-open")} />}
            {chosen !== null && !rows.find((r) => r.horizon_days === chosen)?.sufficient && (
              <p className="cap"><Pill tone="warn">Small sample</Pill> This horizon hasn&apos;t reached {d.min_sample_size} matured picks; read the bars as anecdotes.</p>
            )}
          </Card>
        </>
      )}
      <p className="cap">Evaluated observations count repeated ticker selections; overlapping windows are not independent trades. No capital-constrained portfolio, costs, spreads, slippage or liquidity constraints are modeled.</p>
    </>
  );
}

/** Plain-language method notes: what is counted, when, and what the numbers do not mean. */
function HowMeasured() {
  return (
    <details className="method">
      <summary>How these results are measured</summary>
      <dl className="method-body">
        <div><dt>What counts as a signal</dt><dd>A ticker that appeared in an HSF scan, counted once per ticker per day, with the score and setup it had then. The ranking study uses the latest saved snapshot for each exchange date, including its saved prediction evidence; it does not rescore history with today’s model.</dd></div>
        <div><dt>When an outcome is measured</dt><dd>The overall record uses its own close-to-close outcome convention. The ranking study below uses the first regular-session open strictly after snapshot availability and exits at the close of the last session in its horizon (1d means that session’s open to close). SPY uses the exact same dates. These daily-bar prices are research proxies, not fills. Signals whose period hasn&apos;t ended are pending: they are not wins or losses.</dd></div>
        <div><dt>What the HSF Score is</dt><dd>An opportunity ranking from 0 to 100. It is not a probability of profit, and these tables are not a model&apos;s training accuracy.</dd></div>
        <div><dt>Historical description vs forward evidence</dt><dd>Everything on this page describes signals that have already happened. Sample coverage only says how many there are. Validated forward evidence would come from a test registered in advance and compared with a matched control group of stocks HSF did not pick; HSF runs that test separately and does not yet claim its signals beat that control.</dd></div>
        <div><dt>What this can&apos;t tell you</dt><dd>No costs, slippage or position sizing are included, and samples are small. Past results describe what happened; they don&apos;t predict future results.</dd></div>
      </dl>
    </details>
  );
}

export function TrackRecordView() {
  const { can } = useSession();
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Track record</h1>
          <p className="cap">How every HSF signal did afterwards, against SPY over the same days, with sample sizes. A historical description, not validated forward evidence.</p>
        </div>
      </section>
      {can("can_track_record") ? (
        <>
          <nav className="chips" aria-label="Research sections"><a className="chip" href="#overall-record">Overall record</a><a className="chip" href="#ranking-study">Ranking study</a></nav><HowMeasured /><div id="overall-record" className="stack"><OutcomeEvidence /></div>
          <section id="ranking-study" className="stack" aria-label="Ranking study">
            <h2 className="h2">Ranking study</h2>
            <p className="cap">A separate, narrower study: only each saved scan&apos;s top names under one ranking. It is not the full HSF record above.</p>
            <Research />
          </section>
        </>
      ) : (
        <Card><Locked title="Historical research is part of Pro" plan="pro">How saved scan picks did against SPY over 1 to 20 trading days, with sample sizes.</Locked></Card>
      )}
    </div>
  );
}
