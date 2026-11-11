"use client";

// Outcome Intelligence views (/v1/outcomes/*). Evidence first: every number sits next
// to its sample size and evidence label, every bucket/horizon/setup is shown (weak ones
// included), and nothing is picked as "best". Returns are fractions from the API.
import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { outcomes } from "@/api/account";
import type {
  OutcomeFilterInput, OutcomeGroups, OutcomeHorizons, OutcomeMetrics, OutcomeObservation, OutcomeScores, OutcomeSummary,
  OutcomeSymbol, OutcomeTimeseries,
} from "@/api/account";
import Link from "@/components/Link";
import { Card, Empty, ErrorState, Pill, Skeleton, Stat, StatGrid } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etDate, shortDate } from "@/lib/format";

export function signed(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const p = v * 100;
  return `${p > 0 ? "+" : ""}${p.toFixed(2)}%`;
}

export function share(v: number | null | undefined): string {
  return v === null || v === undefined || !Number.isFinite(v) ? "—" : `${Math.round(v * 100)}%`;
}

const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");

const QUALITY: Record<string, string> = { INSUFFICIENT: "Insufficient", LIMITED: "Limited", MODERATE: "Moderate", STRONG: "Strong" };

export function Quality({ q }: { q: string }) {
  return <Pill tone={q === "INSUFFICIENT" || q === "LIMITED" ? "warn" : undefined}>{QUALITY[q] ?? q} sample coverage</Pill>;
}

function range(r: { start?: string | null; end?: string | null }): string {
  return r.start && r.end ? `${etDate(r.start)} to ${etDate(r.end)}` : "no observations yet";
}

/** One metrics row: values always shown with their own counts; thin samples are muted, never hidden. */
function MetricCells({ m }: { m: OutcomeMetrics }) {
  return (
    <>
      <td className="num mono">{m.matured_count}</td>
      <td className={`num mono ${tone(m.median_return)}`}>{signed(m.median_return)}</td>
      <td className={`num mono ${tone(m.median_excess_return)}`}>{signed(m.median_excess_return)}</td>
      <td className="num mono">{share(m.benchmark_beat_rate)}<span className="cap"> of {m.benchmark_count}</span></td>
      <td className="num mono hide-narrow">{share(m.win_rate)}</td>
      <td className="num mono hide-narrow">{signed(m.average_mfe)} / {signed(m.average_mae)}</td>
    </>
  );
}

function MetricHead({ first }: { first: string }) {
  return (
    <thead>
      <tr>
        <th scope="col">{first}</th>
        <th scope="col" className="num">Matured</th>
        <th scope="col" className="num">Median return</th>
        <th scope="col" className="num">Median vs SPY</th>
        <th scope="col" className="num">Beat SPY</th>
        <th scope="col" className="num hide-narrow">Up</th>
        <th scope="col" className="num hide-narrow">Avg best / worst move</th>
      </tr>
    </thead>
  );
}

/** "1.0" or "1.0, 1.1" from {version: count}. */
function versions(v: Record<string, number> | undefined): string {
  const keys = Object.keys(v ?? {});
  return keys.length ? keys.sort().join(", ") : "—";
}

function SummaryCard({ s }: { s: OutcomeSummary }) {
  const m = s.metrics;
  return (
    <Card title="Every HSF signal so far" id="evidence"
      aside={<Quality q={m.evidence_quality} />}>
      <p className="cap">
        All {s.total_observations} HSF signals observed {range(s.date_range)} ({s.raw_observations} scan readings; one per ticker per day).
        {" "}{s.matured_observations} have a {s.horizon}-trading-day outcome, {s.pending_observations} are still pending
        {s.unavailable_observations ? ` and ${s.unavailable_observations} had no price data` : ""}. No filters applied.
      </p>
      <StatGrid label="Signal counts">
        <Stat label="Signals observed" value={s.total_observations.toLocaleString("en-US")} note={range(s.date_range)} />
        <Stat label="Outcome measured" value={s.matured_observations.toLocaleString("en-US")} note={`after ${s.horizon} trading days`} />
        <Stat label="Still pending" value={s.pending_observations.toLocaleString("en-US")} note="not counted as wins or losses" />
        <Stat label="Score version" value={versions(s.score_versions)} note={Object.keys(s.score_versions ?? {}).length > 1 ? "mixed: compare with care" : "one scoring method"} />
      </StatGrid>
      <p className="cap">Sample coverage describes the number of observations, not performance or predictive reliability. Best / worst move (MFE / MAE) measures the largest rise and drawdown during the holding period, not a realized trade result.</p>
      <dl className="kv">
        <div><dt>Median return</dt><dd className={`mono ${tone(m.median_return)}`}>{signed(m.median_return)}</dd></div>
        <div><dt>Median vs SPY</dt><dd className={`mono ${tone(m.median_excess_return)}`}>{signed(m.median_excess_return)}</dd></div>
        <div><dt>Beat SPY</dt><dd className="mono">{share(m.benchmark_beat_rate)} <span className="cap">of {m.benchmark_count}</span></dd></div>
        <div><dt>Finished up</dt><dd className="mono">{share(m.win_rate)} <span className="cap">of {m.matured_count}</span></dd></div>
        <div><dt>Avg best / worst move</dt><dd className="mono">{signed(m.average_mfe)} / {signed(m.average_mae)}</dd></div>
      </dl>
      {s.coverage.missing_benchmark > 0 && (
        <p className="cap">{s.coverage.missing_benchmark} matured signal{s.coverage.missing_benchmark === 1 ? " has" : "s have"} no SPY comparison yet, so SPY figures use {m.benchmark_count}.</p>
      )}
      {s.warnings.map((w) => <p key={w} className="cap"><Pill tone="warn">Note</Pill> {w}</p>)}
    </Card>
  );
}

/** Median excess return vs SPY per score range on one zero line; bar height follows the
 * sample size, and thin samples are faded, so a lucky 2-signal bucket can't read as a result. */
export function ScoreLadder({ buckets, horizon = 5 }: { buckets: OutcomeScores["buckets"]; horizon?: number }) {
  const rows = buckets.filter((b) => b.median_excess_return !== null && b.median_excess_return !== undefined && Number.isFinite(b.median_excess_return));
  if (rows.length < 2) return null;
  const W = 1200, ROW = 32, TOP = 8, LABEL = 190, VAL = 80;
  const H = TOP + rows.length * ROW + 4;
  const lim = Math.max(0.001, ...rows.map((b) => Math.abs(b.median_excess_return as number)));
  const zero = LABEL + (W - LABEL) / 2;
  const half = (W - LABEL) / 2 - VAL;
  const maxN = Math.max(1, ...rows.map((b) => b.matured_count));
  return (
    <figure className="chart ladder">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`Median ${horizon}-trading-day return vs SPY by HSF score range: ${rows.map((b) => `${b.bucket} ${signed(b.median_excess_return)} (n=${b.matured_count})`).join(", ")}`}>
        <line x1={zero} x2={zero} y1={2} y2={H - 2} className="grid-strong" />
        {rows.map((b, i) => {
          const v = b.median_excess_return as number;
          const y = TOP + i * ROW + ROW / 2;
          const w = (Math.abs(v) / lim) * half;
          const h = 6 + 12 * Math.sqrt(b.matured_count / maxN);
          const thin = b.evidence_quality === "INSUFFICIENT";
          return (
            <g key={b.bucket} className={thin ? "ladder-thin" : undefined}>
              <text x={0} y={y + 4} className="axis-label mono">HSF {b.bucket}</text>
              <text x={LABEL - 12} y={y + 4} className="axis" textAnchor="end">n={b.matured_count}</text>
              <rect x={v >= 0 ? zero : zero - w} y={y - h / 2} width={Math.max(1, w)} height={h} rx={3} className={v >= 0 ? "bar-up" : "bar-down"} />
              <text x={v >= 0 ? zero + w + 6 : zero - w - 6} y={y + 4} textAnchor={v >= 0 ? "start" : "end"} className={`mono axis-val ${v >= 0 ? "up" : "down"}`}>{signed(v)}</text>
            </g>
          );
        })}
      </svg>
      <figcaption className="cap">Median return vs SPY over the holding period, by score range. Thicker bars have more matured signals; faded ones are too thin to read as evidence.</figcaption>
    </figure>
  );
}

function ScoresCard({ d }: { d: OutcomeScores }) {
  const inv = d.calibration.metrics["median_excess_return"]?.inversions ?? [];
  return (
    <Card title="By HSF score" id="scores">
      <p className="cap">Every score range, including weak ones, over {d.horizon} trading days. Ranges with fewer than {d.evidence_thresholds["LIMITED"]} matured signals are greyed: read them as anecdotes.</p>
      <ScoreLadder buckets={d.buckets} horizon={d.horizon ?? 5} />
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Results by HSF score table; scroll sideways for more columns">
        <table className="table">
          <MetricHead first="HSF score" />
          <tbody>
            {d.buckets.map((b) => (
              <tr key={b.bucket} className={b.evidence_quality === "INSUFFICIENT" ? "row-muted" : undefined}>
                <th scope="row" className="row-h">{b.bucket}</th>
                <MetricCells m={b} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {inv.length > 0 && (
        <p className="cap"><Pill tone="warn">Not monotonic</Pill> {inv.map((i) => `HSF ${i.lower_bucket} beat ${i.higher_bucket} vs SPY`).join("; ")}.</p>
      )}
    </Card>
  );
}

function HorizonsCard({ d }: { d: OutcomeHorizons }) {
  return (
    <Card title="By holding period" id="horizons">
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Results by holding period table; scroll sideways for more columns">
        <table className="table">
          <MetricHead first="Held for" />
          <tbody>
            {d.horizons.map((h) => (
              <tr key={h.horizon} className={h.evidence_quality === "INSUFFICIENT" ? "row-muted" : undefined}>
                <th scope="row" className="row-h">{h.horizon} trading day{h.horizon === 1 ? "" : "s"}</th>
                <MetricCells m={h} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cap">Best and worst moves are only recorded for the 5-day window.</p>
    </Card>
  );
}

function SetupsCard({ d }: { d: OutcomeGroups }) {
  return (
    <Card title="By setup" id="setups">
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Results by setup table; scroll sideways for more columns">
        <table className="table">
          <MetricHead first="Setup" />
          <tbody>
            {d.groups.map((g) => (
              <tr key={g.name} className={g.evidence_quality === "INSUFFICIENT" ? "row-muted" : undefined}>
                <th scope="row" className="row-h">{g.name.replace(/_/g, " ")}</th>
                <MetricCells m={g} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cap">Ordered by number of signals, not by performance.</p>
    </Card>
  );
}

/** Track Record's evidence section: the complete dataset, then its breakdowns. */
export function OutcomeEvidence() {
  const sum = useApi("outcomes:summary", (signal) => outcomes.summary(signal));
  const scores = useApi(sum.data ? "outcomes:scores" : null, (signal) => outcomes.scores(signal));
  const hz = useApi(sum.data ? "outcomes:horizons" : null, (signal) => outcomes.horizons(signal));
  const setups = useApi(sum.data ? "outcomes:setups" : null, (signal) => outcomes.setups(signal));
  if (sum.error && !sum.data) {
    if (sum.error.status === 404) return null;   // an API that predates /v1/outcomes: the ranking study still shows
    return <ErrorState error={sum.error} onRetry={sum.reload} what="the HSF evidence" />;
  }
  if (!sum.data) return <Skeleton rows={6} label="Loading the HSF evidence" />;
  if (sum.data.total_observations === 0) {
    return <Card title="Every HSF signal so far"><Empty title="No HSF signals recorded yet." /></Card>;
  }
  return (
    <>
      <SummaryCard s={sum.data} />
      {scores.data ? <ScoresCard d={scores.data} /> : scores.error ? <ErrorState error={scores.error} onRetry={scores.reload} what="score buckets" /> : <Skeleton rows={6} label="Loading score buckets" />}
      {hz.data ? <HorizonsCard d={hz.data} /> : hz.error ? <ErrorState error={hz.error} onRetry={hz.reload} what="holding periods" /> : <Skeleton rows={3} label="Loading holding periods" />}
      {setups.data ? <SetupsCard d={setups.data} /> : setups.error ? <ErrorState error={setups.error} onRetry={setups.reload} what="setups" /> : <Skeleton rows={4} label="Loading setups" />}
      <ThroughTimeCard />
      <ExplorerCard setups={(setups.data?.groups ?? []).map((g) => g.name).filter((n) => n !== "unlabeled")} />
      <p className="cap">{sum.data.disclaimer}</p>
    </>
  );
}

/** Stock page: how earlier HSF signals for this ticker turned out (5-day window). */
/** Stock page (Pro): how past HSF signals like this one did at 1, 3 and 5 trading days.
 * "Like this one" = same primary setup and an HSF Score within 5 points. Descriptive only. */
export function similarRange(score: number): { min_score: number; max_score: number } {
  return { min_score: Math.max(0, Math.round(score) - 5), max_score: Math.min(100, Math.round(score) + 5) };
}

export function SimilarSetups({ score, setup, setupName }: { score: number; setup?: string | null; setupName?: string }) {
  const range = similarRange(score);
  const key = `outcomes:similar:${setup ?? ""}:${range.min_score}-${range.max_score}`;
  const r = useApi(key, (signal) => outcomes.similar({ setup, ...range }, signal));
  if (r.error && !r.data) return r.error.status === 404 ? null : <ErrorState error={r.error} onRetry={r.reload} what="similar past setups" />;
  if (!r.data) return <p className="cap">Loading similar past setups…</p>;
  return <SimilarTable d={r.data} what={`${setup ? `${setupName ?? setup} setups` : "HSF signals"} scoring ${range.min_score}–${range.max_score}`} />;
}

export function SimilarTable({ d, what }: { d: OutcomeHorizons; what: string }) {
  const matured = Math.max(0, ...d.horizons.map((h) => h.matured_count));
  if (matured === 0) return <p className="body-sm">No matured past {what} yet.</p>;
  return (
    <div className="stack-sm similar-setups">
      <p className="body-sm"><strong>Similar past setups:</strong> how earlier {what} moved afterwards.</p>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Similar past setups table; scroll sideways for more columns">
        <table className="table">
          <thead><tr>
            <th scope="col">Held for</th><th scope="col" className="num">Matured</th><th scope="col" className="num">Median return</th>
            <th scope="col" className="num">Median vs SPY</th><th scope="col" className="num">Finished up</th>
          </tr></thead>
          <tbody>
            {d.horizons.map((h) => (
              <tr key={h.horizon} className={h.evidence_quality === "INSUFFICIENT" ? "row-muted" : undefined}>
                <th scope="row" className="row-h">{h.horizon} day{h.horizon === 1 ? "" : "s"}</th>
                <td className="num mono">{h.matured_count}</td>
                <td className={`num mono ${tone(h.median_return)}`}>{signed(h.median_return)}</td>
                <td className={`num mono ${tone(h.median_excess_return)}`}>{signed(h.median_excess_return)}</td>
                <td className="num mono">{share(h.win_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cap">Past signals only, before costs; greyed rows have too few to read as evidence. Not a forecast.</p>
    </div>
  );
}

export function SymbolEvidence({ ticker }: { ticker: string }) {
  const r = useApi(`outcomes:symbol:${ticker}:10`, (signal) => outcomes.symbol(ticker, 10, signal));
  // An API without /v1/outcomes: the older context below still renders. Any other failure
  // says so under the heading, with a retry, instead of leaving the section blank.
  if (r.error && !r.data) return r.error.status === 404 ? null : <ErrorState error={r.error} onRetry={r.reload} what={`${ticker}'s signal history`} />;
  if (!r.data) return <p className="cap">Loading this ticker&apos;s HSF record…</p>;
  if (!Array.isArray(r.data.horizons)) return null;
  return (
    <>
      <SymbolLine d={r.data} />
      <PastSignals d={r.data} />
    </>
  );
}

export function SymbolLine({ d }: { d: OutcomeSymbol }) {
  const h = d.horizons.find((x) => x.horizon === 5) ?? d.horizons[d.horizons.length - 1];
  if (!h || h.sample_size === 0) return <p className="body-sm">No earlier HSF signals for {d.ticker}.</p>;
  if (h.matured_count === 0) return <p className="body-sm">{d.ticker}: {h.sample_size} earlier HSF signal{h.sample_size === 1 ? "" : "s"}, none matured yet (waiting for {h.horizon}-trading-day outcomes).</p>;
  return (
    <div className="stack-sm">
      <p className="body-sm">
        <strong>Historical HSF evidence:</strong> {h.matured_count} matured signal{h.matured_count === 1 ? "" : "s"} for {d.ticker} ({range(d.date_range)}).
        {h.benchmark_count > 0 && <> {h.benchmark_beat_count} of {h.benchmark_count} beat SPY over {h.horizon} trading days.</>}
        {" "}Median {h.horizon}-trading-day return {signed(h.median_return)}{h.benchmark_count > 0 && <>, median vs SPY {signed(h.median_excess_return)}</>}.
        {h.pending_count > 0 && <> {h.pending_count} still pending.</>}
      </p>
      <p className="cap"><Quality q={h.evidence_quality} /> Past signals only; not a forecast.</p>
    </div>
  );
}

// ---- shared pieces for record lists ---------------------------------------------------------------

/** One record's result at a horizon: the return when matured, otherwise its status. */
function resultAt(o: OutcomeObservation, horizon: number, field: "raw_return" | "excess_return" = "raw_return") {
  const r = o.outcomes.find((x) => x.horizon === horizon);
  if (!r) return <span className="cap">—</span>;
  if (r.status === "pending") return <span className="cap">pending</span>;
  if (r.status !== "matured") return <span className="cap">no data</span>;
  const v = r[field];
  return <span className={tone(v)}>{signed(v)}</span>;
}

const setupName = (s: string | null | undefined) => (s ? s.replace(/_/g, " ") : "—");
const signalDay = (o: OutcomeObservation) => shortDate(o.entry_day ?? o.observed_at);

/** Stock page: this ticker's newest HSF signals, each with what happened after 1, 3 and 5 trading days. */
export function PastSignals({ d }: { d: OutcomeSymbol }) {
  const items = d.observations?.items ?? [];
  if (items.length === 0) return null;
  const total = d.observations.total;
  return (
    <details className="past-signals">
      <summary>Past signals on {d.ticker} ({total})</summary>
      <div className="table-wrap" tabIndex={0} role="region" aria-label={`Past HSF signals on ${d.ticker}; scroll sideways for more columns`}>
        <table className="table">
          <thead><tr>
            <th scope="col">Signal day</th><th scope="col" className="num">HSF score</th><th scope="col" className="hide-narrow">Setup</th>
            <th scope="col" className="num">1 day</th><th scope="col" className="num">3 days</th><th scope="col" className="num">5 days</th>
            <th scope="col" className="num hide-narrow">5 days vs SPY</th>
          </tr></thead>
          <tbody>
            {items.map((o, i) => (
              <tr key={o.observation_id ?? `${o.entry_day}-${i}`}>
                <th scope="row" className="row-h">{signalDay(o)}</th>
                <td className="num mono">{o.hsf_score == null ? "—" : Math.round(o.hsf_score)}</td>
                <td className="hide-narrow">{setupName(o.setup)}</td>
                <td className="num mono">{resultAt(o, 1)}</td>
                <td className="num mono">{resultAt(o, 3)}</td>
                <td className="num mono">{resultAt(o, 5)}</td>
                <td className="num mono hide-narrow">{resultAt(o, 5, "excess_return")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="cap">{total > items.length ? `Newest ${items.length} of ${total} signals. ` : ""}Returns use the score and setup HSF had that day, before costs. Explore every signal on <Link href={`/track-record?ticker=${encodeURIComponent(d.ticker)}#explore`}>Track record</Link>.</p>
    </details>
  );
}

// ---- Through time ------------------------------------------------------------------------------------

/** Median 5-day return vs SPY per week or month of signals; thin periods are faded. */
export function PeriodChart({ points }: { points: OutcomeTimeseries["points"] }) {
  const rows = points.filter((p) => p.median_excess_return != null && Number.isFinite(p.median_excess_return));
  if (rows.length < 2) return <Empty title="Not enough matured periods to chart yet." />;
  const W = 720, H = 200, PAD = 12;
  const lim = Math.max(0.001, ...rows.map((p) => Math.abs(p.median_excess_return as number)));
  const mid = H / 2;
  const step = (W - PAD * 2) / rows.length;
  const bw = Math.max(2, step * 0.7);
  const above = rows.filter((p) => (p.median_excess_return as number) > 0).length;
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`Median return vs SPY by period: ${rows.map((p) => `${shortDate(p.period_start)} ${signed(p.median_excess_return)} (n=${p.matured_count})`).join(", ")}`}>
        <line x1={PAD} x2={W - PAD} y1={mid} y2={mid} className="grid" />
        {rows.map((p, i) => {
          const v = p.median_excess_return as number;
          const h = (Math.abs(v) / lim) * (mid - PAD);
          return <rect key={p.period_start} className={`${v >= 0 ? "bar-up" : "bar-down"}${p.evidence_quality === "INSUFFICIENT" ? " ladder-thin" : ""}`}
            x={PAD + i * step + (step - bw) / 2} width={bw} y={v >= 0 ? mid - h : mid} height={Math.max(1, h)} />;
        })}
        <text x={W - PAD} y={PAD + 10} className="axis" textAnchor="end">{signed(lim)}</text>
        <text x={W - PAD} y={H - PAD} className="axis" textAnchor="end">{signed(-lim)}</text>
      </svg>
      <figcaption className="cap">{rows.length} periods from {shortDate(rows[0]!.period_start)}. Each bar is the median 5-trading-day return vs SPY of signals first seen in that period; faded bars have too few matured signals to read as evidence. {above} of {rows.length} periods were above SPY.</figcaption>
    </figure>
  );
}

function ThroughTimeCard() {
  const [period, setPeriod] = useState<"week" | "month">("week");
  const r = useApi(`outcomes:timeseries:${period}`, (signal) => outcomes.timeseries(period, 5, signal));
  return (
    <Card title="Through time" id="through-time" aside={
      <div className="seg-group" role="radiogroup" aria-label="Group by">
        {(["week", "month"] as const).map((p) => (
          <label key={p} className={`seg${period === p ? " on" : ""}`}>
            <input type="radio" className="sr-only" name="outcome-period" checked={period === p} onChange={() => setPeriod(p)} />
            By {p}
          </label>
        ))}
      </div>
    }>
      {r.error && !r.data ? <ErrorState error={r.error} onRetry={r.reload} what="results through time" />
        : !r.data || r.loading ? <Skeleton rows={4} label="Loading results through time" />
        : r.data.points.length === 0 ? <Empty title="No signals recorded yet." />
        : (
          <>
            <PeriodChart points={r.data.points} />
            <details>
              <summary>Table</summary>
              <div className="table-wrap" tabIndex={0} role="region" aria-label="Results by period table; scroll sideways for more columns">
                <table className="table">
                  <thead><tr>
                    <th scope="col">{period === "week" ? "Week of" : "Month"}</th><th scope="col" className="num">Signals</th><th scope="col" className="num">Matured</th>
                    <th scope="col" className="num">Median return</th><th scope="col" className="num">Median vs SPY</th><th scope="col" className="num">Beat SPY</th>
                  </tr></thead>
                  <tbody>
                    {[...r.data.points].reverse().map((p) => (
                      <tr key={p.period_start} className={p.evidence_quality === "INSUFFICIENT" ? "row-muted" : undefined}>
                        <th scope="row" className="row-h">{shortDate(p.period_start)}</th>
                        <td className="num mono">{p.observation_count}</td>
                        <td className="num mono">{p.matured_count}</td>
                        <td className={`num mono ${tone(p.median_return)}`}>{signed(p.median_return)}</td>
                        <td className={`num mono ${tone(p.median_excess_return)}`}>{signed(p.median_excess_return)}</td>
                        <td className="num mono">{share(p.benchmark_beat_rate)}<span className="cap"> of {p.benchmark_count}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
            <p className="cap">Grouped by when HSF first saw the signal. Recent periods include signals still pending, so they fill in over the next week.</p>
          </>
        )}
    </Card>
  );
}

// ---- Explorer ------------------------------------------------------------------------------------------

type Draft = { setup: string; min: string; max: string; from: string; to: string; ticker: string; horizon: 1 | 3 | 5 };
const PAGE = 25;
const EMPTY: Draft = { setup: "", min: "", max: "", from: "", to: "", ticker: "", horizon: 5 };

function toNum(v: string): number | undefined {
  if (v.trim() === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(100, Math.max(0, n)) : undefined;
}

/** Filters from the address bar, so the stock page can link straight to one ticker. */
function initialDraft(): Draft {
  if (typeof window === "undefined") return EMPTY;
  const t = new URLSearchParams(window.location.search).get("ticker") ?? "";
  return { ...EMPTY, ticker: /^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$/.test(t) ? t.toUpperCase() : "" };
}

export function explorerQuery(d: Draft, page: number): OutcomeFilterInput {
  let min = toNum(d.min), max = toNum(d.max);
  if (min !== undefined && max !== undefined && min > max) [min, max] = [max, min];
  return {
    setup: d.setup || undefined, min_score: min, max_score: max,
    start_date: d.from || undefined, end_date: d.to || undefined,
    ticker: d.ticker.trim().toUpperCase() || undefined, horizon: d.horizon, page, page_size: PAGE,
  };
}

function ExplorerCard({ setups }: { setups: string[] }) {
  const [draft, setDraft] = useState<Draft>(initialDraft);
  const [applied, setApplied] = useState<Draft>(draft);
  const [page, setPage] = useState(1);
  const q = explorerQuery(applied, page);
  const r = useApi(`outcomes:query:${JSON.stringify(q)}`, (signal) => outcomes.query(q, signal));
  // Arriving from a stock page link (#explore): this card loads after the page, so scroll once it exists.
  useEffect(() => {
    if (window.location.hash === "#explore") document.getElementById("explore")?.scrollIntoView();
  }, []);
  const set = (patch: Partial<Draft>) => setDraft((d) => ({ ...d, ...patch }));
  const submit = (e: FormEvent) => { e.preventDefault(); setApplied(draft); setPage(1); };
  const reset = () => { setDraft(EMPTY); setApplied(EMPTY); setPage(1); };
  const d = r.data;
  const m = d?.metrics;
  const pages = d ? Math.max(1, Math.ceil(d.observations.total / PAGE)) : 1;
  return (
    <Card title="Explore past signals" id="explore">
      <p className="cap">Pick any mix of setup, score range, dates and ticker to see how those signals did, and the signals themselves.</p>
      <form className="filters history-filters" onSubmit={submit} aria-label="Signal filters">
        <label className="field">Setup<select value={draft.setup} onChange={(e) => set({ setup: e.target.value })}>
          <option value="">Any setup</option>
          {setups.map((s) => <option key={s} value={s}>{setupName(s)}</option>)}
        </select></label>
        <label className="field">HSF score from<input type="number" inputMode="numeric" min={0} max={100} value={draft.min} onChange={(e) => set({ min: e.target.value })} placeholder="0" /></label>
        <label className="field">to<input type="number" inputMode="numeric" min={0} max={100} value={draft.max} onChange={(e) => set({ max: e.target.value })} placeholder="100" /></label>
        <label className="field">From<input type="date" value={draft.from} onChange={(e) => set({ from: e.target.value })} /></label>
        <label className="field">Through<input type="date" min={draft.from || undefined} value={draft.to} onChange={(e) => set({ to: e.target.value })} /></label>
        <label className="field">Ticker<input type="text" value={draft.ticker} maxLength={10} autoCapitalize="characters" onChange={(e) => set({ ticker: e.target.value })} placeholder="Any" /></label>
        <label className="field">Held for<select value={draft.horizon} onChange={(e) => set({ horizon: Number(e.target.value) as 1 | 3 | 5 })}>
          <option value={1}>1 trading day</option><option value={3}>3 trading days</option><option value={5}>5 trading days</option>
        </select></label>
        <div className="row-actions">
          <button type="submit" className="btn btn-primary">Show signals</button>
          <button type="button" className="btn" onClick={reset}>Clear</button>
        </div>
      </form>
      {r.error && !d ? <ErrorState error={r.error} onRetry={r.reload} what="matching signals" />
        : !d || !m ? <Skeleton rows={5} label="Loading matching signals" />
        : d.observations.total === 0 ? <Empty title="No signals match these filters." />
        : (
          <div className="stack-sm" aria-live="polite">
            <p className="body-sm">
              <strong>{d.observations.total.toLocaleString("en-US")} signal{d.observations.total === 1 ? "" : "s"}</strong>, {m.matured_count.toLocaleString("en-US")} with a {q.horizon}-trading-day outcome.
              {m.matured_count > 0 && <> Median return {signed(m.median_return)}{m.benchmark_count > 0 && <>, median vs SPY {signed(m.median_excess_return)}, beat SPY {share(m.benchmark_beat_rate)} of {m.benchmark_count.toLocaleString("en-US")}</>}, finished up {share(m.win_rate)}.</>}
              {m.pending_count > 0 && <> {m.pending_count} still pending.</>}
            </p>
            <p className="cap"><Quality q={m.evidence_quality} /> {m.distinct_days} distinct signal day{m.distinct_days === 1 ? "" : "s"}; signals on the same day move together.</p>
            <div className="table-wrap" tabIndex={0} role="region" aria-label="Matching signals table; scroll sideways for more columns">
              <table className="table">
                <thead><tr>
                  <th scope="col">Signal day</th><th scope="col">Ticker</th><th scope="col" className="num">HSF score</th>
                  <th scope="col" className="hide-narrow">Setup</th><th scope="col" className="num">Return</th><th scope="col" className="num">vs SPY</th>
                </tr></thead>
                <tbody>
                  {d.observations.items.map((o, i) => (
                    <tr key={o.observation_id ?? `${o.ticker}-${o.entry_day}-${i}`}>
                      <th scope="row" className="row-h">{signalDay(o)}</th>
                      <td><Link href={`/stocks/${encodeURIComponent(o.ticker)}`} className="mono">{o.ticker}</Link></td>
                      <td className="num mono">{o.hsf_score == null ? "—" : Math.round(o.hsf_score)}</td>
                      <td className="hide-narrow">{setupName(o.setup)}</td>
                      <td className="num mono">{resultAt(o, q.horizon)}</td>
                      <td className="num mono">{resultAt(o, q.horizon, "excess_return")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {pages > 1 && (
              <div className="row-actions">
                <button type="button" className="btn btn-sm" disabled={page <= 1 || r.loading} onClick={() => setPage((p) => p - 1)}>Newer</button>
                <span className="cap">Page {page} of {pages}</span>
                <button type="button" className="btn btn-sm" disabled={page >= pages || r.loading} onClick={() => setPage((p) => p + 1)}>Older</button>
              </div>
            )}
            <p className="cap">Newest first. Past signals only, before costs. Not a forecast.</p>
          </div>
        )}
    </Card>
  );
}
