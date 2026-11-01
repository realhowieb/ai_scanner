"use client";

// Outcome Intelligence views (/v1/outcomes/*). Evidence first: every number sits next
// to its sample size and evidence label, every bucket/horizon/setup is shown (weak ones
// included), and nothing is picked as "best". Returns are fractions from the API.
import { outcomes } from "@/api/account";
import type { OutcomeGroups, OutcomeHorizons, OutcomeMetrics, OutcomeScores, OutcomeSummary, OutcomeSymbol } from "@/api/account";
import { Card, Empty, ErrorState, Pill, Skeleton, Stat, StatGrid } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etDate } from "@/lib/format";

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
  const r = useApi(`outcomes:symbol:${ticker}`, (signal) => outcomes.symbol(ticker, signal));
  // An API without /v1/outcomes: the older context below still renders. Any other failure
  // says so under the heading, with a retry, instead of leaving the section blank.
  if (r.error && !r.data) return r.error.status === 404 ? null : <ErrorState error={r.error} onRetry={r.reload} what={`${ticker}'s signal history`} />;
  if (!r.data) return <p className="cap">Loading this ticker&apos;s HSF record…</p>;
  if (!Array.isArray(r.data.horizons)) return null;
  return <SymbolLine d={r.data} />;
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
