"use client";

// Outcome Intelligence views (/v1/outcomes/*). Evidence first: every number sits next
// to its sample size and evidence label, every bucket/horizon/setup is shown (weak ones
// included), and nothing is picked as "best". Returns are fractions from the API.
import { outcomes } from "@/api/account";
import type { OutcomeGroups, OutcomeHorizons, OutcomeMetrics, OutcomeScores, OutcomeSummary, OutcomeSymbol } from "@/api/account";
import { Card, Empty, ErrorState, Pill, Skeleton } from "@/components/ui";
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
  return <Pill tone={q === "INSUFFICIENT" || q === "LIMITED" ? "warn" : undefined}>{QUALITY[q] ?? q} evidence</Pill>;
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
        <th scope="col" className="num hide-narrow">Avg MFE / MAE</th>
      </tr>
    </thead>
  );
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

function ScoresCard({ d }: { d: OutcomeScores }) {
  const inv = d.calibration.metrics["median_excess_return"]?.inversions ?? [];
  return (
    <Card title="By HSF score" id="scores">
      <p className="cap">Every score range, including weak ones, over {d.horizon} trading days. Ranges with fewer than {d.evidence_thresholds["LIMITED"]} matured signals are greyed: read them as anecdotes.</p>
      <div className="table-wrap">
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
      <div className="table-wrap">
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
      <div className="table-wrap">
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
export function SymbolEvidence({ ticker }: { ticker: string }) {
  const r = useApi(`outcomes:symbol:${ticker}`, (signal) => outcomes.symbol(ticker, signal));
  if (r.error && !r.data) return null;   // the older context above still renders
  if (!r.data) return <p className="cap">Loading this ticker&apos;s HSF record…</p>;
  if (!Array.isArray(r.data.horizons)) return null;
  return <SymbolLine d={r.data} />;
}

export function SymbolLine({ d }: { d: OutcomeSymbol }) {
  const h = d.horizons.find((x) => x.horizon === 5) ?? d.horizons[d.horizons.length - 1];
  if (!h || h.sample_size === 0) return <p className="body-sm">No earlier HSF signals for {d.ticker}.</p>;
  if (h.matured_count === 0) return <p className="body-sm">{d.ticker}: {h.sample_size} earlier HSF signal{h.sample_size === 1 ? "" : "s"}, none matured yet.</p>;
  return (
    <div className="stack-sm">
      <p className="body-sm">
        <strong>Historical HSF evidence:</strong> {h.matured_count} matured signal{h.matured_count === 1 ? "" : "s"} for {d.ticker} ({range(d.date_range)}).
        {h.benchmark_count > 0 && <> {h.benchmark_beat_count} of {h.benchmark_count} beat SPY over 5 trading days.</>}
        {" "}Median return {signed(h.median_return)}{h.benchmark_count > 0 && <>, median vs SPY {signed(h.median_excess_return)}</>}.
        {h.pending_count > 0 && <> {h.pending_count} still pending.</>}
      </p>
      <p className="cap"><Quality q={h.evidence_quality} /> Past signals only; not a forecast.</p>
    </div>
  );
}
