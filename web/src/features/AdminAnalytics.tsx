"use client";

// Admin analytics (admins only), moved from the classic app's Admin views: product analytics,
// signal performance and HSF Score calibration, Stair-Stepper evidence, research evidence,
// ML v4 readiness and Autonomous Recovery. Read-only; the API caches each view 10 minutes.
import { useState } from "react";
import type { ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, Empty, ErrorState, Pill, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";

type Group = Schemas["AdminPerfGroup"];
const PERIODS = [[7, "7 days"], [30, "30 days"], [90, "90 days"], [365, "1 year"]] as const;

/** A fraction (0.42) as a percent ("42.0%"); signed for returns ("+4.2%"). */
export function frac(v: number | null | undefined, signed = false): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const p = v * 100;
  const rounded = Number(p.toFixed(signed && Math.abs(p) < 0.1 ? 3 : 1));
  return `${signed && rounded > 0 ? "+" : ""}${rounded.toFixed(signed && Math.abs(p) > 0 && Math.abs(p) < 0.1 ? 3 : 1)}%`;
}
const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");
const fmt = (n: number) => n.toLocaleString("en-US");

function Period({ days, onChange }: { days: number; onChange: (d: number) => void }) {
  return (
    <label className="field"><span>Period</span>
      <select value={days} onChange={(e) => onChange(Number(e.target.value))}>
        {PERIODS.map(([d, label]) => <option key={d} value={d}>{label}</option>)}
      </select>
    </label>
  );
}

function Tiles({ items }: { items: [string, string][] }) {
  return (
    <div className="tiles">
      {items.map(([label, value]) => (
        <div key={label} className="tile"><span className="cap">{label}</span><p className="tile-value small">{value}</p></div>
      ))}
    </div>
  );
}

function StatusPill({ status }: { status: string | null | undefined }) {
  const s = status ?? "UNKNOWN";
  const good = ["HEALTHY", "READY", "PASS", "SUFFICIENT"].includes(s);
  const bad = ["CRITICAL", "FAILED", "FAIL", "NOT_READY"].includes(s);
  return <Pill tone={good ? "up" : bad ? "down" : "warn"}>{s.replaceAll("_", " ")}</Pill>;
}

/** Stacked daily bars (one colour per series) with a text summary for screen readers. */
function Bars({ label, days, series }: { label: string; days: string[]; series: { name: string; values: number[]; cls: string }[] }) {
  const w = 600, h = 140;
  const totals = days.map((_, i) => series.reduce((s, x) => s + (x.values[i] ?? 0), 0));
  const max = Math.max(1, ...totals);
  const bw = w / Math.max(1, days.length);
  const sums = series.map((s) => `${s.name} ${fmt(s.values.reduce((a, b) => a + b, 0))}`).join(", ");
  return (
    <figure className="admin-chart">
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label={`${label}: ${sums} from ${days[0]} to ${days.at(-1)}`}>
        {days.map((d, i) => {
          let y = h;
          return (
            <g key={d}>
              <title>{`${d}: ${series.map((s) => `${s.name} ${s.values[i] ?? 0}`).join(", ")}`}</title>
              {series.map((s) => {
                const bh = ((s.values[i] ?? 0) / max) * (h - 4);
                y -= bh;
                return bh > 0 ? <rect key={s.name} className={s.cls} x={i * bw + bw * 0.1} y={y} width={bw * 0.8} height={bh} /> : null;
              })}
            </g>
          );
        })}
      </svg>
      <figcaption className="cap">
        {series.map((s) => <span key={s.name} className="legend"><i className={s.cls} aria-hidden="true" /> {s.name}</span>)}
        <span>{days[0]} to {days.at(-1)}</span>
      </figcaption>
    </figure>
  );
}

function Line({ label, days, values }: { label: string; days: string[]; values: number[] }) {
  const w = 600, h = 140;
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values.map((v, i) => `${((i / Math.max(1, values.length - 1)) * w).toFixed(1)},${(h - 4 - ((v - lo) / span) * (h - 8)).toFixed(1)}`).join(" ");
  return (
    <figure className="admin-chart">
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label={`${label}: ${fmt(values[0] ?? 0)} on ${days[0]} to ${fmt(values.at(-1) ?? 0)} on ${days.at(-1)}`}>
        <polyline points={pts} fill="none" className="admin-line" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption className="cap"><span>{days[0]}: {fmt(values[0] ?? 0)}</span> <span>{days.at(-1)}: {fmt(values.at(-1) ?? 0)}</span></figcaption>
    </figure>
  );
}

/** One row per value, a bar scaled to the largest. */
function HBars({ rows, label }: { rows: [string, number, string?][]; label: string }) {
  const max = Math.max(1, ...rows.map((r) => r[1]));
  return (
    <ul className="stack-sm" aria-label={label}>
      {rows.map(([name, n, note]) => (
        <li key={name} className="scorebar">
          <span className="admin-bar-label">{name}</span>
          <div className="track" aria-hidden="true"><div className="fill" style={{ width: `${(n / max) * 100}%` }} /></div>
          <span className="mono">{fmt(n)}{note ? <span className="cap"> {note}</span> : null}</span>
        </li>
      ))}
    </ul>
  );
}

function useView<T>(key: string, load: (signal: AbortSignal) => Promise<T>, what: string, render: (data: T) => ReactNode) {
  const s = useApi(key, load);
  if (s.error && !s.data) return <ErrorState error={s.error} onRetry={s.reload} what={what} />;
  if (!s.data) return <Skeleton rows={5} label={`Loading ${what}`} />;
  return <>{render(s.data)}</>;
}

// ---- analytics tab -------------------------------------------------------------------------------
export function AnalyticsTab() {
  const [days, setDays] = useState(30);
  const view = useView(`admin-analytics:${days}`, (signal) => unwrap(api.GET("/v1/admin/analytics", { params: { query: { days } }, signal })),
    "product analytics", (a) => <Product a={a} />);
  return (
    <div className="stack">
      <div className="inline-form"><Period days={days} onChange={setDays} /></div>
      {view}
    </div>
  );
}

function Product({ a }: { a: Schemas["AdminAnalytics"] }) {
  const o = a.overview;
  const days = a.users_daily.map((d) => d.day);
  const c = a.conversions;
  return (
    <>
      <Card title="Control center" id="overview" aside={`Updated ${etTime(a.loaded_at)}`}>
        <Tiles items={[["Total users", fmt(o.total_users)], ["Active users (30d)", fmt(o.active_users_30d)],
          ["New users (7d)", fmt(o.new_users_7d)], ["Scans today", fmt(o.scans_today)], ["Paying users", fmt(o.paying_users)]]} />
        {a.unavailable.length > 0 && <p className="cap">Some sources are unavailable: {a.unavailable.join(", ")}. The rest is shown.</p>}
      </Card>
      <Card title="HSF system status" id="systems">
        <ul className="stack-sm" aria-label="Systems">
          {a.systems.map((s) => <li key={s.system}><StatusPill status={s.status} /> <strong>{s.system}</strong> <span className="cap">{s.reason}</span></li>)}
        </ul>
      </Card>
      <Card title="Users over time" id="users-over-time">
        <Line label="Total users" days={days} values={a.users_daily.map((d) => d.total)} />
        <p className="cap">{fmt(a.users_daily.reduce((s, d) => s + d.new, 0))} new in this period.</p>
      </Card>
      <Card title="Current plan distribution" id="plans" aside="Active accounts">
        <HBars label="Accounts by plan" rows={a.plans.map((p) => [p.plan, p.users])} />
      </Card>
      <Card title="Scan activity" id="scans">
        <Bars label="Scans per day" days={a.scans_daily.map((d) => d.day)} series={[
          { name: "Scheduled", values: a.scans_daily.map((d) => d.scheduled), cls: "bar-a" },
          { name: "User", values: a.scans_daily.map((d) => d.user), cls: "bar-b" }]} />
        <p className="cap">Scheduled scans have no account; user scans are account-scoped runs.</p>
      </Card>
      <Card title="Recorded conversion events" id="conversions">
        <Tiles items={[["Unique signup accounts", fmt(c.signups)], ["Unique paid accounts", fmt(c.paid)], ["Recorded signup to paid", frac(c.signup_to_paid)]]} />
        {c.destinations.length > 0 && <HBars label="Paid destination" rows={c.destinations.map((d) => [d.plan, d.users])} />}
        <p className="cap">Counts cover the selected period and deduplicate accounts. The rate measures accounts with both signup and payment recorded in that period; destination bars count events. Only recorded paid-conversion events count. Plan changes are not inferred from current plans, because there is no complete history of them.</p>
      </Card>
      <details className="research-section"><summary>Recovery history</summary><Card title="Autonomous Recovery" id="recovery">
        {a.recovery.length === 0 ? <p className="cap">No Autonomous Recovery events recorded.</p> : (
          <div className="table-wrap"><table className="table">
            <thead><tr><th scope="col">Started</th><th scope="col">Incident</th><th scope="col">Action</th><th scope="col">Result</th></tr></thead>
            <tbody>{a.recovery.map((r, i) => (
              <tr key={`${r.started_at}-${i}`}><td className="cap">{etTime(r.started_at)}</td><td>{r.incident?.replaceAll("_", " ").toLowerCase()}</td><td>{r.action?.replaceAll("_", " ").toLowerCase()}</td><td>{r.result?.replaceAll("_", " ").toLowerCase()}</td></tr>
            ))}</tbody>
          </table></div>
        )}
      </Card></details>
    </>
  );
}

// ---- research tab --------------------------------------------------------------------------------
const METRICS = [["median_return", "Median return"], ["average_return", "Average return"], ["positive_rate", "Positive outcome %"],
  ["n", "Observation count"], ["average_mfe", "MFE"], ["average_mae", "MAE"]] as const;
type Metric = (typeof METRICS)[number][0];

function cell(g: Group | undefined, metric: Metric, minN: number): { text: string; cls: string } {
  if (!g) return { text: "", cls: "" };
  if (metric === "n") return { text: fmt(g.n), cls: "" };
  if (g.n < minN) return { text: `n=${g.n}`, cls: "cap" };
  const v = g[metric];
  return { text: frac(v, metric !== "positive_rate"), cls: metric === "positive_rate" ? "" : tone(v) };
}

function Heatmap({ groups, metric, minN, rowLabel }: { groups: Group[]; metric: Metric; minN: number; rowLabel: string }) {
  const rows = [...new Set(groups.map((g) => g.key ?? "(none)"))];
  const cols = [...new Set(groups.map((g) => g.column ?? "(none)"))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  if (groups.length === 0) return <p className="cap">No matured observations for this matrix yet.</p>;
  const at = (r: string, c: string) => groups.find((g) => (g.key ?? "(none)") === r && (g.column ?? "(none)") === c);
  return (
    <div className="table-wrap"><table className="table">
      <thead><tr><th scope="col">{rowLabel}</th>{cols.map((c) => <th key={c} scope="col" className="num">{c}</th>)}</tr></thead>
      <tbody>{rows.map((r) => (
        <tr key={r}><th scope="row">{r}</th>{cols.map((c) => { const v = cell(at(r, c), metric, minN); return <td key={c} className={`num mono ${v.cls}`}>{v.text}</td>; })}</tr>
      ))}</tbody>
    </table></div>
  );
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <label className="field"><span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">All</option>
        {[...options].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );
}

function Signals({ days }: { days: number }) {
  const [f, setF] = useState({ signal: "", cohort: "", horizon: "", session: "", score_horizon: "" });
  const [minN, setMinN] = useState(30);
  const [scoreMin, setScoreMin] = useState("");
  const [scoreMax, setScoreMax] = useState("");
  const [metric, setMetric] = useState<Metric>("median_return");
  const query = { days, min_n: minN, ...f, ...(scoreMin ? { score_min: Number(scoreMin) } : {}), ...(scoreMax ? { score_max: Number(scoreMax) } : {}) };
  const s = useApi(`admin-signals:${JSON.stringify(query)}`, (signal) => unwrap(api.GET("/v1/admin/signals", { params: { query }, signal })));
  const set = (k: keyof typeof f) => (v: string) => setF((x) => ({ ...x, [k]: v }));
  const d = s.data;
  return (
    <Card title="Signal performance" id="signals">
      {s.error && !d ? <ErrorState error={s.error} onRetry={s.reload} what="signal performance" />
        : !d ? <Skeleton rows={5} label="Loading signal performance" />
        : d.options.signals.length === 0 ? <Empty title="No scanner research observations in this period." />
        : (
          <div className="stack">
            <div className="inline-form">
              <Select label="Signal" value={f.signal} options={d.options.signals} onChange={set("signal")} />
              <Select label="Cohort" value={f.cohort} options={d.options.cohorts} onChange={set("cohort")} />
              <Select label="Horizon" value={f.horizon} options={d.options.horizons} onChange={set("horizon")} />
              <Select label="Session" value={f.session} options={d.options.sessions} onChange={set("session")} />
              <label className="field"><span>Minimum sample</span>
                <input type="number" min={1} max={1000} value={minN} onChange={(e) => setMinN(Math.max(1, Math.min(1000, Number(e.target.value) || 1)))} /></label>
              {d.options.score_low != null && (
                <>
                  <label className="field"><span>HSF Score from</span><input type="number" value={scoreMin} placeholder={String(d.options.score_low)} onChange={(e) => setScoreMin(e.target.value)} /></label>
                  <label className="field"><span>to</span><input type="number" value={scoreMax} placeholder={String(d.options.score_high)} onChange={(e) => setScoreMax(e.target.value)} /></label>
                </>
              )}
            </div>
            <Tiles items={[["Observations", fmt(d.summary.observations)], ["Matured outcome rows", fmt(d.summary.matured)],
              ["Positive outcome", frac(d.summary.positive_rate)], ["Median forward return", frac(d.summary.median_return, true)],
              ["Average forward return", frac(d.summary.average_return, true)], ["Average MFE", frac(d.summary.average_mfe, true)],
              ["Average MAE", frac(d.summary.average_mae, true)], ["Evidence", d.summary.sufficient ? "Sufficient" : "Insufficient"]]} />
            <p className="cap">Missing outcomes are left out, never counted as zero.{d.truncated ? ` Based on the newest ${fmt(d.loaded_observations)} observations in this period.` : ""}</p>
            <label className="field"><span>Matrix metric</span>
              <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>{METRICS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
            </label>
            <h3 className="h3">Signal by horizon</h3>
            <Heatmap groups={d.by_signal_horizon} metric={metric} minN={d.min_n} rowLabel="Signal" />
            <p className="cap">Cells under the evidence threshold (n &lt; {d.min_n}) show their sample count only.</p>
            <h3 className="h3">HSF Score calibration</h3>
            {!d.score_available ? <p className="cap">The canonical HSF Score isn&apos;t stored on these observations, so calibration is unavailable. Breakout Score is not substituted.</p> : (
              <>
                <Select label="Calibration horizon" value={d.score_horizon ?? ""} options={d.options.horizons} onChange={set("score_horizon")} />
                <div className="table-wrap"><table className="table">
                  <thead><tr><th scope="col">HSF Score</th><th scope="col" className="num">n</th><th scope="col" className="num">Positive</th><th scope="col" className="num">Median</th><th scope="col" className="num">Average</th></tr></thead>
                  <tbody>{d.score_buckets.map((b) => (
                    <tr key={b.key}><th scope="row">{b.key}</th><td className="num mono">{fmt(b.n)}</td><td className="num mono">{frac(b.positive_rate)}</td>
                      <td className={`num mono ${tone(b.median_return)}`}>{frac(b.median_return, true)}</td><td className={`num mono ${tone(b.average_return)}`}>{frac(b.average_return, true)}</td></tr>
                  ))}</tbody>
                </table></div>
                <Heatmap groups={d.score_matrix} metric={metric} minN={d.min_n} rowLabel="HSF Score" />
              </>
            )}
          </div>
        )}
    </Card>
  );
}

function StairStepper({ days }: { days: number }) {
  const [horizon, setHorizon] = useState("+10m");
  return useView(`admin-stair:${days}`, (signal) => unwrap(api.GET("/v1/admin/stair-stepper", { params: { query: { days } }, signal })),
    "Stair-Stepper evidence", (d) => {
      const rows = d.windows.filter((w) => w.horizon === horizon);
      return (
        <Card title="Stair-Stepper evidence" id="stair" aside={`Latest ${etTime(d.latest)}`}>
          {d.observations === 0 ? <Empty title="No Stair-Stepper outcomes have matured in this period." /> : (
            <div className="stack-sm">
              <label className="field"><span>Outcome horizon</span>
                <select value={horizon} onChange={(e) => setHorizon(e.target.value)}>{[...d.horizons].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).map((h) => <option key={h} value={h}>{h}</option>)}</select>
              </label>
              <div className="table-wrap"><table className="table">
                <thead><tr><th scope="col">Window (bars)</th><th scope="col" className="num">Observations</th><th scope="col" className="num">Matured</th><th scope="col" className="num">Positive</th>
                  <th scope="col" className="num">Average</th><th scope="col" className="num">Median</th><th scope="col" className="num hide-narrow">MFE</th><th scope="col" className="num hide-narrow">MAE</th></tr></thead>
                <tbody>{rows.map((w) => (
                  <tr key={String(w.window)}><th scope="row">{String(w.window)}</th><td className="num mono">{w.observations ?? 0}</td><td className="num mono">{w.matured}</td>
                    <td className="num mono">{frac(w.positive_rate)}</td><td className={`num mono ${tone(w.average_return)}`}>{frac(w.average_return, true)}</td>
                    <td className={`num mono ${tone(w.median_return)}`}>{frac(w.median_return, true)}</td><td className="num mono hide-narrow">{frac(w.mfe, true)}</td><td className="num mono hide-narrow">{frac(w.mae, true)}</td></tr>
                ))}</tbody>
              </table></div>
            </div>
          )}
          <p className="cap">{d.best_window != null ? <><Pill tone="up">Best-supported window: {String(d.best_window)} bars</Pill> </> : <><Pill tone="warn">No window has enough evidence yet</Pill> </>}{d.verdict_reason}</p>
          <p className="cap">Kept separate from scheduled scanner research.</p>
        </Card>
      );
    });
}

function Evidence({ days }: { days: number }) {
  return useView(`admin-evidence:${days}`, (signal) => unwrap(api.GET("/v1/admin/evidence", { params: { query: { days } }, signal })),
    "research evidence", (e) => (
      <Card title="Research evidence" id="evidence">
        <p className="cap">Latest observation {etTime(e.latest_observation)} · latest matured {etTime(e.latest_matured)} · last maturation run {etTime(e.last_maturation_run)}</p>
        {e.truncated && <p className="cap">Based on the newest {fmt(e.loaded_observations)} observations in this period.</p>}
        <HBars label="Evidence funnel" rows={e.funnel.map((s) => [s.stage, s.count, s.pct != null ? `(${s.pct}%)` : undefined])} />
        <Tiles items={[["Pending", fmt(e.pending)], ["Excluded", fmt(e.excluded)], ["Metadata complete", frac(e.metadata_completeness)], ["Oldest pending", etTime(e.oldest_pending)]]} />
        <h3 className="h3">Cohort coverage</h3>
        {e.cohorts.length === 0 ? <p className="cap">No research cohorts in this period.</p> : <HBars label="Observations by cohort" rows={e.cohorts.map((c) => [c.cohort, c.observations])} />}
        <h3 className="h3">Evidence readiness</h3>
        <p><StatusPill status={e.readiness.state} /> <span className="cap">{e.readiness.reason ?? "No readiness explanation recorded."}</span></p>
        {e.readiness.trading_days != null && (
          <div className="metric-strip">
            <span>Trading days <strong>{String(e.readiness.trading_days)} / {String(e.readiness.trading_days_preferred ?? "—")}</strong></span>
            <span>Scan runs <strong>{String(e.readiness.scan_runs ?? "—")} / {String(e.readiness.scan_runs_preferred ?? "—")}</strong></span>
            <span>Formal evaluation <strong>{e.readiness.formal_evaluation ?? "UNKNOWN"}</strong></span>
          </div>
        )}
      </Card>
    ));
}

function MlReadiness() {
  return useView("admin-ml-readiness", (signal) => unwrap(api.GET("/v1/admin/ml-readiness", { signal })), "ML v4 readiness", (m) => (
    <Card title="ML v4 readiness" id="ml" aside={`Computed ${etTime(m.generated_at)}`}>
      <p><StatusPill status={m.status} /> {m.recommendation}</p>
      {m.blocking_reasons.length > 0 && <p className="notice">Review required: {m.blocking_reasons.join("; ")}</p>}
      <Tiles items={[["Matured signal-days", fmt(m.matured_signal_days)], ["Trading-day coverage", fmt(m.matured_trading_days)],
        ["Unique symbols", fmt(m.matured_symbols)], ["Outcome coverage (recent)", frac(m.outcome_coverage_recent)], ["Walk-forward folds", fmt(m.walk_forward_folds)]]} />
      <div className="table-wrap"><table className="table">
        <thead><tr><th scope="col">Gate</th><th scope="col" className="hide-narrow">Kind</th><th scope="col">Value</th><th scope="col">Needs</th><th scope="col">Progress</th></tr></thead>
        <tbody>{[...m.gates].sort((a, b) => Number(b.status !== "PASS") - Number(a.status !== "PASS")).map((g) => (
          <tr key={g.gate}><th scope="row">{g.gate.replaceAll("_", " ")}{g.why ? <details><summary>Why this gate?</summary><p className="cap">{g.why}</p></details> : null}</th><td className="cap hide-narrow">{g.kind}</td>
            <td className="mono">{typeof g.value === "number" ? fmt(Math.round(g.value * 1000) / 1000) : String(g.value ?? "—")}</td><td className="mono">{g.needs}</td>
            <td><div className="scorebar"><div className="track" aria-hidden="true"><div className="fill" style={{ width: `${Math.min(100, g.progress * 100)}%` }} /></div>
              <StatusPill status={g.status} /></div></td></tr>
        ))}</tbody>
      </table></div>
      {m.blocking_reasons.length > 0 && <p className="cap">Unsatisfied: {m.blocking_reasons.join("; ")}</p>}
      {m.estimate_trading_days != null && <p className="cap">Volume targets reachable in roughly {m.estimate_trading_days} trading days at the recent rate (an estimate; structural gates need fixes first).</p>}
    </Card>
  ));
}

export function ResearchTab() {
  const [days, setDays] = useState(30);
  return (
    <div className="stack">
      <div className="inline-form"><Period days={days} onChange={setDays} /></div>
      <Signals key={days} days={days} />
      <details className="research-section"><summary>Research evidence</summary><Evidence days={days} /></details>
      <details className="research-section"><summary>Stair-Stepper evidence</summary><StairStepper days={days} /></details>
      <details className="research-section"><summary>ML v4 readiness and validation gates</summary><MlReadiness /></details>
    </div>
  );
}
