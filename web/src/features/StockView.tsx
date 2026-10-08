"use client";

import Link from "next/link";
import { useState } from "react";
import type { FormEvent, MouseEvent } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { alerts } from "@/api/userData";
import { backLabel, usePreviousPage } from "@/components/AppShell";
import { Dialog } from "@/components/Dialog";
import { AIText } from "@/components/AIText";
import { Card, Disclaimer, Empty, ErrorLine, ErrorState, Freshness, Locked, Pill, Skeleton } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime, pct, price, probPct, setupLabel } from "@/lib/format";

import { AlertForm, TYPES_UNAVAILABLE, describeAlert, typesUnavailable } from "./AlertForm";
import { SymbolEvidence } from "./OutcomeEvidence";
import { PriceChart } from "./PriceChart";
import { SaveToWatchlistButton } from "./SaveToWatchlist";

type Stock = Schemas["StockDetail"];

const MOVEMENT: Record<string, string> = {
  RISING: "Rising since the previous scan", FALLING: "Falling since the previous scan", UNCHANGED: "Unchanged since the previous scan",
  NEW: "New to the ranked list", NO_BASELINE: "No earlier score to compare", VERSION_CHANGED: "Score model changed since the last reading",
};

function label(k: string): string {
  return k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function List({ items, empty }: { items: string[]; empty: string }) {
  return items.length ? <ul className="bullets">{items.map((r) => <li key={r}>{r}</li>)}</ul> : <p className="cap">{empty}</p>;
}

function Historical({ s }: { s: Stock }) {
  if (s.historical_locked) {
    return <Locked title="Historical research is part of Pro" plan="pro">How past HSF readings in this score range played out, and this ticker&apos;s own record.</Locked>;
  }
  const ctx = s.historical_context as { bucket?: string; positive_rate?: number; n?: number; confidence?: string; sufficient?: boolean } | null | undefined;
  const sum = s.history_summary as { observations?: number; matured?: number; positive?: number } | null | undefined;
  const coh = s.outcome_cohort as { status?: string; score_band?: string; horizon?: string; comparable?: number; follow_through_rate?: number | null; evidence_strength?: string } | null | undefined;
  return (
    <div className="stack-sm">
      <SymbolEvidence ticker={s.ticker} />
      {!ctx && !sum && !coh ? null : <p className="cap">Descriptive research on saved scans. Not a forecast and not evidence that a setup will work.</p>}
      {ctx && (ctx.sufficient && ctx.positive_rate !== undefined ? (
        <p className="body-sm">HSF {ctx.bucket}: {Math.round(ctx.positive_rate * 100)}% of past readings reached +4% within 5 days (n={ctx.n}, {String(ctx.confidence || "").toLowerCase()} confidence).</p>
      ) : (
        <p className="body-sm">HSF {ctx.bucket}: still building history{ctx.n ? ` (n=${ctx.n})` : ""}.</p>
      ))}
      {sum && !!sum.observations && (
        <p className="body-sm">This ticker: {sum.observations} earlier HSF observation{sum.observations === 1 ? "" : "s"}, {sum.matured} matured, {sum.positive} positive (observations, not trades).</p>
      )}
      {coh && coh.follow_through_rate !== null && coh.follow_through_rate !== undefined && (
        <p className="body-sm">Similar states ({coh.status}, HSF {coh.score_band}, {coh.horizon}): {coh.comparable} observations, {Math.round(coh.follow_through_rate * 100)}% persisted or strengthened ({String(coh.evidence_strength || "early").toLowerCase()} evidence).</p>
      )}
    </div>
  );
}

function PriceAlertButton({ s, onCreated }: { s: Stock; onCreated?: () => void }) {
  const [open, setOpen] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const types = useApi(open ? "alert-types" : null, (signal) => alerts.types(signal));
  const quota = useApi(open ? `alert-quota:${s.ticker}` : null, (signal) => alerts.list(signal));
  const full = !!quota.data && quota.data.used >= quota.data.limit;
  const close = () => { setOpen(false); setDone(null); };
  return (
    <>
      <button type="button" className="btn" onClick={() => setOpen(true)}>Set price alert</button>
      <Dialog open={open} title={`Price alert for ${s.ticker}`} onClose={close}>
        {done ? (
          <div className="stack-sm">
            <p className="notice" role="status">Created: {done}.</p>
            <div className="row-actions"><button type="button" className="btn" onClick={close} data-autofocus>Done</button><Link href="/alerts">See all alerts →</Link></div>
          </div>
        ) : types.error && typesUnavailable(types.error.status) ? <p className="notice" role="note">{TYPES_UNAVAILABLE}</p>
          : types.error || quota.error ? <ErrorLine error={types.error ?? quota.error} /> : !types.data || !quota.data ? <p className="cap">Loading…</p> : full ? (
          <div className="stack-sm">
            <p className="strong">You&apos;re using all {quota.data.limit} alert{quota.data.limit === 1 ? "" : "s"} on your plan.</p>
            <p className="cap">Delete one on the <Link href="/alerts">Alerts</Link> page, or upgrade for more.</p>
          </div>
        ) : (
          <>
            <p className="cap">Using {quota.data.used} of {quota.data.limit} alerts.{s.price != null ? ` Last scan price ${price(s.price)}${s.scan_at ? ` at ${etTime(s.scan_at)}` : ""}, not a live quote.` : ""}</p>
            <AlertForm types={types.data} lockType prefill={{ type: "price", ticker: s.ticker, threshold: s.price != null ? Math.round(s.price * 100) / 100 : undefined, direction: "above" }}
              onCreated={(a) => { setDone(describeAlert(a)); onCreated?.(); }} />
          </>
        )}
      </Dialog>
    </>
  );
}

/** Back to the page the user came from (keeping its filters and scroll), else the Scanner. */
function BackLink() {
  const prev = usePreviousPage();
  const label = backLabel(prev);
  if (!prev || !label) return <Link href="/scanner" className="back">← Back to Scanner</Link>;
  const back = (e: MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    window.history.back();
  };
  return <Link href={prev} className="back" onClick={back}>← {label}</Link>;
}

const DEFAULT_ACCOUNT = 10_000;
const DEFAULT_RISK = 1;

/** Pro: the API's trade plan for a name in the latest scan (stop from volatility, 1.5R and 3R targets). */
function TradePlan({ s }: { s: Stock }) {
  const [size, setSize] = useState(String(DEFAULT_ACCOUNT));
  const [risk, setRisk] = useState(String(DEFAULT_RISK));
  const [q, setQ] = useState({ size: DEFAULT_ACCOUNT, risk: DEFAULT_RISK });
  const plan = useApi(`plan:${s.ticker}:${q.size}:${q.risk}`, (signal) =>
    unwrap(api.GET("/v1/stocks/{ticker}/plan", { params: { path: { ticker: s.ticker }, query: { account_size: q.size, risk_pct: q.risk } }, signal })));
  const nSize = Number(size);
  const nRisk = Number(risk);
  const valid = Number.isFinite(nSize) && nSize >= 100 && nSize <= 1e9 && Number.isFinite(nRisk) && nRisk > 0 && nRisk <= 10;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (valid) setQ({ size: nSize, risk: nRisk });
  };
  const p = plan.data;
  return (
    <div className="stack-sm">
      <form className="inline-form" onSubmit={submit} aria-label="Trade plan settings">
        <label className="field"><span>Account size ($)</span><input inputMode="decimal" value={size} onChange={(e) => setSize(e.target.value)} /></label>
        <label className="field"><span>Risk per trade (%)</span><input inputMode="decimal" value={risk} onChange={(e) => setRisk(e.target.value)} /></label>
        <button type="submit" className="btn" disabled={!valid || plan.loading}>Update</button>
      </form>
      {!valid && <p className="form-error">Account size from $100 and risk above 0% up to 10%.</p>}
      {plan.error && !p ? (plan.error.status === 404 ? <p className="cap">{plan.error.message}</p> : <ErrorState error={plan.error} onRetry={plan.reload} what="the trade plan" />)
        : !p ? <Skeleton rows={3} label="Loading the trade plan" /> : (
          <dl className="kv">
            <div><dt>Entry (scan price)</dt><dd className="mono">{price(p.entry)}</dd></div>
            <div><dt>Stop</dt><dd className="mono">{price(p.stop)} <span className="cap">(−{p.stop_pct.toFixed(1)}%)</span></dd></div>
            {p.targets.map((t, i) => <div key={t}><dt>Target {i + 1}</dt><dd className="mono">{price(t)} <span className="cap">({p.target_r[i]}R)</span></dd></div>)}
            <div><dt>Shares</dt><dd className="mono">{p.shares.toLocaleString()} <span className="cap">risking {price(p.risk_budget)}</span></dd></div>
          </dl>
        )}
      <p className="cap">Stop at half the 20-day volatility (2–8%), targets at 1.5R and 3R. Educational only, not advice.</p>
    </div>
  );
}

/** Premium: an AI note on this name's latest scan result, generated on request. */
function AINote({ s }: { s: Stock }) {
  const act = useAction();
  const [note, setNote] = useState<Schemas["AIText"] | null>(null);
  const run = () => void act.run(async () => {
    setNote(await unwrap(api.POST("/v1/ai/notes/{ticker}", { params: { path: { ticker: s.ticker } } })));
    return true;
  });
  return (
    <div className="stack-sm">
      {note ? (note.text ? <AIText text={note.text} /> : <p className="cap">Nothing to explain: {s.ticker} isn&apos;t in the latest scan.</p>)
        : <p className="cap">A short research note on why {s.ticker} ranks where it does, written by Claude from the scan&apos;s data.</p>}
      <ErrorLine error={act.error} />
      <div className="row-actions">
        <button type="button" className="btn" onClick={run} disabled={act.busy}>{act.busy ? "Writing…" : note ? "Write it again" : "Write AI note"}</button>
      </div>
      <p className="cap">AI commentary can be wrong. Research only, not investment advice.</p>
    </div>
  );
}

export function StockView({ s, premium, pro = false, aiNotes = false, onChanged }: {
  s: Stock; premium: boolean; pro?: boolean; aiNotes?: boolean; onChanged?: () => void;
}) {
  const comps = Object.entries(s.score_components || {}).filter(([, v]) => Number.isFinite(v));
  const compMax = Math.max(1, ...comps.map(([, v]) => Math.abs(v)));
  const noScore = s.hsf_score === null || s.hsf_score === undefined;

  return (
    <div className="stack">
      <BackLink />
      <section className="page-head">
        <div className="stack-sm">
          <div className="title-row">
            <h1 className="h1 mono">{s.ticker}</h1>
            {s.primary_setup && <Pill>{setupLabel(s.primary_setup)}</Pill>}
            {s.status && <Pill tone={s.status === "STRONG" ? "up" : s.status === "CAUTION" || s.status === "FADING" ? "warn" : undefined}>{s.status}</Pill>}
          </div>
          {s.price !== null && s.price !== undefined ? (
            <div className="quote">
              <span className="mono big">{price(s.price)}</span>
              <span className={`mono ${(s.change_pct ?? 0) > 0 ? "up" : (s.change_pct ?? 0) < 0 ? "down" : "cap"}`}>{pct(s.change_pct)}</span>
              <span className="cap">{s.from_history ? "from the last recorded observation" : s.scan_at ? `at the ${etTime(s.scan_at)} scan` : ""}, not a live quote</span>
            </div>
          ) : <p className="cap">No price from the scans.</p>}
        </div>
        <div className="row-actions">
          <SaveToWatchlistButton ticker={s.ticker} inLists={s.watchlists.map((w) => w.id)} onSaved={onChanged} />
          <PriceAlertButton s={s} onCreated={onChanged} />
        </div>
      </section>

      {!s.in_latest_scan && (
        <p className="banner" role="status">
          {s.ticker} wasn&apos;t in the latest market scan{s.scan_at ? ` (${etTime(s.scan_at)})` : ""}.
          {s.from_history ? " The score below is from its last recorded HSF observation." : noScore ? " HSF has no recorded score for it." : ""}
        </p>
      )}
      {s.in_latest_scan && !s.has_setup && <p className="notice" role="status">Scanned, but it didn&apos;t qualify as an HSF setup in the latest scan.</p>}

      <div className="split stock-split">
        <div className="col-main">
          <Card title="Price" id="price" aside={s.bars_as_of ? `Bars as of ${etTime(s.bars_as_of)}` : undefined}>
            {s.bars.length >= 2 ? <PriceChart bars={s.bars} asOf={s.bars_as_of} /> : <Empty title="No price history cached for this ticker.">Charts appear once a scan has downloaded its daily bars.</Empty>}
          </Card>
          <Card title="Why it ranks" id="why">
            <List items={s.reasons} empty="No reasons recorded." />
          </Card>
          <div className="grid2">
            <Card title="Risks" id="risks"><List items={s.risks} empty="No specific risks flagged." /></Card>
            <Card title="What to watch" id="watch"><List items={s.watch_next} empty="Nothing specific to watch yet." /></Card>
          </div>
          <Card title="Historical context" id="hist"><Historical s={s} /></Card>
          {s.in_latest_scan && s.has_setup && (
            <Card title="Trade plan" id="plan">
              {pro ? <TradePlan s={s} /> : <Locked title="Trade plans are part of Pro" plan="pro">Entry, stop, targets and position size from your account size and risk.</Locked>}
            </Card>
          )}
          {aiNotes && s.in_latest_scan && <Card title="AI note" id="ai"><AINote s={s} /></Card>}
        </div>

        <aside className="col-side">
          <Card title="HSF Score" id="score" className="score-card" aside={s.scan_at ? <Freshness at={s.scan_at} /> : undefined}>
            {noScore ? <Empty title="No HSF Score">It needs to appear in a scan first.</Empty> : (
              <>
                <div className="score-ring" style={{ ["--p" as string]: `${s.hsf_score}%` }}>
                  <span className="mono">{s.hsf_score}</span>
                </div>
                <p className="cap">
                  {s.movement ? MOVEMENT[s.movement] ?? s.movement : ""}
                  {s.score_change ? ` (${s.score_change > 0 ? "+" : ""}${s.score_change})` : ""}
                </p>
                <p className="cap">An opportunity ranking, not a probability of profit.</p>
              </>
            )}
          </Card>

          {comps.length > 0 && (
            <Card title="Score components" id="comps">
              <ul className="comps">
                {comps.map(([k, v]) => (
                  <li key={k}><span className="cap">{label(k)}</span>
                    <div className="track" aria-hidden="true"><div className="fill" style={{ width: `${(Math.abs(v) / compMax) * 100}%` }} /></div>
                    <span className="mono">{v.toFixed(1)}</span></li>
                ))}
              </ul>
            </Card>
          )}

          <Card title="Signals" id="signals">
            {s.signals.length ? <div className="chips">{s.signals.map((g) => <Pill key={g}>{setupLabel(g)}</Pill>)}</div> : <p className="cap">No active signals.</p>}
            <dl className="kv">
              <div><dt>Breakout score</dt><dd className="mono">{s.breakout_score ?? "—"}</dd></div>
              <div><dt>PreBreakout</dt><dd className="mono">{premium ? probPct(s.prob) : <Pill tone="gold">Premium</Pill>}</dd></div>
              <div><dt>Earnings</dt><dd className="mono">{s.earnings_days === null || s.earnings_days === undefined ? "—" : s.earnings_days === 0 ? "Today" : `in ${s.earnings_days}d`}</dd></div>
            </dl>
          </Card>

          <Card title="Your lists and alerts" id="mine">
            {s.watchlists.length ? <div className="chips">{s.watchlists.map((w) => <Link key={w.id} href={`/watchlists?id=${w.id}`} className="pill">{w.name}</Link>)}</div> : <p className="cap">Not on your watchlists.</p>}
            {s.alerts.length ? (
              <ul className="bullets">{s.alerts.map((a) => <li key={a.id}>{describeAlert(a)}{a.enabled ? "" : " (off)"}</li>)}</ul>
            ) : <p className="cap">No alerts on {s.ticker}.</p>}
          </Card>

          {s.lifecycle.length > 0 && (
            <Card title="Recent history" id="life">
              <ul className="timeline">
                {s.lifecycle.slice(-6).reverse().map((e, i) => (
                  <li key={`${e.time}-${i}`}><span className="cap">{etTime(e.time)}</span> <span className="mono">{e.score ?? "—"}</span> {e.label || e.status || ""}</li>
                ))}
              </ul>
            </Card>
          )}
        </aside>
      </div>
      <Disclaimer />
    </div>
  );
}

