"use client";

import Link from "next/link";

import type { Schemas } from "@/api/client";
import { Card, Disclaimer, Empty, Freshness, Locked, Pill } from "@/components/ui";
import { etTime, pct, price, setupLabel } from "@/lib/format";

import { PriceChart } from "./PriceChart";

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
  if (!ctx && !sum && !coh) return <p className="cap">No historical context for this ticker yet.</p>;
  return (
    <div className="stack-sm">
      <p className="cap">Descriptive research on saved scans. Not a forecast and not evidence that a setup will work.</p>
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

export function StockView({ s, premium }: { s: Stock; premium: boolean }) {
  const comps = Object.entries(s.score_components || {}).filter(([, v]) => Number.isFinite(v));
  const compMax = Math.max(1, ...comps.map(([, v]) => Math.abs(v)));
  const noScore = s.hsf_score === null || s.hsf_score === undefined;

  return (
    <div className="stack">
      <Link href="/scanner" className="back">← Back to Scanner</Link>
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
              <div><dt>PreBreakout</dt><dd className="mono">{premium ? (s.prob !== null && s.prob !== undefined ? `${Math.round(s.prob * 100)}%` : "—") : <Pill tone="gold">Premium</Pill>}</dd></div>
              <div><dt>Earnings</dt><dd className="mono">{s.earnings_days === null || s.earnings_days === undefined ? "—" : s.earnings_days === 0 ? "Today" : `in ${s.earnings_days}d`}</dd></div>
            </dl>
          </Card>

          <Card title="Your lists and alerts" id="mine">
            {s.watchlists.length ? <div className="chips">{s.watchlists.map((w) => <Pill key={w.id}>{w.name}</Pill>)}</div> : <p className="cap">Not on your watchlists.</p>}
            {s.alerts.length ? (
              <ul className="bullets">{s.alerts.map((a) => <li key={a.id}>{label(a.type)}{a.threshold !== null && a.threshold !== undefined ? ` ${a.direction ?? ""} ${a.threshold}` : ""}{a.enabled ? "" : " (off)"}</li>)}</ul>
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

