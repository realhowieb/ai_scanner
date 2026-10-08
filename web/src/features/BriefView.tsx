"use client";

// Market Brief: the day's snapshot (indexes, breadth, sectors, top opportunities with
// movement since the previous snapshot, movers, golden crosses), the earnings calendar
// (Pro) and an AI narrative (Premium, on request). Everything comes from /v1/brief,
// /v1/earnings and /v1/ai/brief-narrative; sections the API leaves empty say so.
import { useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, Disclaimer, Empty, ErrorLine, ErrorState, Locked, Pill, ScoreBadge, Skeleton, TickerLink } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { etTime, pct, price, probPct } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

type Brief = Schemas["Brief"];
type Opp = { ticker?: string; score?: number | null; status?: string | null; movement_state?: string | null; score_delta?: number | null };

const PHASE: Record<string, string> = { premarket: "Pre-market", regular: "Market open", afterhours: "After hours", closed: "Market closed" };
const MOVE: Record<string, string> = { RISING: "Rising", FALLING: "Falling", UNCHANGED: "Unchanged", NEW: "New", NO_BASELINE: "", VERSION_CHANGED: "Model changed" };

const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");

function MoverList({ rows, empty, plain = false }: { rows: { ticker: string; chg_pct?: number | null; extra?: string }[]; empty: string; plain?: boolean }) {
  if (!rows.length) return <p className="cap">{empty}</p>;
  return (
    <ul className="rows">
      {rows.map((r) => (
        <li key={r.ticker} className="row">
          {plain ? <span className="sector">{r.ticker}</span> : <TickerLink ticker={r.ticker} />}
          <span className={`mono strong move ${tone(r.chg_pct)}`}>{pct(r.chg_pct)}</span>
          <span className="cap grow">{r.extra ?? ""}</span>
        </li>
      ))}
    </ul>
  );
}

function Earnings() {
  const { can } = useSession();
  const allowed = can("can_earnings");
  const e = useApi(allowed ? "earnings:7" : null, (signal) => unwrap(api.GET("/v1/earnings", { params: { query: { days: 7 } }, signal })));
  return (
    <Card title="Earnings this week" id="earnings">
      {!allowed ? <Locked title="The earnings calendar is part of Pro" plan="pro">Which names report in the next 7 days, before or after the bell.</Locked>
        : e.error && !e.data ? <ErrorState error={e.error} onRetry={e.reload} what="the earnings calendar" />
        : !e.data ? <Skeleton rows={4} label="Loading earnings" />
        : e.data.length === 0 ? <Empty title="No earnings on file for the next 7 days." />
        : (
          <ul className="rows">
            {e.data.slice(0, 40).map((r) => (
              <li key={`${r.ticker}-${r.earnings_date}`} className="row">
                <TickerLink ticker={r.ticker} />
                <span className="cap grow">{r.days_until === 0 ? "Today" : r.days_until === 1 ? "Tomorrow" : r.earnings_date ?? ""}</span>
                {r.time && <Pill>{r.time === "bmo" ? "Before open" : r.time === "amc" ? "After close" : r.time.toUpperCase()}</Pill>}
              </li>
            ))}
          </ul>
        )}
      {allowed && e.data && e.data.length > 40 && <p className="cap">Showing the first 40 of {e.data.length}.</p>}
    </Card>
  );
}

function Narrative() {
  const act = useAction();
  const [text, setText] = useState<Schemas["AIText"] | null>(null);
  return (
    <Card title="AI market narrative" id="ai">
      {text?.text ? <div className="ai-text body-sm">{text.text}</div> : text ? <p className="cap">Nothing to summarize yet today.</p>
        : <p className="cap">A short read of today&apos;s brief, written by Claude from the data on this page.</p>}
      <ErrorLine error={act.error} />
      <div className="row-actions">
        <button type="button" className="btn" disabled={act.busy}
          onClick={() => void act.run(async () => { setText(await unwrap(api.GET("/v1/ai/brief-narrative"))); return true; })}>
          {act.busy ? "Writing…" : text ? "Write it again" : "Write narrative"}
        </button>
      </div>
      <p className="cap">AI commentary can be wrong. Research only, not investment advice.</p>
    </Card>
  );
}

function BriefBody({ b }: { b: Brief }) {
  const { can } = useSession();
  const opps = (b.opportunities as Opp[]).filter((o) => o.ticker);
  return (
    <>
      {b.market.length > 0 && (
        <section aria-label="Indexes" className="index-strip">
          {b.market.map((m) => (
            <div key={m.label} className="index-tile">
              <span className="cap">{m.label}</span>
              <span className="mono strong">{m.last != null ? m.last.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "—"}</span>
              <span className={`mono cap ${tone(m.chg_pct)}`}>{pct(m.chg_pct)}</span>
            </div>
          ))}
          {b.breadth && (
            <div className="index-tile">
              <span className="cap">Breadth</span>
              <span className="mono strong"><span className="up">{b.breadth.advancers ?? 0}</span> / <span className="down">{b.breadth.decliners ?? 0}</span></span>
              <span className="cap">advancing / declining</span>
            </div>
          )}
        </section>
      )}

      <div className="split">
        <div className="col-main">
          <Card title="Top opportunities" id="opps" aside={b.has_previous_snapshot ? "Movement since the previous snapshot" : "First snapshot today"}>
            {opps.length === 0 ? <Empty title="No ranked opportunities in this snapshot." /> : (
              <ul className="rows">
                {opps.map((o) => (
                  <li key={o.ticker} className="row">
                    <TickerLink ticker={o.ticker!} />
                    <ScoreBadge score={o.score} />
                    {o.status && <Pill tone={o.status === "STRONG" ? "up" : o.status === "FADING" || o.status === "CAUTION" ? "warn" : undefined}>{o.status}</Pill>}
                    <span className="grow" />
                    <span className={`cap ${tone(o.score_delta)}`}>{MOVE[o.movement_state ?? ""] ?? ""}{o.score_delta ? ` ${o.score_delta > 0 ? "+" : ""}${o.score_delta}` : ""}</span>
                  </li>
                ))}
              </ul>
            )}
          </Card>
          <div className="grid2">
            <Card title="Gappers" id="gap">
              <MoverList rows={b.gappers.map((g) => ({ ticker: g.ticker, chg_pct: g.gap_pct, extra: `${price(g.last)}${g.earnings_days != null ? ` · earnings in ${g.earnings_days}d` : ""}` }))} empty="No gappers in this snapshot." />
            </Card>
            <Card title="Sectors" id="sectors">
              <MoverList rows={b.sectors.map((s) => ({ ticker: String(s.sector ?? ""), chg_pct: typeof s.chg_pct === "number" ? s.chg_pct : null }))} empty="No sector data." plain />
            </Card>
            <Card title="Top gainers" id="gainers"><MoverList rows={b.gainers} empty="No gainers listed." /></Card>
            <Card title="Top losers" id="losers"><MoverList rows={b.losers} empty="No losers listed." /></Card>
          </div>
        </div>
        <aside className="col-side">
          <Card title="Breakout scores" id="bo">
            {b.top_breakout_scores.length ? (
              <ul className="rows">{b.top_breakout_scores.map((t) => <li key={String(t.ticker)} className="row"><TickerLink ticker={String(t.ticker)} /><span className="grow" /><span className="mono">{String(t.score ?? "—")}</span></li>)}</ul>
            ) : <p className="cap">None in this snapshot.</p>}
          </Card>
          <Card title="PreBreakout picks" id="pb">
            {b.prebreakout_locked ? <Locked title="PreBreakout is part of Premium" plan="premium">The model&apos;s top candidates before they break out.</Locked>
              : b.prebreakout_picks.length ? (
                <ul className="rows">{b.prebreakout_picks.map((p) => <li key={p.ticker} className="row"><TickerLink ticker={p.ticker} /><span className="grow" /><span className="mono">{probPct(p.prob)}</span></li>)}</ul>
              ) : <p className="cap">No PreBreakout picks in this snapshot.</p>}
          </Card>
          <Card title="Golden crosses" id="gc">
            {b.golden_crosses.length ? <div className="chips">{b.golden_crosses.map((t) => <TickerLink key={t} ticker={t} />)}</div> : <p className="cap">None today.</p>}
          </Card>
          {b.earnings_today.length > 0 && (
            <Card title="Reporting today" id="et"><div className="chips">{b.earnings_today.map((t) => <TickerLink key={t} ticker={t} />)}</div></Card>
          )}
          <Earnings />
          {can("can_ai_notes") && <Narrative />}
        </aside>
      </div>
    </>
  );
}

export function BriefView() {
  const brief = useApi("brief", (signal) => unwrap(api.GET("/v1/brief", { signal })));
  const b = brief.data;
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Market Brief</h1>
          <p className="cap">
            {b?.available ? `${PHASE[b.phase ?? ""] ?? "Snapshot"} · snapshot ${etTime(b.snapshot_time)}` : "The day's market snapshot from HSF scans."}
          </p>
        </div>
      </section>
      {brief.error && !b ? <ErrorState error={brief.error} onRetry={brief.reload} what="the Market Brief" />
        : !b ? <Skeleton rows={10} label="Loading the Market Brief" />
        : !b.available ? <Card><Empty title="Today's brief isn't ready yet.">It appears after the day&apos;s first scan snapshot.</Empty></Card>
        : <BriefBody b={b} />}
      <p className="cap">Snapshot data from HSF scans, not live quotes.</p>
      <Disclaimer />
    </div>
  );
}
