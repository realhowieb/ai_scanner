"use client";

import Link from "@/components/Link";

import type { Schemas } from "@/api/client";
import { Card, Empty, Freshness, Pill, ScoreBadge, Stat, StatGrid, TickerLink } from "@/components/ui";
import { etTime, pct } from "@/lib/format";

type Brief = Schemas["Brief"];
type Opportunity = { ticker?: string; score?: number | null; score_delta?: number | null; movement_state?: string; primary_setup?: string; rank?: number; previous_rank?: number; status?: string };
const tone = (n: number) => n > 0 ? "up" : n < 0 ? "down" : "";
const finite = (n: unknown): n is number => typeof n === "number" && Number.isFinite(n);

export function RankMovement({ rank, previous }: { rank?: number; previous?: number }) {
  if (!finite(rank) || !finite(previous)) return null;
  const change = previous - rank;
  return <span className={tone(change)} aria-label={`Rank ${change > 0 ? "improved" : change < 0 ? "fell" : "unchanged"} by ${Math.abs(change)}`}>{change > 0 ? "↑" : change < 0 ? "↓" : "="}{Math.abs(change)}</span>;
}

/** The 60-second read: indexes, breadth, sector leaders and the top-ranked name, from this snapshot only. */
export function BriefGlance({ b }: { b: Brief }) {
  const sectors = b.sectors.filter((s) => finite(s.chg_pct));
  const strongest = sectors.reduce<(typeof sectors)[number] | undefined>((a, s) => !a || Number(s.chg_pct) > Number(a.chg_pct) ? s : a, undefined);
  const weakest = sectors.reduce<(typeof sectors)[number] | undefined>((a, s) => !a || Number(s.chg_pct) < Number(a.chg_pct) ? s : a, undefined);
  const adv = b.breadth?.advancers;
  const dec = b.breadth?.decliners;
  const breadth = finite(adv) && finite(dec) && adv >= 0 && dec >= 0 && adv + dec > 0 ? (100 * adv) / (adv + dec) : null;
  const lead = (b.opportunities as Opportunity[]).find((o) => o.ticker && finite(o.score));
  const indexes = b.market.filter((m) => finite(m.chg_pct)).slice(0, 3);
  if (!indexes.length && breadth == null && !strongest && !lead) return null;
  return (
    <section className="brief-glance" aria-labelledby="glance-h">
      <h2 className="h3" id="glance-h">At a glance</h2>
      <StatGrid label="Brief at a glance">
        {indexes.map((m) => <Stat key={m.label} label={m.label} value={pct(m.chg_pct)} tone={Number(m.chg_pct) > 0 ? "up" : Number(m.chg_pct) < 0 ? "down" : undefined}
          note={finite(m.last) ? m.last.toLocaleString("en-US", { maximumFractionDigits: 2 }) : undefined} />)}
        {breadth != null && <Stat label="Advancing" value={`${breadth.toFixed(0)}%`} note="HSF snapshot symbols" />}
        {strongest && <Stat label="Strongest sector" value={pct(Number(strongest.chg_pct))} tone={Number(strongest.chg_pct) > 0 ? "up" : Number(strongest.chg_pct) < 0 ? "down" : undefined} note={String(strongest.sector ?? "")} />}
        {weakest && weakest !== strongest && <Stat label="Weakest sector" value={pct(Number(weakest.chg_pct))} tone={Number(weakest.chg_pct) > 0 ? "up" : Number(weakest.chg_pct) < 0 ? "down" : undefined} note={String(weakest.sector ?? "")} />}
        {lead && <Stat label="Top HSF Score" value={<Link href={`/stocks/${encodeURIComponent(String(lead.ticker))}`}>{lead.ticker}</Link>} note={`HSF ${lead.score}${lead.primary_setup ? ` · ${lead.primary_setup}` : ""}`} />}
      </StatGrid>
      <p className="cap">Index and sector values are the latest supplied by the Brief; separate quote timestamps are unavailable. Breadth counts HSF snapshot symbols, not the whole market.</p>
    </section>
  );
}

export function MarketPulse({ b }: { b: Brief }) {
  const adv = b.breadth?.advancers;
  const dec = b.breadth?.decliners;
  const scores = b.opportunities.map((o) => o.score).filter(finite);
  const topScore = scores.length ? Math.max(...scores) : null;
  const validBreadth = finite(adv) && finite(dec) && adv >= 0 && dec >= 0 && adv + dec > 0;
  return <Card title="Market Pulse" id="pulse" className="brief-pulse" source="data">
    <Freshness at={b.snapshot_time} label="Snapshot" stale={b.freshness?.state === "stale" ? true : b.freshness?.state === "fresh" ? false : undefined} />
    {validBreadth && <section aria-label="HSF snapshot breadth">
      <h3 className="h3">HSF snapshot breadth</h3>
      <div className="brief-breadth"><span>Advancing {adv}</span><span>Declining {dec}</span></div>
      <div className="brief-breadth-bar" aria-hidden="true"><span style={{ width: `${100 * adv / (adv + dec)}%` }} /></div>
      <p className="cap">{(100 * adv / (adv + dec)).toFixed(0)}% advancing among moving snapshot symbols; unchanged excluded. Not whole-market breadth.</p>
    </section>}
    <h3 className="h3">HSF Market Intelligence</h3>
    <dl className="brief-counts">
      <div><dt>Top HSF Score</dt><dd>{topScore ?? "—"}</dd></div>
      {!b.prebreakout_locked && <div><dt>PreBreakout picks shown</dt><dd>{b.prebreakout_picks.length}</dd></div>}
      <div><dt>Golden crosses shown</dt><dd>{b.golden_crosses.length}</dd></div>
    </dl>
  </Card>;
}

export function OpportunityRadar({ b }: { b: Brief }) {
  const opportunities = (b.opportunities as Opportunity[]).filter((o) => o.ticker);
  return <Card title="HSF Opportunity Radar" id="opps" source="ranking">
    <p className="cap">Top names by HSF Score, the same scores as the Scanner. Snapshot {etTime(b.snapshot_time)}.</p>
    {!opportunities.length ? <Empty title="No ranked opportunities in this snapshot." /> : <ol className="brief-radar">
      {opportunities.map((o, i) => <li key={o.ticker}>
        <div className="brief-radar-identity"><TickerLink ticker={o.ticker!} /><span className="cap">#{o.rank ?? i + 1} <RankMovement rank={o.rank} previous={o.previous_rank} /></span></div>
        <div><span className="cap">HSF Score </span><ScoreBadge score={o.score} />
          {finite(o.score_delta) && <span className={`cap ${tone(o.score_delta)}`}> {o.score_delta > 0 ? "Rising +" : o.score_delta < 0 ? "Falling " : "Unchanged "}{o.score_delta}</span>}
          {!finite(o.score_delta) && o.movement_state === "NEW" && <span className="cap"> New</span>}
        </div>
        <div className="brief-radar-setup"><span>{o.primary_setup ?? "Setup unavailable"}</span>{o.status && <Pill>{o.status}</Pill>}</div>
      </li>)}
    </ol>}
    <Link className="btn" href="/scanner">View Scanner →</Link>
  </Card>;
}
