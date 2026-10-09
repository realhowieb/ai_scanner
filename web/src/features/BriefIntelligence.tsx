"use client";

import Link from "next/link";

import type { Schemas } from "@/api/client";
import { Card, Empty, Freshness, Pill, ScoreBadge, TickerLink } from "@/components/ui";
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

export function MarketPulse({ b }: { b: Brief }) {
  const sectors = b.sectors.filter((s) => finite(s.chg_pct));
  const strongest = sectors.reduce<(typeof sectors)[number] | undefined>((a, s) => !a || Number(s.chg_pct) > Number(a.chg_pct) ? s : a, undefined);
  const weakest = sectors.reduce<(typeof sectors)[number] | undefined>((a, s) => !a || Number(s.chg_pct) < Number(a.chg_pct) ? s : a, undefined);
  const adv = b.breadth?.advancers;
  const dec = b.breadth?.decliners;
  const validBreadth = finite(adv) && finite(dec) && adv >= 0 && dec >= 0 && adv + dec > 0;
  return <Card title="Market Pulse" id="pulse" className="brief-pulse">
    <Freshness at={b.snapshot_time} label="Snapshot" />
    <div className="brief-indexes">
      {b.market.map((m) => <div className="index-tile" key={m.label}>
        <span className="cap">{m.label}</span>
        <span className="mono strong">{finite(m.last) ? m.last.toLocaleString("en-US", { maximumFractionDigits: 2 }) : "—"}</span>
        <span className={`mono ${finite(m.chg_pct) ? tone(m.chg_pct) : ""}`}>{pct(m.chg_pct)}</span>
      </div>)}
    </div>
    <p className="cap">Index and sector values are the latest supplied by the Brief; separate quote timestamps are unavailable.</p>
    {!b.market.length && <p className="cap">Index data unavailable in this snapshot.</p>}
    {validBreadth && <section aria-label="HSF snapshot breadth">
      <h3 className="h3">HSF snapshot breadth</h3>
      <div className="brief-breadth"><span>Advancing {adv}</span><span>Declining {dec}</span></div>
      <div className="brief-breadth-bar" aria-hidden="true"><span style={{ width: `${100 * adv / (adv + dec)}%` }} /></div>
      <p className="cap">{(100 * adv / (adv + dec)).toFixed(0)}% advancing among moving snapshot symbols; unchanged excluded. Not whole-market breadth.</p>
    </section>}
    <h3 className="h3">HSF Market Intelligence</h3>
    <dl className="brief-counts">
      <div><dt>HSF 80+ in Radar</dt><dd>{b.opportunities.filter((o) => finite(o.score) && o.score >= 80).length}</dd></div>
      {!b.prebreakout_locked && <div><dt>PreBreakout picks shown</dt><dd>{b.prebreakout_picks.length}</dd></div>}
      <div><dt>Golden crosses shown</dt><dd>{b.golden_crosses.length}</dd></div>
    </dl>
    {strongest && weakest && <div className="brief-sector-context cap"><span>Strongest sector: <strong>{String(strongest.sector)}</strong> {pct(Number(strongest.chg_pct))}</span><span>Weakest sector: <strong>{String(weakest.sector)}</strong> {pct(Number(weakest.chg_pct))}</span></div>}
  </Card>;
}

export function OpportunityRadar({ b }: { b: Brief }) {
  const opportunities = (b.opportunities as Opportunity[]).filter((o) => o.ticker);
  return <Card title="HSF Opportunity Radar" id="opps">
    <p className="cap">Top opportunities in canonical Brief order. Snapshot {etTime(b.snapshot_time)}.</p>
    {!opportunities.length ? <Empty title="No ranked opportunities in this snapshot." /> : <ol className="brief-radar">
      {opportunities.map((o, i) => <li key={o.ticker}>
        <div className="brief-radar-identity"><TickerLink ticker={o.ticker!} /><span className="cap">Brief #{o.rank ?? i + 1} <RankMovement rank={o.rank} previous={o.previous_rank} /></span></div>
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
