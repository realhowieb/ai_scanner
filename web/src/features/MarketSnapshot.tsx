import Link from "next/link";

import { DataFreshness } from "@/features/DataFreshness";

import type { Schemas } from "@/api/client";
import { ago, compact, etTime, pct } from "@/lib/format";

type Today = Schemas["Today"];
type Snapshot = Schemas["Snapshot"];

/** Shape + words carry the status, not colour alone (as in the Streamlit trust banner). */
const STATUS_SHAPES: Record<string, string> = { ok: "●", limited: "▲", issue: "■", unknown: "○" };

function moveClass(v: number | null | undefined): string {
  return v && v > 0 ? "up" : v && v < 0 ? "down" : "";
}

/** One line on what the page is built from: latest scan, universe, ranked count and system status. */
export function StatusStrip({ market, snapshot }: { market: Today["market"]; snapshot: Snapshot | null | undefined }) {
  const scanAt = market.latest_scan_at;
  const age = ago(scanAt);
  const status = snapshot?.status;
  return (
    <section className="status-strip" aria-label="Market data status">
      <DataFreshness info={market.freshness} />
      <span className="status-scope">Full U.S. market</span>
      <span>{scanAt ? <>Last scan {etTime(scanAt)}{age ? ` (${age})` : ""}</> : "Latest market scan unavailable"}</span>
      {snapshot?.universe_symbols ? <span>Universe {snapshot.universe_symbols.toLocaleString("en-US")} tradable stocks</span> : null}
      {snapshot?.ranked_count ? <span>{snapshot.ranked_count.toLocaleString("en-US")} ranked setups</span> : null}
      {status && (
        <span className={`status-level lvl-${status.level}`}>
          <span aria-hidden="true">{STATUS_SHAPES[status.level] ?? "○"}</span> System: {status.label}
        </span>
      )}
    </section>
  );
}

function Tile({ label, value, delta, tone, href }: { label: string; value: string; delta: string; tone: string; href?: string }) {
  return (
    <div className="tile">
      <p className="cap">{label}</p>
      <p className="tile-value mono">{href ? <Link href={href}>{value}</Link> : value}</p>
      <span className={`tile-delta mono ${tone}`}>{delta}</span>
    </div>
  );
}

/** SPY, QQQ, and the latest scan's top gainer and most active name. */
export function SnapshotTiles({ snapshot }: { snapshot: Snapshot }) {
  const idx = (sym: string) => snapshot.indices.find((q) => q.symbol === sym);
  const spy = idx("SPY");
  const qqq = idx("QQQ");
  const g = snapshot.top_gainer;
  const a = snapshot.most_active;
  const stock = (t: string) => `/stocks/${encodeURIComponent(t)}`;
  return (
    <div className="tiles">
      <Tile label="S&P 500 (SPY)" value={spy ? spy.last.toFixed(2) : "—"} delta={pct(spy?.chg_pct)} tone={moveClass(spy?.chg_pct)} />
      <Tile label="Nasdaq 100 (QQQ)" value={qqq ? qqq.last.toFixed(2) : "—"} delta={pct(qqq?.chg_pct)} tone={moveClass(qqq?.chg_pct)} />
      <Tile label="Top gainer" value={g?.ticker ?? "—"} href={g ? stock(g.ticker) : undefined} delta={pct(g?.chg_pct)} tone={moveClass(g?.chg_pct)} />
      <Tile label="Most active" value={a?.ticker ?? "—"} href={a ? stock(a.ticker) : undefined}
        delta={a?.volume ? `${compact(a.volume)} shares` : "—"} tone="" />
    </div>
  );
}
