"use client";

// Today: the setups whose HSF Score moved most since the previous full-market scan.
// Computed by the API (GET /v1/scans/latest/movers) from the two latest saved scans,
// inside the plan's Scanner row cap. Hidden when the API has no such route or there is
// only one scan to compare.
import { insights } from "@/api/userData";
import type { ScoreMovers as Movers } from "@/api/userData";
import { Card, Freshness, TickerLink } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime, setupLabel } from "@/lib/format";

type Row = Movers["rising"][number];

function MoverList({ rows, dir }: { rows: Row[]; dir: "up" | "down" }) {
  if (!rows.length) return <p className="cap">{dir === "up" ? "No score rose since the previous scan." : "No score fell since the previous scan."}</p>;
  return (
    <ul className="mover-list">
      {rows.map((r) => (
        <li key={r.ticker} className="mover-row">
          <TickerLink ticker={r.ticker} />
          <span className="cap grow">{r.setup ? setupLabel(r.setup) : ""}</span>
          <span className="mono">{r.previous_score} → {r.score}</span>
          <span className={`delta-chip mono ${dir}`} aria-label={`${dir === "up" ? "up" : "down"} ${Math.abs(r.change)} points`}>
            {dir === "up" ? "▲ +" : "▼ −"}{Math.abs(r.change)}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function ScoreMoversCard() {
  const movers = useApi("score-movers", (signal) => insights.movers(signal));
  const d = movers.data;
  if (movers.error || !d || !Array.isArray(d.rising) || !Array.isArray(d.falling) || !d.previous_scan_at || (!d.rising.length && !d.falling.length)) return null;
  return (
    <Card title="Score movers" id="movers" aside={d.scan_at ? <Freshness at={d.scan_at} label="Full-market scan" /> : undefined}>
      <div className="grid2">
        <div className="stack-sm"><h3 className="h3">Rising</h3><MoverList rows={d.rising} dir="up" /></div>
        <div className="stack-sm"><h3 className="h3">Falling</h3><MoverList rows={d.falling} dir="down" /></div>
      </div>
      <p className="cap">HSF Score change since the {etTime(d.previous_scan_at)} scan, for setups ranked in both scans. A rising score can mean a setup is forming; it isn&apos;t a forecast.</p>
    </Card>
  );
}
