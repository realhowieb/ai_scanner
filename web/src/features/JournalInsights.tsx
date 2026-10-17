"use client";

// Journal insights (Pro): closed-trade analytics computed by the API from your journal:
// win/loss profile, a running P&L curve, and results by HSF Score at entry, holding time
// and source. Hidden on an API without the route; locked below Pro.
import { insights } from "@/api/userData";
import type { JournalInsights as Data } from "@/api/userData";
import { Card, Locked } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime, num, pct } from "@/lib/format";

const W = 320, H = 150, L = 44, R = 8, T = 10, B = 22;
const money = (v: number) => `${v < 0 ? "−" : ""}$${Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
const tone = (v: number | null | undefined) => ((v ?? 0) > 0 ? "up" : (v ?? 0) < 0 ? "down" : "");

function Curve({ points }: { points: Data["curve"] }) {
  if (points.length < 2) return null;
  const vals = [0, ...points.map((p) => p.cumulative_pnl)];
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;
  const n = points.length;
  const x = (i: number) => L + (i / (n - 1)) * (W - L - R);
  const y = (v: number) => T + (1 - (v - lo) / span) * (H - T - B);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.cumulative_pnl).toFixed(1)}`).join("");
  const last = points[n - 1]!;
  return (
    <figure className="score-hist">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
        aria-label={`Running P&L over ${n} closed trades, ending at ${money(last.cumulative_pnl)}.`}>
        {[hi, 0, lo].filter((v, i, a) => a.indexOf(v) === i).map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className={t === 0 ? "sh-floor" : "sh-grid"} />
            <text x={L - 5} y={y(t) + 3.5} textAnchor="end" className="sh-axis">{money(t)}</text>
          </g>
        ))}
        <path d={d} className="sh-line" />
        {points.map((p, i) => (
          <circle key={i} cx={x(i)} cy={y(p.cumulative_pnl)} r={i === n - 1 ? 3.5 : 4} className={i === n - 1 ? "sh-last" : "sh-hit"}>
            <title>{`${p.closed_at ? etTime(p.closed_at) : ""} · ${p.ticker ?? ""} ${money(p.pnl)} · total ${money(p.cumulative_pnl)}`}</title>
          </circle>
        ))}
      </svg>
      <figcaption className="cap">Running P&amp;L across your closed trades, oldest first. Hover a point for the trade.</figcaption>
    </figure>
  );
}

function Groups({ title, rows }: { title: string; rows: Data["by_score"] }) {
  if (!rows.length) return null;
  return (
    <div className="table-wrap">
      <table className="table">
        <caption className="cap">{title}</caption>
        <thead><tr><th scope="col">Group</th><th scope="col" className="num">Trades</th><th scope="col" className="num">Win rate</th>
          <th scope="col" className="num">Avg return</th><th scope="col" className="num hide-narrow">P&amp;L</th></tr></thead>
        <tbody>
          {rows.map((g) => (
            <tr key={g.label}>
              <th scope="row">{g.label}</th>
              <td className="num mono">{g.count}</td>
              <td className="num mono">{num(g.win_rate, 0)}%</td>
              <td className={`num mono ${tone(g.avg_return_pct)}`}>{pct(g.avg_return_pct)}</td>
              <td className={`num mono hide-narrow ${tone(g.pnl)}`}>{money(g.pnl)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function JournalInsightsCard({ pro, version = 0 }: { pro: boolean; version?: number }) {
  const res = useApi(pro ? "journal-insights" : null, (signal) => insights.journal(signal), version);
  if (!pro) {
    return (
      <Card title="Insights" id="insights">
        <Locked title="Journal insights are part of Pro" plan="pro">Win rate by HSF Score at entry and by holding time, average win and loss, and a running P&amp;L chart.</Locked>
      </Card>
    );
  }
  const d = res.data;
  if (res.error || !d || typeof d.closed !== "number") return null;
  if (!d.summary) {
    return <Card title="Insights" id="insights"><p className="cap">Close a trade to see your win rate, average win and loss, and what works best for you.</p></Card>;
  }
  const s = d.summary;
  return (
    <Card title="Insights" id="insights" aside={`${d.closed} closed trade${d.closed === 1 ? "" : "s"}`}>
      <div className="metric-strip" aria-label="Journal insights summary">
        <span><strong>{num(s.win_rate, 0)}%</strong> win rate</span>
        <span><strong className={`mono ${tone(s.total_pnl)}`}>{money(s.total_pnl)}</strong> total P&amp;L</span>
        <span><strong className="mono up">{pct(s.avg_win_pct)}</strong> avg win</span>
        <span><strong className="mono down">{pct(s.avg_loss_pct)}</strong> avg loss</span>
        <span><strong>{s.profit_factor == null ? "—" : num(s.profit_factor, 2)}</strong> profit factor</span>
        {s.avg_hold_days != null && <span><strong>{num(s.avg_hold_days, 1)}</strong> days avg hold</span>}
      </div>
      <Curve points={d.curve} />
      <Groups title="By HSF Score when you logged the trade" rows={d.by_score} />
      <Groups title="By how long you held" rows={d.by_hold} />
      <Groups title="By where the trade came from" rows={d.by_source} />
      <p className="cap">Best {s.best.ticker} {pct(s.best.return_pct)}, worst {s.worst.ticker} {pct(s.worst.return_pct)}. Profit factor is dollars won divided by dollars lost. Trades logged before this update have no recorded score. Your own records, not HSF results.</p>
    </Card>
  );
}
