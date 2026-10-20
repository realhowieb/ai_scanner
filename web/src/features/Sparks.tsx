"use client";

import { api, unwrap } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { num, price } from "@/lib/format";

/** Daily closes per ticker (up to 21, oldest first) from /v1/sparklines; one request per ticker set. */
/** The daily-bars request with its load state, so a page can refresh it on demand. */
export function useDailyBarsState(tickers: string[]) {
  const syms = [...new Set(tickers.map((t) => t.trim().toUpperCase()).filter(Boolean))].sort().slice(0, 50).join(",");
  return useApi(syms ? `spark-daily:${syms}` : null, (signal) =>
    unwrap(api.GET("/v1/sparklines", { params: { query: { symbols: syms } }, signal })));
}

export function useDailyBars(tickers: string[]) {
  return useDailyBarsState(tickers).data;
}

export function useDailySparks(tickers: string[]): Record<string, number[]> {
  return useDailyBars(tickers)?.series ?? {};
}

/** A 20-session close sparkline with an end dot: green when the last close is above the first. */
export function DailySpark({ points, ticker, width = 96, height = 26 }: { points: number[] | undefined; ticker: string; width?: number; height?: number }) {
  if (!points || points.length < 2) return <span className="cap" aria-hidden="true">—</span>;
  const lo = Math.min(...points), hi = Math.max(...points);
  const span = hi - lo || 1;
  const xy = points.map((p, i) => [(i / (points.length - 1)) * (width - 4) + 1, height - 3 - ((p - lo) / span) * (height - 6)] as const);
  const d = xy.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  const first = points[0]!, last = points[points.length - 1]!;
  const up = last >= first;
  const [ex, ey] = xy[xy.length - 1]!;
  const change = first ? ((last - first) / first) * 100 : 0;
  return (
    <svg className={`spark ${up ? "up" : "down"}`} width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
      aria-label={`${ticker} last ${points.length - 1} sessions: ${up ? "up" : "down"} ${Math.abs(change).toFixed(1)}%, ${price(first)} to ${price(last)}`}>
      <title>{`${points.length - 1} sessions: ${change >= 0 ? "+" : ""}${change.toFixed(1)}%`}</title>
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={ex} cy={ey} r="2.2" fill="currentColor" />
    </svg>
  );
}

/** Relative volume as a short meter (full at 3x), amber from 1.5x, with the number beside it. */
export function RvolMeter({ rvol }: { rvol: number | null | undefined }) {
  if (rvol === null || rvol === undefined || !Number.isFinite(rvol)) return <span className="mono">—</span>;
  const w = Math.max(3, Math.min(100, (rvol / 3) * 100));
  return (
    <span className="rvol" title="Relative volume: today's volume vs its recent average">
      <span className="rvol-track" aria-hidden="true"><span className={`rvol-fill${rvol >= 1.5 ? " hot" : ""}`} style={{ width: `${w}%` }} /></span>
      <span className="mono">{num(rvol)}</span>
    </span>
  );
}
