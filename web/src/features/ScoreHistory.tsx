"use client";

import { api, unwrap } from "@/api/client";
import { Card, RANKED_MIN } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { etTime } from "@/lib/format";

const W = 320, H = 150, L = 26, R = 8, T = 10, B = 22;
const STRONG = 75;
const SIGNAL_NAMES: Record<string, string> = {
  breakout: "Breakout", gapper: "Gapper", golden_cross: "Golden cross", gainer: "Gainer", prebreakout: "PreBreakout",
};

const shortDay = (iso: string) => new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" });

/** The ticker's HSF Score in each recent full-market scan (the Scanner's scores), with the
 * 75 "strong" line, the 40 ranked floor, gaps where it wasn't ranked, and a diamond where a
 * signal was added. Replaces the older lifecycle list, whose scores came from the Brief. */
export function ScoreHistory({ ticker }: { ticker: string }) {
  const res = useApi(`score-history:${ticker}`, (signal) =>
    unwrap(api.GET("/v1/stocks/{ticker}/score-history", { params: { path: { ticker } }, signal })));
  const points = res.data?.points ?? [];
  const scored = points.filter((p) => p.score !== null && p.score !== undefined);
  if (!res.data || scored.length === 0) return null;

  const n = points.length;
  const x = (i: number) => L + (n <= 1 ? (W - L - R) / 2 : (i / (n - 1)) * (W - L - R));
  const y = (s: number) => T + (1 - s / 100) * (H - T - B);
  // One path segment per run of consecutive ranked scans; unranked scans leave a gap.
  const segments: string[] = [];
  let seg = "";
  points.forEach((p, i) => {
    if (p.score === null || p.score === undefined) { if (seg) segments.push(seg); seg = ""; return; }
    seg += `${seg ? "L" : "M"}${x(i).toFixed(1)},${y(p.score).toFixed(1)}`;
  });
  if (seg) segments.push(seg);
  let lastIdx = 0;
  points.forEach((p, i) => { if (p.score !== null && p.score !== undefined) lastIdx = i; });
  const last = points[lastIdx]!;
  const firstScore = scored[0]!.score!;
  const change = last.score! - firstScore;
  const ticks = n > 1 ? [0, Math.floor((n - 1) / 2), n - 1] : [0];

  return (
    <Card title="HSF Score history" id="life">
      <figure className="score-hist">
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
          aria-label={`${ticker} HSF Score over the last ${n} full-market scans: ${firstScore} to ${last.score}, ranked in ${scored.length} of ${n}.`}>
          {[0, RANKED_MIN, STRONG, 100].map((t) => (
            <g key={t}>
              <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className={t === STRONG ? "sh-strong" : t === RANKED_MIN ? "sh-floor" : "sh-grid"} />
              <text x={L - 5} y={y(t) + 3.5} textAnchor="end" className="sh-axis">{t}</text>
            </g>
          ))}
          {segments.map((d) => <path key={d} d={d} className="sh-line" />)}
          {points.map((p, i) => p.score === null || p.score === undefined ? null : (
            <g key={`${p.time}-${i}`}>
              {p.signals_added.length > 0 && (
                <path d={`M${x(i)},${y(p.score) - 11} l4.5,4.5 l-4.5,4.5 l-4.5,-4.5z`} className="sh-signal" />
              )}
              <circle cx={x(i)} cy={y(p.score)} r={i === lastIdx ? 3.5 : 2} className={i === lastIdx ? "sh-last" : "sh-dot"}>
                <title>{`${p.time ? etTime(p.time) : ""} · HSF ${p.score}${p.signals_added.length ? ` · added ${p.signals_added.map((s) => SIGNAL_NAMES[s] ?? s).join(", ")}` : ""}`}</title>
              </circle>
            </g>
          ))}
          {ticks.map((i) => points[i]?.time ? (
            <text key={i} x={x(i)} y={H - 6} textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"} className="sh-axis">{shortDay(points[i]!.time!)}</text>
          ) : null)}
        </svg>
        <figcaption className="cap">
          HSF {last.score} in the latest of {n} full-market scans{scored.length > 1 ? `, ${change > 0 ? "up" : change < 0 ? "down" : "unchanged"}${change ? ` ${Math.abs(change)}` : ""} from the first` : ""}. Ranked in {scored.length} of {n}; gaps are scans where it wasn&apos;t ranked. Dashed lines: 75 strong, 40 ranked. ◆ a signal was added.
        </figcaption>
      </figure>
    </Card>
  );
}
