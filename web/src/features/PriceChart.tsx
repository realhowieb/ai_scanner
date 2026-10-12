"use client";

import type { Schemas } from "@/api/client";
import { etDate, price, shortDate } from "@/lib/format";

type Bar = Schemas["Bar"];

const W = 720;
const H = 300;
const VOL_H = 56;
const PAD = { l: 8, r: 56, t: 12, b: 8 };

type Cross = { i: number; kind: "golden" | "death" };

/** Where EMA 9 crosses EMA 21: the scanner's golden cross (up) and death cross (down). */
export function emaCrosses(bars: Bar[]): Cross[] {
  const out: Cross[] = [];
  for (let i = 1; i < bars.length; i++) {
    const a = bars[i - 1]!, b = bars[i]!;
    if (a.ema9 == null || a.ema21 == null || b.ema9 == null || b.ema21 == null) continue;
    const prev = a.ema9 - a.ema21, curr = b.ema9 - b.ema21;
    if (prev <= 0 && curr > 0) out.push({ i, kind: "golden" });
    else if (prev >= 0 && curr < 0) out.push({ i, kind: "death" });
  }
  return out;
}

/** SVG path through the defined points, broken where a value is missing. */
function linePath(vals: (number | null | undefined)[], x: (i: number) => number, y: (v: number) => number): string {
  let d = "", pen = false;
  vals.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) { pen = false; return; }
    d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}

/** Daily candles plus volume (completed sessions only), with the EMA 9/21 lines and
 * their golden and death crosses. Not a live chart. */
export function PriceChart({ bars }: { bars: Bar[] }) {
  const data = bars.filter((b) => Number.isFinite(b.close));
  if (data.length < 2) return null;
  const highs = data.map((b) => b.high ?? b.close);
  const lows = data.map((b) => b.low ?? b.close);
  const emas = data.flatMap((b) => [b.ema9, b.ema21]).filter((v): v is number => v != null && Number.isFinite(v));
  const max = Math.max(...highs, ...emas);
  const min = Math.min(...lows, ...emas);
  const span = max - min || 1;
  const vmax = Math.max(1, ...data.map((b) => b.volume ?? 0));
  const plotH = H - VOL_H - PAD.t - PAD.b;
  const step = (W - PAD.l - PAD.r) / data.length;
  const y = (v: number) => PAD.t + (1 - (v - min) / span) * plotH;
  const bw = Math.max(1, step * 0.6);
  const last = data[data.length - 1]!;
  const first = data[0]!;
  const ticks = [max, min + span / 2, min];
  const xAt = (i: number) => PAD.l + i * step + step / 2;
  const crosses = emaCrosses(data);
  const hasEma = data.some((b) => b.ema9 != null);
  const lastGolden = [...crosses].reverse().find((c) => c.kind === "golden");
  const tri = 5;

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Daily price from ${etDate(first.date)} to ${etDate(last.date)}, last close ${price(last.close)}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={W - PAD.r + 6} y={y(t) + 4} className="axis">{t.toFixed(2)}</text>
          </g>
        ))}
        {data.map((b, i) => {
          const x = PAD.l + i * step + step / 2;
          const o = b.open ?? b.close;
          const up = b.close >= o;
          const vh = ((b.volume ?? 0) / vmax) * (VOL_H - 8);
          return (
            <g key={b.date} className={up ? "c-up" : "c-down"}>
              <line x1={x} x2={x} y1={y(b.high ?? Math.max(o, b.close))} y2={y(b.low ?? Math.min(o, b.close))} />
              <rect x={x - bw / 2} width={bw} y={y(Math.max(o, b.close))} height={Math.max(1, Math.abs(y(o) - y(b.close)))} />
              <rect className="vol" x={x - bw / 2} width={bw} y={H - PAD.b - vh} height={vh} />
            </g>
          );
        })}
        {hasEma && <path className="ema ema-21" d={linePath(data.map((b) => b.ema21), xAt, y)} />}
        {hasEma && <path className="ema ema-9" d={linePath(data.map((b) => b.ema9), xAt, y)} />}
        {crosses.map(({ i, kind }) => {
          const b = data[i]!;
          const x = xAt(i);
          const golden = kind === "golden";
          const yy = golden ? Math.min(H - VOL_H - tri, y(b.low ?? b.close) + 6) : Math.max(tri, y(b.high ?? b.close) - 6);
          const pts = golden ? `${x},${yy} ${x - tri},${yy + tri * 1.6} ${x + tri},${yy + tri * 1.6}` : `${x},${yy} ${x - tri},${yy - tri * 1.6} ${x + tri},${yy - tri * 1.6}`;
          return (
            <polygon key={`${kind}-${b.date}`} className={golden ? "x-golden" : "x-death"} points={pts}>
              <title>{`${golden ? "Golden" : "Death"} cross (EMA 9 ${golden ? "above" : "below"} EMA 21), ${etDate(b.date)}`}</title>
            </polygon>
          );
        })}
      </svg>
      {hasEma && (
        <div className="chart-legend" aria-hidden="true">
          <span><i className="sw ema-9" />EMA 9</span>
          <span><i className="sw ema-21" />EMA 21</span>
          <span><i className="mk x-golden">▲</i>Golden cross</span>
          <span><i className="mk x-death">▼</i>Death cross</span>
        </div>
      )}
      <figcaption className="cap">
        {data.length} daily bars, {etDate(first.date)} to {etDate(last.date)}.{hasEma ? (lastGolden ? ` Last golden cross ${shortDate(data[lastGolden.i]!.date)} (${data.length - 1 - lastGolden.i === 0 ? "latest session" : `${data.length - 1 - lastGolden.i} sessions ago`}).` : " No golden cross in this range.") : ""} End-of-day prices, not live.
      </figcaption>
    </figure>
  );
}
