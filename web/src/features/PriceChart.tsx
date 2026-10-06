"use client";

import type { Schemas } from "@/api/client";
import { etDate, price } from "@/lib/format";

type Bar = Schemas["Bar"];

const W = 720;
const H = 300;
const VOL_H = 56;
const PAD = { l: 8, r: 56, t: 12, b: 8 };

/** Daily candles plus volume, from the bars the scans cached. Not a live chart. */
export function PriceChart({ bars, asOf }: { bars: Bar[]; asOf?: string | null }) {
  const data = bars.filter((b) => Number.isFinite(b.close));
  if (data.length < 2) return null;
  const highs = data.map((b) => b.high ?? b.close);
  const lows = data.map((b) => b.low ?? b.close);
  const max = Math.max(...highs);
  const min = Math.min(...lows);
  const span = max - min || 1;
  const vmax = Math.max(1, ...data.map((b) => b.volume ?? 0));
  const plotH = H - VOL_H - PAD.t - PAD.b;
  const step = (W - PAD.l - PAD.r) / data.length;
  const y = (v: number) => PAD.t + (1 - (v - min) / span) * plotH;
  const bw = Math.max(1, step * 0.6);
  const last = data[data.length - 1]!;
  const first = data[0]!;
  const ticks = [max, min + span / 2, min];

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
      </svg>
      <figcaption className="cap">
        {data.length} daily bars, {etDate(first.date)} to {etDate(last.date)}{asOf ? ` · cached by HSF scans` : ""}. Not live prices.
      </figcaption>
    </figure>
  );
}
