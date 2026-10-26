import type { Schemas } from "@/api/client";

/** Canvas requires ascending, unique sessions and finite OHLC values. */
export function chartBars(bars: Schemas["Bar"][]): Schemas["Bar"][] {
  const days = new Map<string, Schemas["Bar"]>();
  for (const bar of bars) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(bar.date) || !Number.isFinite(bar.close)) continue;
    const date = new Date(`${bar.date}T12:00:00Z`);
    if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== bar.date) continue;
    const open = bar.open != null && Number.isFinite(bar.open) ? bar.open : bar.close;
    days.set(bar.date, { ...bar, open,
      high: Math.max(open, bar.close, bar.high != null && Number.isFinite(bar.high) ? bar.high : bar.close),
      low: Math.min(open, bar.close, bar.low != null && Number.isFinite(bar.low) ? bar.low : bar.close),
      volume: bar.volume != null && Number.isFinite(bar.volume) ? Math.max(0, bar.volume) : null });
  }
  return [...days.values()].sort((a, b) => a.date.localeCompare(b.date));
}

type Bar = Schemas["Bar"];
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

/** Wilder's RSI over closing prices: null until `period` changes have been seen.
 * Computed from the full history so the first visible values are already settled. */
export function rsi(closes: number[], period = 14): (number | null)[] {
  const out: (number | null)[] = closes.map(() => null);
  if (closes.length <= period) return out;
  let gain = 0, loss = 0;
  for (let i = 1; i <= period; i++) {
    const ch = closes[i]! - closes[i - 1]!;
    if (ch >= 0) gain += ch; else loss -= ch;
  }
  gain /= period; loss /= period;
  const value = () => (loss === 0 ? (gain === 0 ? 50 : 100) : 100 - 100 / (1 + gain / loss));
  out[period] = value();
  for (let i = period + 1; i < closes.length; i++) {
    const ch = closes[i]! - closes[i - 1]!;
    gain = (gain * (period - 1) + Math.max(0, ch)) / period;
    loss = (loss * (period - 1) + Math.max(0, -ch)) / period;
    out[i] = value();
  }
  return out;
}
