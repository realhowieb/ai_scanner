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
