"use client";

import type { Schemas } from "@/api/client";
import { watchlistQuotes } from "@/features/WatchlistsView";
import { useApi } from "@/hooks/useApi";
import { inMarketHours, useAutoRefresh } from "@/hooks/useAutoRefresh";

/** How often the Scanner re-reads live prices. The API shares each ticker set for 60 s. */
export const LIVE_QUOTES_MS = 60_000;

type Quote = Schemas["LatestQuote"];
type Priced = { ticker: string; last?: number | null; chg_pct?: number | null; gap_pct?: number | null; scan_last?: number | null };

/** Latest trades for the rows on screen, re-read every minute while the market is open and
 * the tab is visible. Outside market hours (or with nothing on screen) it asks for nothing,
 * so the rows keep the scan's prices. */
export function useLiveQuotes(tickers: string[], enabled: boolean) {
  const on = enabled && tickers.length > 0 && inMarketHours();
  // Sorted so the same rows in any order share one request (and the API's cache).
  const sorted = on ? [...new Set(tickers.map((t) => t.toUpperCase()))].sort() : [];
  const key = on ? `live-quotes:${sorted.join(",")}` : null;
  const state = useApi(key, (signal) => watchlistQuotes(sorted, signal));
  useAutoRefresh(state.reload, { everyMs: LIVE_QUOTES_MS, enabled: key !== null });
  return { quotes: key ? (state.data?.quotes ?? {}) : {}, updatedAt: key && state.data ? state.updatedAt : null };
}

/** The rows with Last, Chg % and Gap % taken from a live trade where there is one. A row with
 * no trade (or only a daily-bar fallback) keeps the scan's numbers; a quote without a gap
 * keeps the scan's gap. A live row keeps the scan's price as `scan_last` (for "since scan"). */
export function withLiveQuotes<T extends Priced>(rows: T[], quotes: Record<string, Quote | undefined>): T[] {
  if (!Object.keys(quotes).length) return rows;
  return rows.map((r) => {
    const q = quotes[r.ticker.toUpperCase()];
    if (!q || q.source !== "trade" || q.last == null) return r;
    return { ...r, scan_last: r.last, last: q.last, chg_pct: q.chg_pct ?? r.chg_pct, gap_pct: q.gap_pct ?? r.gap_pct };
  });
}
