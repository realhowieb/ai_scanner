"use client";

// The Scanner's Watchlist tab: one of the user's watchlists in the Scanner table. Tickers
// ranked in the latest scan carry that scan's row (GET /v1/watchlists/{id} items[].latest,
// not capped by plan: they are the user's own stocks); the rest show their latest trade
// (GET /v1/quotes) and "Not ranked". Sorting happens here, since the list is small.
import Link from "@/components/Link";
import { useMemo } from "react";

import type { Schemas } from "@/api/client";
import { watchlists } from "@/api/userData";
import { Card, Empty, ErrorState, Skeleton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import type { useWorkspace } from "@/hooks/useWorkspace";

import { SetupTable } from "./SetupTable";
import type { ServerSort } from "./SetupTable";
import { StarterWatchlistButton } from "./Starters";
import { watchlistQuotes } from "./WatchlistsView";

type Row = Schemas["ScanSetup"];
const SORT_KEY: Record<ServerSort, keyof Row> = { score: "score", chg_pct: "chg_pct", gap_pct: "gap_pct", rvol: "rvol", prob: "prob" };

/** The watchlist the tab shows: the one in the link, else the default, else the first. */
export function useScannerWatchlists(asked: number, enabled: boolean) {
  const lists = useApi(enabled ? "wl-list" : null, (signal) => watchlists.list(signal));
  const all = lists.data ?? [];
  const chosen = all.find((w) => w.id === asked) ?? all.find((w) => w.is_default) ?? all[0] ?? null;
  return { lists, all, chosen };
}

/** Ranked first (by the chosen sort), then unranked ones, each group high to low with blanks last. */
export function sortWatchlistRows(rows: Row[], ranks: Record<string, number | null>, sort: ServerSort): Row[] {
  const key = SORT_KEY[sort];
  const val = (r: Row) => {
    if (sort === "score" && ranks[r.ticker] == null) return null;
    const v = r[key];
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  };
  return [...rows].sort((a, b) => {
    const ua = ranks[a.ticker] == null ? 1 : 0, ub = ranks[b.ticker] == null ? 1 : 0;
    if (sort === "score" && ua !== ub) return ua - ub;
    const va = val(a), vb = val(b);
    if (va == null || vb == null) return va == null && vb == null ? a.ticker.localeCompare(b.ticker) : va == null ? 1 : -1;
    return vb - va || (ranks[a.ticker] ?? 1e9) - (ranks[b.ticker] ?? 1e9);
  });
}

export function WatchlistControls({ all, chosenId, sort, sorts, onPick, onSort }: {
  all: { id: number; name: string; symbol_count: number }[]; chosenId: number | null; sort: ServerSort;
  sorts: { id: ServerSort; label: string }[]; onPick: (id: number) => void; onSort: (sort: ServerSort) => void;
}) {
  return (
    <section aria-label="Watchlist options" className="filters scanner-filters">
      {all.length > 1 && <label className="inline-field">Watchlist
        <select value={chosenId ?? ""} onChange={(e) => onPick(Number(e.target.value))}>
          {all.map((w) => <option key={w.id} value={w.id}>{w.name} ({w.symbol_count})</option>)}
        </select>
      </label>}
      <label className="inline-field">Sort by
        <select value={sort} onChange={(e) => onSort(e.target.value as ServerSort)}>
          {sorts.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
        </select>
      </label>
      {chosenId != null && <Link className="btn btn-sm" href={`/watchlists?id=${chosenId}`}>Manage watchlist</Link>}
    </section>
  );
}

export function WatchlistResults({ state, premium, prefs, sort, onSort, onCreated }: {
  state: ReturnType<typeof useScannerWatchlists>; premium: boolean; prefs: ReturnType<typeof useWorkspace>;
  sort: ServerSort; onSort: (sort: ServerSort) => void; onCreated: (id: number) => void;
}) {
  const { lists, chosen } = state;
  const detail = useApi(chosen ? `wl:${chosen.id}` : null, (signal) => watchlists.get(chosen!.id, signal));
  const chosenId = chosen?.id;
  const items = useMemo(() => (detail.data && detail.data.id === chosenId ? detail.data.items : []), [detail.data, chosenId]);
  const unranked = useMemo(() => items.filter((i) => !i.latest).map((i) => i.ticker).sort(), [items]);
  const quotes = useApi(unranked.length ? `wl-quotes:${unranked.join(",")}` : null, (signal) => watchlistQuotes(unranked, signal));

  const { rows, ranks } = useMemo(() => {
    const q = quotes.data?.quotes ?? {};
    const ranks: Record<string, number | null> = {};
    const rows: Row[] = items.map((i) => {
      if (i.latest) {
        const { rank, ...row } = i.latest;
        ranks[i.ticker] = rank;
        return { ...row, ticker: i.ticker } as Row;
      }
      ranks[i.ticker] = null;
      const p = q[i.ticker];
      return { ticker: i.ticker, score: 0, fading: false, signals: [], last: p?.last ?? null, chg_pct: p?.chg_pct ?? null, gap_pct: p?.gap_pct ?? null };
    });
    return { rows, ranks };
  }, [items, quotes.data]);
  const sorted = useMemo(() => sortWatchlistRows(rows, ranks, sort), [rows, ranks, sort]);

  if (lists.error && !lists.data) return <ErrorState error={lists.error} onRetry={lists.reload} what="your watchlists" />;
  if (!lists.data || (chosen && !detail.data && !detail.error) || (detail.loading && detail.data?.id !== chosen?.id)) {
    return <Skeleton rows={6} label="Loading your watchlist" />;
  }
  if (!chosen) return (
    <Card><Empty title="You don't have a watchlist yet.">
      <p>Save stocks to a watchlist and they show here with their HSF Score and setup.</p>
      <StarterWatchlistButton onDone={(id) => { lists.reload(); onCreated(id); }} />
    </Empty></Card>
  );
  if (detail.error && !detail.data) return <ErrorState error={detail.error} onRetry={detail.reload} what="this watchlist" />;
  if (!items.length) return (
    <Card><Empty title={`${chosen.name} has no stocks yet.`}>
      <StarterWatchlistButton listId={chosen.id} onDone={() => { detail.reload(); lists.reload(); }} />
    </Empty></Card>
  );
  const rankedCount = items.length - unranked.length;
  return (
    <>
      <p className="cap" data-testid="watchlist-scan-summary">
        {chosen.name}: {items.length} stock{items.length === 1 ? "" : "s"} · {rankedCount} ranked in the latest scan
        {detail.data?.scan_total ? ` of ${detail.data.scan_total}` : ""}. Rank is the stock&apos;s place in the whole scan.
        {unranked.length > 0 && " Stocks that didn't rank show their latest price."}
      </p>
      <section className={`card flush${detail.loading || quotes.loading ? " dim" : ""}`} aria-busy={detail.loading || quotes.loading}>
        <SetupTable rows={sorted} premium={premium} sparks displayControls={false} prefs={prefs} ranks={ranks} sort={sort} onSort={onSort} />
      </section>
    </>
  );
}
