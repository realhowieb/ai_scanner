"use client";

import Link from "@/components/Link";
import { ResearchGlossary } from "@/components/ResearchGlossary";
import { createContext, useContext, useState } from "react";
import type { ReactNode } from "react";
import { columnOrderingFeature, columnVisibilityFeature, createColumnHelper, rowSortingFeature, tableFeatures, useTable } from "@tanstack/react-table";
import type { SortingState } from "@tanstack/react-table";
import { useWorkspace } from "@/hooks/useWorkspace";
import { DEFAULT_COLUMNS, TablePreferences, columnOrder } from "./TablePreferences";

import type { Schemas } from "@/api/client";
import { Hint } from "@/components/Hint";
import { Pill, ScoreBadge, ScoreBar, TickerLink } from "@/components/ui";
import { num, pct, price, setupLabel } from "@/lib/format";

import { SaveToWatchlistButton } from "./SaveToWatchlist";
import { DailySpark, PreBreakoutMeter, RvolMeter, useDailySparks, useNewsFlags } from "./Sparks";

type Row = Schemas["ScanSetup"];
/** Server sort ids (GET /v1/scans/latest `sort`). Every server sort is high-to-low. */
export type ServerSort = "score" | "chg_pct" | "gap_pct" | "rvol" | "prob";
/** Column id → the server sort it maps to. Other columns cannot be sorted. */
const SERVER_SORT: Partial<Record<string, ServerSort>> = { score: "score", change: "chg_pct", gap: "gap_pct", rvol: "rvol", prob: "prob" };

// Headless table (TanStack Table): column visibility and order come from the saved
// Table display settings; sorting is manual because the API ranks and pages the rows.
const features = tableFeatures({ columnVisibilityFeature, columnOrderingFeature, rowSortingFeature });
const col = createColumnHelper<typeof features, Row>();
const upDown = (v: number | null | undefined) => (v ?? 0) > 0 ? "up" : (v ?? 0) < 0 ? "down" : "";

const HINTS: Record<string, [string, ReactNode]> = {
  score: ["HSF Score", "An opportunity ranking from 0 to 100 that blends trend, momentum, volume and setup quality. It is not a probability of profit."],
  rvol: ["Relative volume", "Today's volume compared with the stock's recent average. 2.0 means twice as much trading as usual; the bar turns amber from 1.5."],
  prob: ["PreBreakout", "The model's calibrated chance of a breakout soon. The bar shows where the model ranks the name in this scan (full bar = top); names in the top 10% are labelled."],
};

/** What the cells need from the table's props. Cells read it from context so the column
 * definitions stay module constants: FlexRender renders cell functions as components, and a
 * new function each render would remount every cell (losing an open Save dialog, for one). */
type NewsFlag = Schemas["NewsFlag"];
type Ctx = { numbered: boolean; offset: number; sparks: boolean; series: Record<string, number[]>; news: Record<string, NewsFlag>; choose: (ticker: string) => ReactNode };
const TableCtx = createContext<Ctx>({ numbered: true, offset: 0, sparks: false, series: {}, news: {}, choose: () => null });

/** "News" marker: the ticker had a story in the last 24 hours. Links to the stock page's News card. */
function NewsMark({ ticker, flag }: { ticker: string; flag: NewsFlag | undefined }) {
  if (!flag) return null;
  return <Link href={`/stocks/${encodeURIComponent(ticker)}#news`} className="pill news-flag" title={flag.headline}
    aria-label={`${ticker} news: ${flag.count} ${flag.count === 1 ? "story" : "stories"} in 24 hours. Latest: ${flag.headline}`}>News</Link>;
}
const useCtx = () => useContext(TableCtx);

function RankHead() { return useCtx().numbered ? "#" : <span className="sr-only">Row</span>; }
function RankCell({ row }: { row: { index: number } }) { const c = useCtx(); return c.numbered ? c.offset + row.index + 1 : ""; }
function TickerCell({ row }: { row: { original: Row } }) {
  const c = useCtx(); const r = row.original;
  return <>{c.sparks && c.choose(r.ticker)} <TickerLink ticker={r.ticker} />{r.fading && <> <Pill tone="warn">Fading</Pill></>}<NewsMark ticker={r.ticker} flag={c.news[r.ticker]} /></>;
}
function SparkCell({ row }: { row: { original: Row } }) { const c = useCtx(); return <DailySpark points={c.series[row.original.ticker]} ticker={row.original.ticker} />; }

const COLUMNS = col.columns([
  col.display({ id: "rank", header: RankHead, cell: RankCell }),
  col.display({ id: "ticker", header: "Ticker", cell: TickerCell }),
  col.display({ id: "setup", header: "Setup", cell: (c) => <Pill>{setupLabel(c.row.original.primary_setup)}</Pill> }),
  col.display({ id: "score", header: "HSF Score", cell: (c) => <ScoreBar score={c.row.original.score} /> }),
  col.display({ id: "spark", header: "20 sessions", cell: SparkCell }),
  col.display({ id: "last", header: "Last", cell: (c) => price(c.row.original.last) }),
  col.display({ id: "change", header: "Chg %", cell: (c) => pct(c.row.original.chg_pct) }),
  col.display({ id: "gap", header: "Gap %", cell: (c) => pct(c.row.original.gap_pct) }),
  col.display({ id: "rvol", header: "RVOL", cell: (c) => <RvolMeter rvol={c.row.original.rvol} /> }),
  col.display({ id: "prob", header: "PreBreakout", cell: (c) => <PreBreakoutMeter prob={c.row.original.prob} rank={c.row.original.prob_rank} /> }),
  col.display({ id: "save", header: () => <span className="sr-only">Save</span>, cell: (c) => <SaveToWatchlistButton ticker={c.row.original.ticker} compact /> }),
]);
const TH_CLASS: Record<string, string> = { score: "w-score", last: "num", change: "num", gap: "num" };

/** Ranked rows, as a table on wide screens and as cards on phones. Rows come ranked from the API. */
/** `numbered` = rows are in HSF rank order, so the position is the rank; off for other sort orders. */
/** The PreBreakout column shows only on Premium; other plans get one note under the rows instead of "Premium" per row. */
/** `sparks` adds each ticker's 20-session sparkline: for current scans only, not saved history. */
/** `sort` + `onSort` make the sortable headers buttons that ask the server for a new order. */
export function SetupTable({ rows, offset = 0, premium, numbered = true, sparks = false, displayControls = true, sort, onSort }: {
  rows: Row[]; offset?: number; premium: boolean; numbered?: boolean; sparks?: boolean; displayControls?: boolean;
  sort?: ServerSort; onSort?: (sort: ServerSort) => void;
}) {
  const prefs = useWorkspace();
  const [selected, setSelected] = useState<string[]>([]);
  const table = prefs.data?.table;
  const shown = sparks ? (table?.columns ?? [...DEFAULT_COLUMNS]) : [...DEFAULT_COLUMNS];
  const visible = (c: string) => shown.some((x) => x === c);
  const toggle = (ticker: string) => setSelected((prev) => prev.includes(ticker) ? prev.filter((t) => t !== ticker) : prev.length < 4 ? [...prev, ticker] : prev);
  const choose = (ticker: string) => <input type="checkbox" aria-label={`Compare ${ticker}`} checked={selected.includes(ticker)}
    disabled={selected.length >= 4 && !selected.includes(ticker)} onChange={() => toggle(ticker)} />;
  const series = useDailySparks(sparks && visible("spark") ? rows.map((r) => r.ticker) : []);
  const news = useNewsFlags(sparks ? rows.map((r) => r.ticker) : []);

  const ctx = { numbered, offset, sparks, series, news, choose };
  const columnVisibility = Object.fromEntries(DEFAULT_COLUMNS.map((c) => [c, visible(c) && (c !== "spark" || sparks) && (c !== "prob" || premium)]));
  const sorting: SortingState = sort ? [{ id: Object.keys(SERVER_SORT).find((k) => SERVER_SORT[k] === sort) ?? "score", desc: true }] : [];
  const t = useTable({
    features, columns: COLUMNS, data: rows,
    state: { columnVisibility, columnOrder: ["rank", "ticker", "setup", "score", ...columnOrder(shown), "save"], sorting },
    manualSorting: true, enableSortingRemoval: false, enableMultiSort: false, sortDescFirst: true,
    enableSorting: Boolean(onSort),
    onSortingChange: (next) => {
      const resolved = typeof next === "function" ? next(sorting) : next;
      const id = resolved[0]?.id;
      if (onSort) onSort(id && SERVER_SORT[id] ? SERVER_SORT[id] : "score");
    },
  });

  const cellClass = (id: string, r: Row) => id === "rank" ? "mono cap"
    : id === "last" || id === "gap" ? "num mono" : id === "change" ? `num mono ${upDown(r.chg_pct)}` : id === "save" ? "num" : undefined;

  return (
    <div className={sparks && table?.compact ? "scanner-results is-compact" : "scanner-results"}>
      {sparks && displayControls && prefs.data && <TablePreferences key={JSON.stringify(table)} initial={table ?? { columns: [...DEFAULT_COLUMNS], compact: false }} premium={premium} />}
      {sparks && selected.length > 0 && <div className="row-actions compare-actions">
        <span className="cap">Select 2–4 stocks to compare · {selected.length} selected</span>
        {selected.length >= 2 && <Link className="btn btn-sm" href={`/stocks/compare?symbols=${encodeURIComponent(selected.join(","))}`}>Compare selected</Link>}
        {selected.length > 0 && <button type="button" className="btn btn-sm" onClick={() => setSelected([])}>Clear selection</button>}
      </div>}
      <p className="table-scroll-hint cap">Scroll the table for more columns · tick boxes select 2–4 stocks to compare.</p>
      <TableCtx.Provider value={ctx}>
      <div className="table-wrap only-wide" tabIndex={0} role="region" aria-label="Ranked setups table; scroll for more columns">
        <table className="table">
          <thead>
            {t.getHeaderGroups().map((g) => (
              <tr key={g.id}>
                {g.headers.map((h) => {
                  const id = h.column.id;
                  const canSort = Boolean(onSort && SERVER_SORT[id]) && (id !== "prob" || premium);
                  const on = h.column.getIsSorted();
                  const hint = HINTS[id];
                  return (
                    <th key={h.id} scope="col" className={TH_CLASS[id]} aria-sort={canSort ? (on ? "descending" : "none") : undefined}>
                      {canSort
                        ? <button type="button" className="th-sort" onClick={() => h.column.toggleSorting(true)}>
                            <t.FlexRender header={h} />{on && <span aria-hidden="true"> ↓</span>}
                          </button>
                        : <t.FlexRender header={h} />}
                      {hint && <Hint label={hint[0]}>{hint[1]}</Hint>}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {t.getRowModel().rows.map((row) => (
              <tr key={`${row.original.ticker}-${row.index}`}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className={cellClass(cell.column.id, row.original)}><t.FlexRender cell={cell} /></td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </TableCtx.Provider>
      <ol className="cards only-narrow" start={offset + 1}>
        {rows.map((r, i) => (
          <li key={`${r.ticker}-${i}`} className="result-card setup-card">
            <div className="sc-top">
              {numbered && <span className="mono cap">#{offset + i + 1}</span>}
              {sparks && choose(r.ticker)}
              <TickerLink ticker={r.ticker} />
              {visible("setup") && <Pill>{setupLabel(r.primary_setup)}</Pill>}
              {r.fading && <Pill tone="warn">Fading</Pill>}
              <NewsMark ticker={r.ticker} flag={news[r.ticker]} />
              <span className="grow" />
              <SaveToWatchlistButton ticker={r.ticker} compact />
            </div>
            <dl className="sc-facts">
              <div><dt>HSF</dt><dd><ScoreBadge score={r.score} /></dd></div>
              <div hidden={!visible("last")}><dt>Last</dt><dd className="mono">{price(r.last)}</dd></div>
              <div hidden={!visible("change")}><dt>Chg</dt><dd className={`mono ${upDown(r.chg_pct)}`}>{pct(r.chg_pct)}</dd></div>
              <div hidden={!visible("gap")}><dt>Gap</dt><dd className="mono">{pct(r.gap_pct)}</dd></div>
              <div hidden={!visible("rvol")}><dt>RVOL</dt><dd className="mono">{num(r.rvol)}</dd></div>
              {premium && <div hidden={!visible("prob")}><dt>PreBreakout</dt><dd><PreBreakoutMeter prob={r.prob} rank={r.prob_rank} /></dd></div>}
              {sparks && visible("spark") && <div className="sc-spark"><dt className="sr-only">20 sessions</dt><dd><DailySpark points={series[r.ticker]} ticker={r.ticker} width={72} height={22} /></dd></div>}
            </dl>
          </li>
        ))}
      </ol>
      <ResearchGlossary />
      {!premium && <p className="cap table-note">PreBreakout probability is part of <Link href="/pricing">Premium</Link>.</p>}
    </div>
  );
}

