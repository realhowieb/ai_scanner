"use client";

// Earnings scanner: the latest scan's setups that report earnings soon, grouped by day.
// Today is free on every plan; tomorrow and the 7-day window are Pro (can_earnings).
import Link from "next/link";
import { createContext, useContext } from "react";
import { createSortedRowModel, createColumnHelper, rowSortingFeature, tableFeatures, useTable } from "@tanstack/react-table";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, Empty, ErrorState, Locked, Pill, ScoreBadge, ScoreBar, Skeleton, TickerLink, UpgradeButton } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { useAutoRefresh } from "@/hooks/useAutoRefresh";
import { num, pct, price, setupLabel } from "@/lib/format";

import { SaveToWatchlistButton } from "./SaveToWatchlist";
import { RvolMeter } from "./Sparks";
import { Hint } from "@/components/Hint";

export const EARNINGS_WHEN = [
  { id: "today", label: "Today" },
  { id: "tomorrow", label: "Tomorrow", pro: true },
  { id: "week", label: "Next 7 days", pro: true },
] as const;
export const EARNINGS_TIMES = [
  { id: "", label: "Any time" },
  { id: "bmo", label: "Before open" },
  { id: "amc", label: "After close" },
] as const;
export type EarningsWhen = (typeof EARNINGS_WHEN)[number]["id"];
export type EarningsTime = "" | "bmo" | "amc";
export type EarningsFilters = { when: EarningsWhen; time: EarningsTime; all: boolean };

type Row = Schemas["EarningsSetup"];

const TIME_LABEL: Record<string, string> = { bmo: "Before open", amc: "After close" };

export function readEarningsFilters(sp: URLSearchParams): EarningsFilters {
  const w = sp.get("when");
  const t = sp.get("etime");
  return {
    when: (EARNINGS_WHEN.some((x) => x.id === w) ? w : "today") as EarningsWhen,
    time: (t === "bmo" || t === "amc" ? t : "") as EarningsTime,
    all: sp.get("all") === "1",
  };
}

export function earningsLocked(f: EarningsFilters, pro: boolean): boolean {
  return f.when !== "today" && !pro;
}

/** The earnings scan for these filters; null key (no request) while the window is locked. */
export function useEarningsScan(f: EarningsFilters, enabled: boolean) {
  const key = enabled ? `earnings-scan:${f.when}:${f.time}:${f.all ? 1 : 0}` : null;
  const res = useApi(key, (signal) => unwrap(api.GET("/v1/scans/latest/earnings", {
    params: { query: { window: f.when, ...(f.time ? { time: f.time } : {}), ...(f.all ? { include_unranked: true } : {}) } },
    signal,
  })));
  useAutoRefresh(res.reload, { enabled: key !== null });
  return res;
}

function dayLabel(date: string, days: number): string {
  const d = new Date(`${date}T12:00:00Z`);
  const named = d.toLocaleDateString("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" });
  return days === 0 ? `Today · ${named}` : days === 1 ? `Tomorrow · ${named}` : named;
}

function groups(rows: Row[]): { date: string; label: string; rows: Row[] }[] {
  const out: { date: string; label: string; rows: Row[] }[] = [];
  for (const r of rows) {
    const last = out[out.length - 1];
    if (last && last.date === r.earnings_date) last.rows.push(r);
    else out.push({ date: r.earnings_date, label: dayLabel(r.earnings_date, r.days_until), rows: [r] });
  }
  return out;
}

function When({ time }: { time: string | null | undefined }) {
  const label = (time && TIME_LABEL[time]) || "Time TBA";
  return <span className={`pill earn-time${time === "bmo" ? " earn-bmo" : time === "amc" ? " earn-amc" : ""}`}>{label}</span>;
}

/** Runs a custom scan on every stock reporting in the window, so the unranked ones get an HSF Score too. */
function ScoreReporters({ f, count }: { f: EarningsFilters; count: number }) {
  if (count <= 0) return null;
  return (
    <div className="row-actions earn-score">
      <span className="cap">{count} stock{count === 1 ? "" : "s"} report{count === 1 ? "s" : ""} {f.when === "week" ? "in the next 7 days" : f.when} but {count === 1 ? "isn't" : "aren't"} in the latest scan&apos;s ranking.</span>
      <Link className="btn btn-sm" href={scoreHref(f)}>Score every reporter</Link>
    </div>
  );
}

function Tags({ r }: { r: Row }) {
  return <>{r.on_watchlist && <> <Pill>Watchlist</Pill></>}{r.fading && <> <Pill tone="warn">Fading</Pill></>}</>;
}

const scoreHref = (f: EarningsFilters) => `/scanner/custom?universe=earnings&when=${f.when}&start=1`;
/** Scored by the market scan or by the user's own earnings scan. */
const hasScore = (r: Row) => r.score !== null && r.score !== undefined;

function SetupCell({ r, f }: { r: Row; f: EarningsFilters }) {
  if (r.ranked) return <Pill>{setupLabel(r.primary_setup)}</Pill>;
  if (hasScore(r)) return <><Pill>{setupLabel(r.primary_setup)}</Pill> <span className="pill earn-yours" title="Scored by your earnings scan">Your scan</span></>;
  return <><span className="cap">Not ranked</span> <Link className="earn-score-link" href={scoreHref(f)}>Score it</Link></>;
}

function timeOf(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }) + " ET";
}

/** Window, timing and ranked-only controls. */
export function EarningsControls({ f, pro, set }: { f: EarningsFilters; pro: boolean; set: (patch: Record<string, string | number>) => void }) {
  return (
    <div className="row-actions earn-controls">
      <div className="seg-group" role="group" aria-label="Reporting window">
        {EARNINGS_WHEN.map((w) => (
          <button key={w.id} type="button" className={`seg${f.when === w.id ? " on" : ""}`} aria-pressed={f.when === w.id}
            onClick={() => set({ when: w.id === "today" ? "" : w.id })}>
            {w.label}{"pro" in w && w.pro && !pro && <span className="chip-lock"> · Pro</span>}
          </button>
        ))}
      </div>
      <div className="seg-group" role="group" aria-label="Report timing">
        {EARNINGS_TIMES.map((t) => (
          <button key={t.id || "any"} type="button" className={`seg${f.time === t.id ? " on" : ""}`} aria-pressed={f.time === t.id}
            onClick={() => set({ etime: t.id })}>{t.label}</button>
        ))}
      </div>
      <div className="seg-group" role="group" aria-label="Which names">
        <button type="button" className={`seg${!f.all ? " on" : ""}`} aria-pressed={!f.all} onClick={() => set({ all: 0 })}>Ranked setups</button>
        <button type="button" className={`seg${f.all ? " on" : ""}`} aria-pressed={f.all} onClick={() => set({ all: 1 })}>All reporters</button>
      </div>
    </div>
  );
}

// Wide-screen table (TanStack Table). Rows stay grouped by report day; a header click sorts
// within each day (high to low, then low to high, then back to the scan's HSF order).
const earnFeatures = tableFeatures({ rowSortingFeature, sortedRowModel: createSortedRowModel() });
const ecol = createColumnHelper<typeof earnFeatures, Row>();
const blank = (v: number | null | undefined) => v ?? undefined;
const FiltersCtx = createContext<EarningsFilters>({ when: "today", time: "", all: false });
function EarnSetupCell({ row }: { row: { original: Row } }) { return <SetupCell r={row.original} f={useContext(FiltersCtx)} />; }
const EARN_COLUMNS = ecol.columns([
  ecol.display({ id: "ticker", header: "Ticker", cell: (c) => <><TickerLink ticker={c.row.original.ticker} /><Tags r={c.row.original} /></> }),
  ecol.display({ id: "reports", header: "Reports", cell: (c) => <When time={c.row.original.time} /> }),
  ecol.display({ id: "setup", header: "Setup", cell: EarnSetupCell }),
  ecol.accessor((r) => hasScore(r) ? blank(r.score) : undefined, { id: "score", header: "HSF Score", sortUndefined: "last",
    cell: (c) => hasScore(c.row.original) ? <ScoreBar score={c.row.original.score} /> : <span className="cap">—</span> }),
  ecol.accessor((r) => blank(r.last), { id: "last", header: "Last", sortUndefined: "last", cell: (c) => price(c.row.original.last) }),
  ecol.accessor((r) => blank(r.chg_pct), { id: "change", header: "Chg %", sortUndefined: "last", cell: (c) => pct(c.row.original.chg_pct) }),
  ecol.accessor((r) => blank(r.gap_pct), { id: "gap", header: "Gap %", sortUndefined: "last", cell: (c) => pct(c.row.original.gap_pct) }),
  ecol.accessor((r) => hasScore(r) ? blank(r.rvol) : undefined, { id: "rvol", header: "RVOL", sortUndefined: "last",
    cell: (c) => hasScore(c.row.original) ? <RvolMeter rvol={c.row.original.rvol} /> : null }),
  ecol.display({ id: "save", header: () => <span className="sr-only">Save</span>, cell: (c) => <SaveToWatchlistButton ticker={c.row.original.ticker} compact /> }),
]);
const EARN_TH: Record<string, string> = { score: "w30", last: "num", change: "num", gap: "num" };
const EARN_HINTS: Record<string, [string, string]> = {
  score: ["HSF Score", "An opportunity ranking from 0 to 100 from the latest scan. Rows marked Your scan were scored by your own earnings scan."],
  rvol: ["Relative volume", "Today's volume compared with the stock's recent average. 2.0 means twice as much trading as usual."],
};

function EarningsTable({ rows, days, f }: { rows: Row[]; days: { date: string; label: string; rows: Row[] }[]; f: EarningsFilters }) {
  const t = useTable({ features: earnFeatures, columns: EARN_COLUMNS, data: rows, sortDescFirst: true, enableMultiSort: false });
  const sorted = t.getRowModel().rows;
  const tdClass = (id: string, r: Row) => id === "last" || id === "gap" ? "num mono"
    : id === "change" ? `num mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}` : id === "save" ? "num" : undefined;
  return (
    <FiltersCtx.Provider value={f}>
      <div className="table-wrap only-wide" tabIndex={0} role="region" aria-label="Setups reporting earnings">
        <table className="table earn-table">
          <thead>
            {t.getHeaderGroups().map((g) => (
              <tr key={g.id}>
                {g.headers.map((h) => {
                  const on = h.column.getIsSorted();
                  const hint = EARN_HINTS[h.column.id];
                  return (
                    <th key={h.id} scope="col" className={EARN_TH[h.column.id]}
                      aria-sort={h.column.getCanSort() ? (on === "desc" ? "descending" : on === "asc" ? "ascending" : "none") : undefined}>
                      {h.column.getCanSort()
                        ? <button type="button" className="th-sort" onClick={h.column.getToggleSortingHandler()}>
                            <t.FlexRender header={h} />{on && <span aria-hidden="true">{on === "desc" ? " ↓" : " ↑"}</span>}
                          </button>
                        : <t.FlexRender header={h} />}
                      {hint && <Hint label={hint[0]}>{hint[1]}</Hint>}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          {days.map((g) => (
            <tbody key={g.date}>
              <tr className="earn-day"><th scope="colgroup" colSpan={EARN_COLUMNS.length}>{g.label}</th></tr>
              {sorted.filter((row) => row.original.earnings_date === g.date).map((row) => (
                <tr key={row.original.ticker}>
                  {row.getAllCells().map((cell) => <td key={cell.id} className={tdClass(cell.column.id, row.original)}><t.FlexRender cell={cell} /></td>)}
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>
    </FiltersCtx.Provider>
  );
}

export function EarningsResults({ f, pro, res, onClear }: {
  f: EarningsFilters; pro: boolean;
  res: ReturnType<typeof useEarningsScan>; onClear: () => void;
}) {
  if (earningsLocked(f, pro)) {
    return (
      <Card><Locked title="The earnings week ahead is part of Pro" plan="pro">
        Who reports today is free. Pro adds tomorrow and the next 7 days, so you can line up setups before they report.
      </Locked></Card>
    );
  }
  const { data, error, loading, refreshing, reload } = res;
  if (error && !data) return <ErrorState error={error} onRetry={reload} what="the earnings scanner" />;
  if (loading && !data) return <Skeleton rows={6} label="Loading earnings" />;
  if (!data) return null;
  const window = EARNINGS_WHEN.find((w) => w.id === f.when)!.label.toLowerCase();
  if (data.setups.length === 0) {
    return (
      <Card><Empty title={`No ${f.all ? "stocks" : "ranked setups"} report ${f.when === "week" ? "in the next 7 days" : window}${f.time ? ` ${(TIME_LABEL[f.time] ?? "").toLowerCase()}` : ""}.`}>
        {!f.all && data.unranked_total > 0
          ? <><p>{data.unranked_total} stock{data.unranked_total === 1 ? "" : "s"} report but didn&apos;t rank in the latest scan. <button type="button" className="btn btn-sm" onClick={onClear}>Show all reporters</button></p>
            <p><Link className="btn btn-sm" href={scoreHref(f)}>Score every reporter</Link></p></>
          : <p>Dates come from the earnings calendar, which refreshes every day.</p>}
      </Empty></Card>
    );
  }
  const days = groups(data.setups);
  return (
    <>
      <section className={`card flush${refreshing ? " dim" : ""}`} aria-busy={refreshing}>
        <EarningsTable rows={data.setups} days={days} f={f} />
        <div className="only-narrow">
          {days.map((g) => (
            <section key={g.date} aria-label={g.label}>
              <h3 className="earn-day-h">{g.label}</h3>
              <ul className="cards">
                {g.rows.map((r) => (
                  <li key={r.ticker} className="result-card setup-card">
                    <div className="sc-top">
                      <TickerLink ticker={r.ticker} />
                      <When time={r.time} />
                      <Tags r={r} />
                      <span className="grow" />
                      <SaveToWatchlistButton ticker={r.ticker} compact />
                    </div>
                    <dl className="sc-facts">
                      {hasScore(r) && <>
                        <div><dt>HSF</dt><dd><ScoreBadge score={r.score} /></dd></div>
                        <div><dt>Setup</dt><dd>{setupLabel(r.primary_setup)}</dd></div>
                      </>}
                      <div><dt>Last</dt><dd className="mono">{price(r.last)}</dd></div>
                      <div><dt>Chg</dt><dd className={`mono ${(r.chg_pct ?? 0) > 0 ? "up" : (r.chg_pct ?? 0) < 0 ? "down" : ""}`}>{pct(r.chg_pct)}</dd></div>
                      {hasScore(r) && <div><dt>RVOL</dt><dd className="mono">{num(r.rvol)}</dd></div>}
                    </dl>
                    {!r.ranked && (hasScore(r)
                      ? <p className="cap">Scored by your earnings scan.</p>
                      : <p className="cap">Reports, but didn&apos;t rank in the latest scan. <Link href={scoreHref(f)}>Score it</Link></p>)}
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </section>
      <p className="cap">
        Ranked setups are stocks in the latest scan that report {f.when === "week" ? "in the next 7 days" : window}, soonest first, then by HSF Score.
        {f.all ? " All reporters adds stocks that report but didn't rank, with live prices." : ""}
        {f.all && data.custom_scan_at ? ` Rows marked Your scan were scored by your earnings scan at ${timeOf(data.custom_scan_at)}.` : ""} Dates and times come from the earnings calendar and can move.
      </p>
      <ScoreReporters f={f} count={data.unranked_total} />
      {data.limited && (
        <div className="cap-note" role="note">
          <p>Your plan shows the top <b>{data.max_results}</b> of <b>{data.total}</b> ranked setups that report in this window.</p>
          <UpgradeButton plan={data.max_results < 100 ? "pro" : "premium"} />
        </div>
      )}
      {!pro && <p className="cap">See tomorrow and the next 7 days with <Link href="/pricing">Pro</Link>.</p>}
    </>
  );
}
