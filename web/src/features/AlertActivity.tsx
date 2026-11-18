"use client";

// Alerts → Activity and Tickers: fired alerts as a compact table grouped by day, with
// filters (kind, time range, ticker, new names only), a "Fired most" panel, and a per-ticker
// table. Everything here works on the events the page already loaded.
import Link from "@/components/Link";
import { Fragment, useMemo, useState } from "react";

import { watchlists } from "@/api/userData";
import { Card, ErrorLine, Pill } from "@/components/ui";
import { useAction } from "@/hooks/useAction";
import { useApi } from "@/hooks/useApi";
import { useNarrow } from "@/hooks/useNarrow";
import { bestValue, dayLabel, etDay, tickerStats } from "@/lib/alertEvents";
import type { FiredItem, FiredRow, TickerStat } from "@/lib/alertEvents";
import { etTime } from "@/lib/format";

export type Range = "today" | "7d" | "30d" | "all";
export const RANGES: [Range, string][] = [["today", "Today"], ["7d", "7 days"], ["30d", "30 days"], ["all", "All loaded"]];
export type ActivityFilter = { kind: string; range: Range; ticker: string; freshOnly: boolean };
export const DEFAULT_FILTER: ActivityFilter = { kind: "", range: "7d", ticker: "", freshOnly: false };

const DAY_MS = 24 * 60 * 60 * 1000;

export function inRange(item: FiredItem, range: Range, now = Date.now()): boolean {
  if (range === "all") return true;
  const t = item.firedAt ? new Date(item.firedAt).getTime() : NaN;
  if (Number.isNaN(t)) return false;
  if (range === "today") return etDay(item.firedAt) === etDay(new Date(now).toISOString());
  return now - t <= (range === "7d" ? 7 : 30) * DAY_MS;
}

const hasTicker = (item: FiredItem, t: string) => item.ticker === t || item.rows.some((r) => r.ticker === t);

/** The alerts a filter keeps, newest first. `ignoreTicker` for the Fired most panel, which picks the ticker. */
export function applyFilter(items: FiredItem[], f: ActivityFilter, ignoreTicker = false): FiredItem[] {
  const t = f.ticker.trim().toUpperCase();
  return items.filter((i) => (!f.kind || i.title === f.kind) && inRange(i, f.range)
    && (ignoreTicker || !t || hasTicker(i, t))
    && (!f.freshOnly || i.rows.length === 0 || i.fresh === null || i.fresh.size > 0));
}

const KIND_CLASS: Record<string, string> = { Breakout: "al-k-breakout", Watchlist: "al-k-watchlist", Price: "al-k-price", Move: "al-k-price", RVOL: "al-k-volume" };
export const kindLabel = (title: string) => (title === "RVOL" ? "Volume" : title);
export function KindTag({ title, extra }: { title: string; extra?: string }) {
  return <span className={`al-kind ${KIND_CLASS[title] ?? "al-k-other"}`}>{kindLabel(title)}{extra ? ` ${extra}` : ""}</span>;
}

function headOf(item: FiredItem): string {
  const many = item.rows.length > 1 || item.total > 1;
  return many ? `${item.title} · ${item.total} names` : item.ticker ? `${item.title} · ${item.ticker}` : item.title;
}

const tierText = (item: FiredItem) => item.tiers.filter((t) => t.rule).map((t) => `≥${t.rule}`).join(" · ");

function NameChip({ r, item, focus }: { r: FiredRow; item: FiredItem; focus: string }) {
  const isNew = item.fresh?.has(r.ticker) ?? false;
  const repeat = item.fresh !== null && !isNew;
  return (
    <li className={`al-name${repeat ? " rep" : ""}${focus === r.ticker ? " focus" : ""}`} title={r.earnings ? `${r.detail} · ${r.earnings}` : r.detail || undefined}>
      {isNew && <span className="al-new" aria-hidden="true" />}
      <Link href={`/stocks/${encodeURIComponent(r.ticker)}`} className="mono strong">{r.ticker}</Link>
      {isNew && <span className="sr-only"> (new)</span>}
      {r.value !== null && <span className="mono al-val">{r.value.toFixed(1)}</span>}
      {r.earnings && <span className="fired-warn" aria-label={r.earnings}>⚠️</span>}
    </li>
  );
}

function AddToWatchlist({ tickers }: { tickers: string[] }) {
  const lists = useApi("watchlists", (signal) => watchlists.list(signal));
  const act = useAction();
  const [done, setDone] = useState<string | null>(null);
  if (!lists.data?.length) return null;
  return (
    <span className="al-add">
      <label className="sr-only" htmlFor={`al-add-${tickers.join("-").slice(0, 40)}`}>Add these tickers to a watchlist</label>
      <select id={`al-add-${tickers.join("-").slice(0, 40)}`} value="" disabled={act.busy} className="al-select"
        onChange={(e) => {
          const id = Number(e.target.value);
          const name = lists.data?.find((w) => w.id === id)?.name ?? "the list";
          void act.run(async () => {
            const r = await watchlists.addTickers(id, tickers);
            setDone(`Added ${r.added.length} to ${name}${r.added.length < tickers.length ? ` (${tickers.length - r.added.length} already there)` : ""}.`);
            return true;
          });
        }}>
        <option value="">{act.busy ? "Adding…" : "Add to watchlist…"}</option>
        {lists.data.map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
      </select>
      {done && <span className="cap" role="status">{done}</span>}
      <ErrorLine error={act.error} />
    </span>
  );
}

function CopyTickers({ tickers }: { tickers: string[] }) {
  const [copied, setCopied] = useState(false);
  return (
    <button type="button" className="link-btn" onClick={() => {
      void navigator.clipboard?.writeText(tickers.join(", ")).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }, () => undefined);
    }}>{copied ? "Copied" : "Copy tickers"}</button>
  );
}

/** The opened alert: every listed name, matched vs listed, and what to do with them. */
function Detail({ item, focus }: { item: FiredItem; focus: string }) {
  const tickers = item.rows.map((r) => r.ticker);
  const unlisted = item.total - item.rows.length;
  return (
    <div className="al-detail-body stack-xs">
      <div className="al-detail-head">
        <span className="strong">{unlisted > 0 ? `${item.total} matched · top ${item.rows.length} listed in the alert` : `${item.rows.length} name${item.rows.length === 1 ? "" : "s"}`}</span>
        {item.tiers.length > 1 && item.tiers.map((t) => <Pill key={t.rule}>{`≥ ${t.rule}: ${t.count}`}</Pill>)}
        {item.title === "Breakout" && <Link href="/scanner?signal=breakout">Open breakouts in Scanner →</Link>}
        {item.title === "Watchlist" && <Link href="/watchlists">Open watchlists →</Link>}
        {tickers.length > 1 && <CopyTickers tickers={tickers} />}
        {tickers.length > 0 && <AddToWatchlist tickers={tickers} />}
      </div>
      {item.rows.length > 0 && (
        <ul className="al-grid" aria-label={`All ${item.title} names`}>
          {item.rows.map((r) => <NameChip key={r.ticker} r={r} item={item} focus={focus} />)}
        </ul>
      )}
      {item.text && <span className="body-sm">{item.text}</span>}
      {item.rows.some((r) => r.value !== null) && <span className="cap">Sorted by score.{unlisted > 0 ? ` ${unlisted} more matched but weren't listed (an alert keeps its top ${item.rows.length}).` : ""}</span>}
      {item.tiers.length > 1 && <span className="cap">Your ≥ {item.tiers.map((t) => t.rule).join(" and ≥ ")} breakout alerts fired on the same scan, shown together.</span>}
      <span className="cap">Fired {etTime(item.firedAt)}.</span>
    </div>
  );
}

const TOP = 5;
const TOP_PHONE = 3;

function TopNames({ item, focus, n }: { item: FiredItem; focus: string; n: number }) {
  if (item.rows.length === 0) {
    return <span className="al-text">{item.ticker && <Link href={`/stocks/${encodeURIComponent(item.ticker)}`} className="mono strong">{item.ticker}</Link>} <span className="cap">{item.text}</span></span>;
  }
  // The searched ticker is always among the names shown.
  let rows = item.rows.slice(0, n);
  if (focus && !rows.some((r) => r.ticker === focus)) {
    const hit = item.rows.find((r) => r.ticker === focus);
    if (hit) rows = [...rows.slice(0, n - 1), hit];
  }
  const rest = item.total - rows.length;
  return (
    <ul className="al-names" aria-label={`${item.title} matches`}>
      {rows.map((r) => <NameChip key={r.ticker} r={r} item={item} focus={focus} />)}
      {rest > 0 && <li className="cap al-more">+{rest}</li>}
    </ul>
  );
}

export function ActivityTable({ items, filter, onFilter, next, onLoadOlder, loadingOlder, olderError }: {
  items: FiredItem[]; filter: ActivityFilter; onFilter: (f: ActivityFilter) => void;
  next: boolean; onLoadOlder: () => void; loadingOlder: boolean; olderError?: React.ReactNode;
}) {
  const narrow = useNarrow();
  const [open, setOpen] = useState<string[]>([]);
  const shown = useMemo(() => applyFilter(items, filter), [items, filter]);
  const focus = filter.ticker.trim().toUpperCase();
  const days = useMemo(() => {
    const out: { day: string; items: FiredItem[] }[] = [];
    for (const it of shown) {
      const d = etDay(it.firedAt);
      const last = out[out.length - 1];
      if (last && last.day === d) last.items.push(it); else out.push({ day: d, items: [it] });
    }
    return out;
  }, [shown]);
  const toggle = (k: string) => setOpen((o) => (o.includes(k) ? o.filter((x) => x !== k) : [...o, k]));
  const filtered = filter.kind || filter.ticker || filter.freshOnly;
  const rangeName = RANGES.find(([r]) => r === filter.range)![1].toLowerCase();

  const expander = (it: FiredItem) => (
    <button type="button" className="al-exp" aria-expanded={open.includes(it.key)} aria-label={`${open.includes(it.key) ? "Hide" : "Show"} details: ${headOf(it)}`}
      onClick={() => toggle(it.key)}>{open.includes(it.key) ? "▴" : "▸"}</button>
  );

  return (
    <div className="stack-sm">
      {shown.length === 0 ? (
        <p className="cap" role="status">No alerts {filter.range === "all" ? "loaded" : filter.range === "today" ? "today" : `in the last ${rangeName}`}{filtered ? " match these filters" : ""}.{" "}
          {filtered && <button type="button" className="link-btn" onClick={() => onFilter({ ...DEFAULT_FILTER, range: filter.range })}>Clear filters</button>}
          {filter.range !== "all" && <> <button type="button" className="link-btn" onClick={() => onFilter({ ...filter, range: "all" })}>Show all loaded</button></>}
        </p>
      ) : narrow ? (
        <div className="al-phone">
          {days.map((g) => (
            <section key={g.day} aria-label={dayLabel(g.day)}>
              <h3 className="al-day-h">{dayLabel(g.day)}</h3>
              <ul className="al-cards">
                {g.items.map((it) => (
                  <li key={it.key} className="al-card">
                    <div className="al-card-head">
                      <KindTag title={it.title} extra={tierText(it)} />
                      <span className="strong">{it.rows.length > 1 || it.total > 1 ? `${it.total} ${it.total > it.rows.length ? "matched" : "names"}` : it.ticker ?? ""}</span>
                      <span className="cap al-time">{etTime(it.firedAt, false).replace(" ET", "")}</span>
                      {expander(it)}
                    </div>
                    {!open.includes(it.key) ? <TopNames item={it} focus={focus} n={TOP_PHONE} /> : <Detail item={it} focus={focus} />}
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      ) : (
        <div className="table-wrap" role="region" aria-label="Fired alerts" tabIndex={0}>
          <table className="table al-table">
            <thead>
              <tr>
                <th scope="col">Time (ET)</th>
                <th scope="col">Alert</th>
                <th scope="col" className="num">Names</th>
                <th scope="col">Top names <span className="cap al-legend">(<span className="al-new" aria-hidden="true" /> new since that alert last fired; repeats dimmed)</span></th>
                <th scope="col" className="num">Best</th>
                <th scope="col"><span className="sr-only">Details</span></th>
              </tr>
            </thead>
            <tbody>
              {days.map((g) => (
                <Fragment key={g.day}>
                  <tr className="al-day"><th scope="colgroup" colSpan={6}>{dayLabel(g.day)} · {g.items.length} alert{g.items.length === 1 ? "" : "s"}</th></tr>
                  {g.items.map((it) => {
                    const best = bestValue(it);
                    return (
                      <Fragment key={it.key}>
                        <tr className={open.includes(it.key) ? "open" : undefined}>
                          <td className="cap al-time">{etTime(it.firedAt, false).replace(" ET", "")}</td>
                          <td><KindTag title={it.title} extra={tierText(it)} /><span className="sr-only"> {headOf(it)}</span></td>
                          <td className="num mono">{it.rows.length ? it.total : it.ticker ? 1 : ""}</td>
                          <td className="al-top"><TopNames item={it} focus={focus} n={TOP} /></td>
                          <td className="num mono up">{best !== null ? best.toFixed(1) : ""}</td>
                          <td className="al-exp-cell">{expander(it)}</td>
                        </tr>
                        {open.includes(it.key) && <tr className="al-detail"><td /><td colSpan={5}><Detail item={it} focus={focus} /></td></tr>}
                      </Fragment>
                    );
                  })}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="al-foot">
        <span className="cap">Showing {shown.length} of {items.length} loaded alert{items.length === 1 ? "" : "s"}.</span>
        {next && <button type="button" className="btn btn-sm" disabled={loadingOlder} onClick={onLoadOlder}>{loadingOlder ? "Loading…" : "Load older"}</button>}
      </div>
      {olderError}
    </div>
  );
}

export function ActivityFilters({ items, filter, onFilter, tickerBox = true }: {
  items: FiredItem[]; filter: ActivityFilter; onFilter: (f: ActivityFilter) => void; tickerBox?: boolean;
}) {
  const kinds = useMemo(() => [...new Set(items.map((i) => i.title))], [items]);
  return (
    <div className="al-filters">
      {kinds.length > 1 && (
        <div className="seg-group" role="radiogroup" aria-label="Alert kind">
          {[["", "All"] as const, ...kinds.map((k) => [k, kindLabel(k)] as const)].map(([k, label]) => (
            <label key={k || "all"} className={`seg${filter.kind === k ? " on" : ""}`}>
              <input type="radio" className="sr-only" name="al-kind" checked={filter.kind === k} onChange={() => onFilter({ ...filter, kind: k })} />{label}
            </label>
          ))}
        </div>
      )}
      <div className="seg-group" role="radiogroup" aria-label="Time range">
        {RANGES.map(([r, label]) => (
          <label key={r} className={`seg${filter.range === r ? " on" : ""}`}>
            <input type="radio" className="sr-only" name="al-range" checked={filter.range === r} onChange={() => onFilter({ ...filter, range: r })} />{label}
          </label>
        ))}
      </div>
      {tickerBox && (
        <>
          <label className="check"><input type="checkbox" checked={filter.freshOnly} onChange={(e) => onFilter({ ...filter, freshOnly: e.target.checked })} /><span>New names only</span></label>
          <label className="field al-search"><span className="sr-only">Filter by ticker</span>
            <input type="search" placeholder="Filter ticker…" value={filter.ticker} maxLength={10} autoCapitalize="characters" spellCheck={false}
              aria-label="Filter by ticker" onChange={(e) => onFilter({ ...filter, ticker: e.target.value.toUpperCase() })} />
          </label>
        </>
      )}
    </div>
  );
}

/** Names that keep showing up in the chosen range; a click filters Activity to one. */
export function FiredMost({ items, filter, onPick, onAll }: { items: FiredItem[]; filter: ActivityFilter; onPick: (t: string) => void; onAll: () => void }) {
  const stats = useMemo(() => tickerStats(applyFilter(items, { ...filter, freshOnly: false }, true)).slice(0, 8), [items, filter]);
  if (stats.length === 0) return null;
  const max = stats[0]!.count;
  const focus = filter.ticker.trim().toUpperCase();
  return (
    <Card title="Fired most" className="al-most" aside={<button type="button" className="link-btn" onClick={onAll}>All tickers →</button>}>
      <p className="cap">Names that keep showing up ({RANGES.find(([r]) => r === filter.range)![1].toLowerCase()}). Click one to filter.</p>
      <ul className="al-most-list">
        {stats.map((s) => (
          <li key={s.ticker}>
            <button type="button" className={`al-most-btn${focus === s.ticker ? " on" : ""}`} aria-pressed={focus === s.ticker}
              aria-label={`${s.ticker}: fired ${s.count} time${s.count === 1 ? "" : "s"}`} onClick={() => onPick(focus === s.ticker ? "" : s.ticker)}>
              <span className="mono strong">{s.ticker}</span>
              <span className="al-track" aria-hidden="true"><span className="al-bar" style={{ width: `${Math.max(8, (s.count / max) * 100)}%` }} /></span>
              <span className="cap">{s.count}× · {s.kinds.map(kindLabel).join(", ")}</span>
            </button>
          </li>
        ))}
      </ul>
    </Card>
  );
}

type TSort = "count" | "last" | "best" | "ticker";
const TSORT_LABEL: Record<TSort, string> = { count: "Times fired", last: "Last fired", best: "Best score", ticker: "Ticker" };

/** Tickers tab: every name that fired in the range, how often, when, and its best score. */
export function TickersTable({ items, filter, onFilter, onShow }: { items: FiredItem[]; filter: ActivityFilter; onFilter: (f: ActivityFilter) => void; onShow: (t: string) => void }) {
  const [sort, setSort] = useState<{ key: TSort; desc: boolean }>({ key: "count", desc: true });
  const [q, setQ] = useState("");
  const all = useMemo(() => tickerStats(applyFilter(items, { ...filter, ticker: "", freshOnly: false })), [items, filter]);
  const rows = useMemo(() => {
    const t = q.trim().toUpperCase();
    const list = t ? all.filter((s) => s.ticker.includes(t)) : all;
    const val = (s: TickerStat): number | string => sort.key === "count" ? s.count : sort.key === "best" ? s.best ?? -Infinity
      : sort.key === "last" ? (s.last ? new Date(s.last).getTime() : 0) : s.ticker;
    return [...list].sort((a, b) => {
      const x = val(a), y = val(b);
      const c = typeof x === "string" ? x.localeCompare(String(y)) : x - (y as number);
      return sort.desc ? -c : c;
    });
  }, [all, q, sort]);
  const th = (key: TSort, cls = "") => (
    <th scope="col" className={cls} aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}>
      <button type="button" className="th-sort" aria-label={`Sort by ${TSORT_LABEL[key]}`}
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "ticker" }))}>
        {TSORT_LABEL[key]}{sort.key === key && <span aria-hidden="true" className="th-arrow">{sort.desc ? " ▼" : " ▲"}</span>}
      </button>
    </th>
  );
  return (
    <Card title="Tickers" id="alert-tickers" aside={`${all.length} name${all.length === 1 ? "" : "s"}`}>
      <div className="al-filters">
        <ActivityFilters items={items} filter={filter} onFilter={onFilter} tickerBox={false} />
        <label className="field al-search"><span className="sr-only">Find a ticker</span>
          <input type="search" placeholder="Find ticker…" value={q} aria-label="Find a ticker" onChange={(e) => setQ(e.target.value.toUpperCase())} />
        </label>
      </div>
      {rows.length === 0 ? <p className="cap">No tickers fired in this range.</p> : (
        <div className="table-wrap" role="region" aria-label="Tickers that fired" tabIndex={0}>
          <table className="table al-table al-tickers">
            <thead><tr>{th("ticker")}{th("count", "num")}<th scope="col">Alerts</th><th scope="col">First fired</th>{th("last")}{th("best", "num")}<th scope="col"><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.ticker}>
                  <td><Link href={`/stocks/${encodeURIComponent(s.ticker)}`} className="mono strong">{s.ticker}</Link></td>
                  <td className="num mono">{s.count}</td>
                  <td>{s.kinds.map((k) => <KindTag key={k} title={k} />)}</td>
                  <td className="cap">{s.first ? etTime(s.first) : ""}</td>
                  <td className="cap">{s.last ? etTime(s.last) : ""}</td>
                  <td className="num mono up">{s.best !== null ? s.best.toFixed(1) : ""}</td>
                  <td className="num"><button type="button" className="btn btn-sm" aria-label={`Show ${s.ticker} alerts`} onClick={() => onShow(s.ticker)}>Show alerts</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="cap">Counts are from the alerts loaded on this page. Use Load older on Activity to look further back.</p>
    </Card>
  );
}
