"use client";

// Day Trader (Pro): live quotes, gap, VWAP, relative volume and the 0-100 DT score for
// a symbol source, from /v1/day-trader. Refreshes every 15 s while a session is trading (not
// while the tab is hidden); rows slide to their new DT rank, with rank moves and changed
// prices/scores highlighted, and a pause keeps the order still while you work.
// Rows sort by any column, filter by side, price, RVOL and volume, explain their score,
// and flag quotes that look wrong (split, bad print, stale prior close).
import { Fragment, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { Card, Empty, ErrorState, Locked, Pill, Skeleton, TickerLink } from "@/components/ui";
import { PriceAlertButton } from "@/features/PriceAlert";
import { SaveToWatchlistButton } from "@/features/SaveToWatchlist";
import { StairSteppers } from "@/features/StairSteppers";
import { useApi } from "@/hooks/useApi";
import { compact, etTime, num, pct, price } from "@/lib/format";
import { useSession } from "@/session/SessionProvider";

type Row = Schemas["DayTraderRow"];

const SOURCES = [
  { id: "watchlist", label: "My default watchlist" },
  { id: "movers", label: "Top movers" },
  { id: "movers_sp500", label: "S&P 500 movers" },
  { id: "movers_nasdaq", label: "NASDAQ movers" },
  { id: "premarket", label: "Pre-market scan" },
  { id: "postmarket", label: "After-hours scan" },
  { id: "scan_picks", label: "Latest scan picks" },
  { id: "megacaps", label: "Mega caps" },
] as const;
type Source = (typeof SOURCES)[number]["id"];
const STATE: Record<string, string> = { premarket: "Pre-market", open: "Market open", afterhours: "After hours", closed: "Market closed" };
const metric = (v: unknown, fallback: string | number = 0) => (typeof v === "number" || typeof v === "string" ? v : fallback);
export const DT_REFRESH_MS = 15_000;
/** After a failed refresh, wait this long before trying again (no tight retry loop). */
export const DT_RETRY_MS = 30_000;
const SPARK_EVERY = 8;   // sparklines reload every 8th quote refresh (~2 min; 1-minute bars change slowly)
const SPARK_MAX = 40;
const tone = (v: number | null | undefined) => (v == null ? "" : v > 0 ? "up" : v < 0 ? "down" : "");

const QUALITY: Record<string, { label: string; tone?: "up" | "warn" }> = {
  strong: { label: "Strong", tone: "up" }, developing: { label: "Developing" }, weak: { label: "Weak", tone: "warn" },
};
export const FLAG_HELP: Record<string, string> = {
  "Extreme move": "Moved 100% or more. Often a split, a bad print or a stale previous close.",
  "Far from VWAP": "The last trade is 40% or more away from the day's VWAP, so one print may be off.",
  "Stale quote": "No trade in several days.",
};

type SortKey = "score" | "ticker" | "last" | "chg" | "gap" | "vwap" | "rvol" | "volume";
const COLS: { key: SortKey; label: string; narrow?: boolean; get: (r: Row) => number | string | null | undefined }[] = [
  { key: "ticker", label: "Ticker", get: (r) => r.ticker },
  { key: "score", label: "DT score", get: (r) => r.day_trade_score },
  { key: "last", label: "Last", get: (r) => r.last },
  { key: "chg", label: "Chg %", get: (r) => r.session_chg_pct ?? r.chg_pct },
  { key: "gap", label: "Gap %", narrow: true, get: (r) => r.gap_pct },
  { key: "vwap", label: "vs VWAP", narrow: true, get: (r) => r.vs_vwap_pct },
  { key: "rvol", label: "RVOL", get: (r) => r.rvol },
  { key: "volume", label: "Volume", narrow: true, get: (r) => r.volume },
];

type Side = "all" | "long" | "short";
export type Filters = { side: Side; minPrice: number; maxPrice: number; minRvol: number; minVolume: number; hideFlagged: boolean };
export const NO_FILTERS: Filters = { side: "all", minPrice: 0, maxPrice: 0, minRvol: 0, minVolume: 0, hideFlagged: false };
const PRICE_BANDS = [
  { id: "any", label: "Any price", min: 0, max: 0 }, { id: "u5", label: "Under $5", min: 0, max: 5 },
  { id: "5-20", label: "$5 – $20", min: 5, max: 20 }, { id: "20-100", label: "$20 – $100", min: 20, max: 100 },
  { id: "o100", label: "Over $100", min: 100, max: 0 },
];
const RVOL_MIN = [0, 1, 1.5, 2, 3];
const VOLUME_MIN = [0, 100_000, 500_000, 1_000_000];

const flagged = (r: Row) => (r.quote_flags?.length ?? 0) > 0;

export function applyFilters(rows: Row[], f: Filters): Row[] {
  return rows.filter((r) => {
    if (f.side === "long" && r.dt_direction !== "bullish") return false;
    if (f.side === "short" && r.dt_direction !== "bearish") return false;
    if (f.minPrice && (r.last ?? 0) < f.minPrice) return false;
    if (f.maxPrice && (r.last ?? Infinity) >= f.maxPrice) return false;
    if (f.minRvol && (r.rvol ?? 0) < f.minRvol) return false;
    if (f.minVolume && (r.volume ?? 0) < f.minVolume) return false;
    if (f.hideFlagged && flagged(r)) return false;
    return true;
  });
}

/** Sorted copy. The default (score, descending) puts flagged quotes and unscored rows last. */
export function sortRows(rows: Row[], key: SortKey, desc: boolean): Row[] {
  if (key === "score" && desc) {
    return [...rows].sort((a, b) => Number(flagged(a)) - Number(flagged(b))
      || Number(a.day_trade_score == null) - Number(b.day_trade_score == null)
      || (b.day_trade_score ?? 0) - (a.day_trade_score ?? 0));
  }
  const col = COLS.find((c) => c.key === key)!;
  return [...rows].sort((a, b) => {
    const x = col.get(a), y = col.get(b);
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    const c = typeof x === "string" ? x.localeCompare(String(y)) : x - (y as number);
    return desc ? -c : c;
  });
}

function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(max-width: 760px)");
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  return narrow;
}

export function Sparkline({ points, ticker }: { points: number[] | undefined; ticker: string }) {
  if (!points || points.length < 2) return <span className="cap" aria-hidden="true">—</span>;
  const w = 84, h = 24;
  const lo = Math.min(...points), hi = Math.max(...points);
  const span = hi - lo || 1;
  const d = points.map((p, i) => `${i ? "L" : "M"}${((i / (points.length - 1)) * w).toFixed(1)},${(h - 2 - ((p - lo) / span) * (h - 4)).toFixed(1)}`).join("");
  const up = points[points.length - 1]! >= points[0]!;
  return (
    <svg className={`spark ${up ? "up" : "down"}`} width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img"
      aria-label={`${ticker} today: ${up ? "up" : "down"} from ${price(points[0])} to ${price(points[points.length - 1])}`}>
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function Score({ r }: { r: Row }) {
  if (r.traded_today === false) return <span className="cap" title="No trade yet today. The score starts with today's first trade.">Waiting</span>;
  if (r.day_trade_score == null) return <span className="cap" title="Too little evidence to score">—</span>;
  // A flagged quote's score rests on numbers we already say not to trust: show it muted, with no tier.
  if (flagged(r)) {
    return <span className="dt-score"><span className="mono cap" title="Check the quote before relying on this score">{Math.round(r.day_trade_score)}</span></span>;
  }
  const q = QUALITY[r.dt_quality];
  return (
    <span className="dt-score">
      <span className="mono strong">{Math.round(r.day_trade_score)}</span>
      {q && <Pill tone={q.tone}>{q.label}</Pill>}
    </span>
  );
}

function Flags({ r }: { r: Row }) {
  const flags = r.quote_flags ?? [];
  if (!flags.length) return null;
  return <span title={flags.map((f) => FLAG_HELP[f] ?? f).join(" ")}><Pill tone="warn">Check quote</Pill></span>;
}

function Change({ r, offHours }: { r: Row; offHours: boolean }) {
  if (!offHours || r.session_chg_pct == null) return <span className={tone(r.chg_pct)}>{pct(r.chg_pct)}</span>;
  return (
    <span className="dt-chg">
      <span className={tone(r.session_chg_pct)}>{pct(r.session_chg_pct)}</span>
      {r.ext_chg_pct != null && <span className={`cap ${tone(r.ext_chg_pct)}`}>AH {pct(r.ext_chg_pct)}</span>}
    </span>
  );
}

function Why({ r }: { r: Row }) {
  const reasons = r.dt_reasons ?? [];
  const conflicts = r.dt_conflicts ?? [];
  const flags = r.quote_flags ?? [];
  return (
    <div className="dt-why">
      <p className="cap">
        {r.traded_today === false ? "No trade yet today. The score starts with today's first trade, so yesterday's session can't rank this name."
          : r.day_trade_score == null ? "Too little evidence to score this setup."
          : flagged(r) ? "No tier while the quote is flagged: check it before relying on this score."
          : `${QUALITY[r.dt_quality]?.label ?? "Unrated"} ${r.dt_direction} setup.`}
      </p>
      {reasons.length > 0 && <div className="chips">{reasons.map((x) => <Pill key={x}>{x}</Pill>)}</div>}
      {conflicts.length > 0 && <div className="chips">{conflicts.map((x) => <Pill key={x} tone="warn">{x}</Pill>)}</div>}
      {flags.map((f) => <p key={f} className="cap"><Pill tone="warn">{f}</Pill> {FLAG_HELP[f]}</p>)}
    </div>
  );
}

function Actions({ r }: { r: Row }) {
  return (
    <span className="dt-actions">
      <SaveToWatchlistButton ticker={r.ticker} compact />
      <PriceAlertButton ticker={r.ticker} at={r.last} compact label={`Set a price alert for ${r.ticker}`}
        note={r.last != null ? `Last trade ${price(r.last)}.` : undefined} />
    </span>
  );
}

/** On a phone the side chips stay in view and the rest folds into "More filters", so the
 * quotes start on the first screen instead of below five controls. */
function FilterBar({ f, set, shown, total, narrow }: { f: Filters; set: (f: Filters) => void; shown: number; total: number; narrow: boolean }) {
  const band = PRICE_BANDS.find((b) => b.min === f.minPrice && b.max === f.maxPrice)?.id ?? "any";
  const active = JSON.stringify(f) !== JSON.stringify(NO_FILTERS);
  const more = f.minPrice !== 0 || f.maxPrice !== 0 || f.minRvol !== 0 || f.minVolume !== 0 || f.hideFlagged;
  const sides = (
    <div className="chips" role="group" aria-label="Side">
      {(["all", "long", "short"] as const).map((s) => (
        <button key={s} type="button" className="chip" aria-pressed={f.side === s} onClick={() => set({ ...f, side: s })}>
          {s === "all" ? "All" : s === "long" ? "Long setups" : "Short setups"}
        </button>
      ))}
    </div>
  );
  const fields = (
    <>
      <label className="inline-field">Price
        <select value={band} onChange={(e) => { const b = PRICE_BANDS.find((x) => x.id === e.target.value)!; set({ ...f, minPrice: b.min, maxPrice: b.max }); }}>
          {PRICE_BANDS.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
        </select>
      </label>
      <label className="inline-field">Min RVOL
        <select value={f.minRvol} onChange={(e) => set({ ...f, minRvol: Number(e.target.value) })}>
          {RVOL_MIN.map((v) => <option key={v} value={v}>{v ? `${v}×` : "Any"}</option>)}
        </select>
      </label>
      <label className="inline-field">Min volume
        <select value={f.minVolume} onChange={(e) => set({ ...f, minVolume: Number(e.target.value) })}>
          {VOLUME_MIN.map((v) => <option key={v} value={v}>{v ? compact(v) : "Any"}</option>)}
        </select>
      </label>
      <label className="inline-check"><input type="checkbox" checked={f.hideFlagged} onChange={(e) => set({ ...f, hideFlagged: e.target.checked })} /> Hide flagged quotes</label>
    </>
  );
  const status = active && <>
    <span className="cap">Showing {shown} of {total}</span>
    <button type="button" className="btn btn-sm" onClick={() => set(NO_FILTERS)}>Clear</button>
  </>;
  if (narrow) {
    return (
      <section className="stack-sm" aria-label="Filters">
        <div className="filters">{sides}{status}</div>
        <details className="filter-panel">
          <summary><span className="strong">More filters</span>{more && <span className="cap">Filters active</span>}</summary>
          <div className="filters">{fields}</div>
        </details>
      </section>
    );
  }
  return (
    <section className="filters" aria-label="Filters">
      {sides}
      {fields}
      {status}
    </section>
  );
}

type Data = Schemas["DayTrader"];
type Dir = -1 | 0 | 1;
export type Move = { rank: number; last: Dir; score: Dir };
const sign = (a: number | null | undefined, b: number | null | undefined): Dir =>
  a == null || b == null || a === b ? 0 : a > b ? 1 : -1;

/** What changed between two refreshes of the same source: DT-rank moves (positive = moved up)
 * and the direction of each changed price and score. Empty across a source switch. */
export function diffRefresh(prev: Data | null, cur: Data | null): Record<string, Move> {
  if (!prev || !cur || prev.source !== cur.source || prev.as_of === cur.as_of) return {};
  const rankOf = (d: Data) => new Map(sortRows(d.rows, "score", true).map((r, i) => [r.ticker, i]));
  const before = rankOf(prev), after = rankOf(cur);
  const old = new Map(prev.rows.map((r) => [r.ticker, r]));
  const out: Record<string, Move> = {};
  for (const r of cur.rows) {
    const o = old.get(r.ticker);
    if (!o) continue;
    const m: Move = { rank: (before.get(r.ticker) ?? 0) - (after.get(r.ticker) ?? 0), last: sign(r.last, o.last),
      score: sign(r.day_trade_score, o.day_trade_score) };
    if (m.rank || m.last || m.score) out[r.ticker] = m;
  }
  return out;
}

/** Keep a frozen order: known tickers where they were, new ones at the end. */
export function holdOrder(rows: Row[], order: string[]): Row[] {
  const at = new Map(order.map((t, i) => [t, i]));
  return [...rows].sort((a, b) => (at.get(a.ticker) ?? Infinity) - (at.get(b.ticker) ?? Infinity));
}

function useVisible(): boolean {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    if (typeof document === "undefined") return;
    const on = () => setVisible(document.visibilityState !== "hidden");
    on();
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);
  return visible;
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [active]);
  return now;
}

/** Slide rows from where they were to where they are now (FLIP), keyed by data-t. */
function useSlide(container: React.RefObject<HTMLElement | null>, order: string) {
  const last = useRef<Map<string, number>>(new Map());
  useLayoutEffect(() => {
    const box = container.current;
    if (!box) return;
    const els = Array.from(box.querySelectorAll<HTMLElement>("[data-t]"));
    const now = new Map(els.map((el) => [el.dataset.t!, el.getBoundingClientRect().top]));
    const still = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!still) {
      for (const el of els) {
        const before = last.current.get(el.dataset.t!);
        const dy = before == null ? 0 : before - now.get(el.dataset.t!)!;
        if (dy && typeof el.animate === "function") {
          el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 450, easing: "cubic-bezier(.2,.7,.2,1)" });
        }
      }
    }
    last.current = now;
  }, [container, order]);
}

/** Re-mounts on each new value so the highlight animation replays. */
function Flash({ v, dir, children }: { v: unknown; dir: Dir | undefined; children: React.ReactNode }) {
  return <span key={`${String(v)}`} className={dir ? (dir > 0 ? "flash-up" : "flash-down") : undefined}>{children}</span>;
}

function RankMove({ n }: { n: number | undefined }) {
  if (!n) return null;
  return <span className={`dt-move ${n > 0 ? "up" : "down"}`} aria-label={`${n > 0 ? "up" : "down"} ${Math.abs(n)} place${Math.abs(n) === 1 ? "" : "s"}`}>
    {n > 0 ? "▲" : "▼"}{Math.abs(n)}</span>;
}

function Live() {
  const [source, setSource] = useState<Source>("movers");
  const [tick, setTick] = useState(0);
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "score", desc: true });
  const [open, setOpen] = useState<string | null>(null);
  const [held, setHeld] = useState<string[] | null>(null);
  const narrow = useNarrow();
  const visible = useVisible();
  // `at` is when this device received the answer. Timing uses it rather than the server's
  // as_of, so a phone whose clock runs ahead of the server doesn't refresh nonstop.
  const dt = useApi(`dt:${source}`, (signal) => unwrap(api.GET("/v1/day-trader", { params: { query: { source } }, signal }))
    .then((data) => ({ data, at: Date.now() })), tick);
  const d = dt.data?.data ?? null;
  const gotAt = dt.data?.at ?? 0;
  const failed = !!dt.error && !!d;
  const trading = !!d && d.state !== "closed";
  const live = trading && visible;
  // Next refresh one cycle after the last answer (at once when returning to a tab whose
  // quotes are already older than that); nothing is scheduled while a request is in flight.
  // A failed refresh keeps the last quotes on screen and retries after DT_RETRY_MS.
  useEffect(() => {
    if (!live || dt.loading || dt.refreshing) return;
    const wait = failed ? DT_RETRY_MS : Math.min(DT_REFRESH_MS, Math.max(0, DT_REFRESH_MS - (Date.now() - gotAt)));
    const t = setTimeout(() => setTick((n) => n + 1), wait);
    return () => clearTimeout(t);
  }, [live, dt.loading, dt.refreshing, gotAt, failed]);
  const now = useNow(live);
  // The previous refresh, kept alongside the current one to show what moved.
  const [hist, setHist] = useState<{ cur: Data | null; prev: Data | null }>({ cur: null, prev: null });
  if (d !== hist.cur) setHist({ cur: d, prev: hist.cur });
  const moves = useMemo(() => diffRefresh(hist.prev, hist.cur), [hist]);
  const switching = dt.loading && !!d && d.source !== source;
  const offHours = !!d && (d.state === "afterhours" || d.state === "closed");
  const all = useMemo(() => d?.rows ?? [], [d]);
  const rows = useMemo(() => {
    const sorted = sortRows(applyFilters(all, filters), sort.key, sort.desc);
    return held ? holdOrder(sorted, held) : sorted;
  }, [all, filters, sort, held]);
  const [showAll, setShowAll] = useState(false);
  const shownRows = showAll ? rows : rows.slice(0, 15);
  const candidate = (side: "long" | "short", value: unknown) => {
    const row = all.find((r) => r.ticker === value);
    const hasStrong = all.some((r) => r.dt_direction === (side === "long" ? "bullish" : "bearish") && r.dt_quality === "strong" && !flagged(r));
    if (!row) return <span>{hasStrong ? `See ${side} setups for candidates` : `No strong ${side} setup`}</span>;
    const quality = row.dt_quality;
    const strong = quality === "strong" && !flagged(row);
    return <span className="dt-candidate">{!strong && !hasStrong && <>No strong {side} setup · </>}<strong>{row.ticker}</strong> {strong ? `top ${side} candidate` : `highest-ranked ${side}`} · DT {row.day_trade_score == null ? "—" : Math.round(row.day_trade_score)} · {quality ?? "unrated"}</span>;
  };
  const listRef = useRef<HTMLElement>(null);
  useSlide(listRef, rows.map((r) => r.ticker).join(","));
  // Sparklines for the top names; 1-minute bars change slowly, so reload every third quote refresh.
  const sparkSyms = all.slice(0, SPARK_MAX).map((r) => r.ticker).join(",");
  const spark = useApi(sparkSyms ? `dt-spark:${sparkSyms}` : null, (signal) =>
    unwrap(api.GET("/v1/day-trader/sparklines", { params: { query: { symbols: sparkSyms } }, signal })), Math.floor(tick / SPARK_EVERY));
  const series = spark.data?.series ?? {};
  const sortBy = (key: SortKey) => {
    setHeld(null);
    setSort((s) => (s.key === key ? { key, desc: !s.desc } : { key, desc: key !== "ticker" }));
  };
  const togglePause = () => setHeld((h) => (h ? null : rows.map((r) => r.ticker)));
  const ago = gotAt ? Math.max(0, Math.round((now - gotAt) / 1000)) : 0;
  const toggle = (t: string) => setOpen((o) => (o === t ? null : t));

  return (
    <>
      <DayTraderHeading state={d?.state} />
      <section className="filters" aria-label="Source">
        <label className="inline-field">Symbols
          <select value={source} onChange={(e) => setSource(e.target.value as Source)}>
            {SOURCES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
        </label>
        {rows.length > 0 && <a href="#stair-steppers" className="cap">Jump to Stair-Steppers</a>}
        <span className="grow" />
        {d && (live
          ? failed
            ? <span className="cap dt-live dt-stale" role="status"><span className="dt-dot" aria-hidden="true" />{STATE[d.state] ?? d.state} · Couldn&apos;t refresh, retrying · checked {etTime(d.as_of)}</span>
            : <span className="cap dt-live" role="status"><span className="dt-dot" aria-hidden="true" />{STATE[d.state] ?? d.state} · Live · checked {ago}s ago</span>
          : <span className="cap">{STATE[d.state] ?? d.state} · checked {etTime(d.as_of)}{trading ? " · paused while this tab is hidden" : ""}</span>)}
        {trading && <button type="button" className="btn btn-sm" aria-pressed={!!held} onClick={togglePause}
          title="Keep rows in place while you work; prices still update">{held ? "Resume order" : "Pause order"}</button>}
        <button type="button" className="btn btn-sm" onClick={() => setTick((n) => n + 1)} disabled={dt.loading}>Refresh</button>
      </section>
      {d && all.length > 0 && (
        <div className="metric-strip" aria-label="Day Trader summary">
          <span><strong>{metric(d.summary?.strong, all.filter((r) => r.dt_quality === "strong").length)}</strong> strong</span>
          <span><strong>{metric(d.summary?.developing, all.filter((r) => r.dt_quality === "developing").length)}</strong> developing</span>
          <span><strong>{metric(d.summary?.flagged, all.filter((r) => (r.quote_flags ?? []).length).length)}</strong> flagged</span>
          {candidate("long", d.summary?.best_long)}
          {candidate("short", d.summary?.best_short)}
        </div>
      )}
      {d && all.length > 0 && <FilterBar f={filters} set={setFilters} shown={rows.length} total={all.length} narrow={narrow} />}
      {d?.state === "premarket" && all.length > 0 && <p className="cap">Pre-market: Chg % is the latest trade today vs yesterday&apos;s close, and Volume and VWAP count pre-market trading only.{source.startsWith("movers") ? " Movers lists today's pre-market gappers until the open." : ""}</p>}
      {offHours && all.length > 0 && <p className="cap">Market is {d!.state === "afterhours" ? "in after hours" : "closed"}: last prices can be from an earlier session. Checked time is the request time, not the last trade time. The API does not supply the session date or individual trade timestamps; they cannot be verified here. Chg % and DT score use the regular session; AH shows the move since the close. Candidate highlights are relative rankings, not minimum-quality guarantees.</p>}
      {dt.error && !d ? <ErrorState error={dt.error} onRetry={dt.reload} what="Day Trader" />
        : !d ? <Skeleton rows={8} label="Loading live quotes" />
        : all.length === 0 && d.state === "premarket" && source.startsWith("movers") && d.symbols.length === 0
          ? <Card><Empty title="No pre-market gappers yet today.">Top movers lists stocks trading before the open. Our market data usually shows pre-market trades from about 8:00 AM ET, so check back then, or pick My default watchlist to see last night&apos;s prices.</Empty></Card>
        : all.length === 0 ? <Card><Empty title={d.symbols.length ? "No live quotes for these symbols right now." : "No symbols in this source."}>
            {source === "watchlist" ? "Add tickers to your default watchlist, or pick another source." : "Try another source."}</Empty></Card>
        : rows.length === 0 ? <Card><Empty title="No rows match these filters."><button type="button" className="btn btn-sm" onClick={() => setFilters(NO_FILTERS)}>Clear filters</button></Empty></Card>
        : narrow ? (
          <section ref={listRef} className={`dt-cards${switching ? " dim" : ""}`} aria-label="Day Trader symbols">
            <div className="chips" role="group" aria-label="Sort by">
              {(["score", "chg", "rvol", "volume"] as const).map((k) => (
                <button key={k} type="button" className="chip" aria-pressed={sort.key === k} onClick={() => sortBy(k)}>
                  {COLS.find((c) => c.key === k)!.label}{sort.key === k ? (sort.desc ? " ↓" : " ↑") : ""}
                </button>
              ))}
            </div>
            {shownRows.map((r) => (
              <article key={r.ticker} data-t={r.ticker} className={`card dt-card${flagged(r) ? " row-muted" : ""}`}>
                <div className="dt-card-head">
                  <TickerLink ticker={r.ticker} /><RankMove n={moves[r.ticker]?.rank} /><Flags r={r} />
                  <span className="grow" />
                  <Flash v={r.day_trade_score} dir={moves[r.ticker]?.score}><Score r={r} /></Flash>
                </div>
                <div className="dt-card-row">
                  <span className="mono"><Flash v={r.last} dir={moves[r.ticker]?.last}>{price(r.last)}</Flash></span>
                  <span className="mono"><Change r={r} offHours={offHours} /></span>
                  <Sparkline points={series[r.ticker]} ticker={r.ticker} />
                </div>
                <dl className="dt-card-stats">
                  <div><dt>Gap</dt><dd className="mono">{pct(r.gap_pct)}</dd></div>
                  <div><dt>vs VWAP</dt><dd className={`mono ${tone(r.vs_vwap_pct)}`}>{pct(r.vs_vwap_pct)}</dd></div>
                  <div><dt>RVOL</dt><dd className="mono">{num(r.rvol)}</dd></div>
                  <div><dt>Volume</dt><dd className="mono">{compact(r.volume)}</dd></div>
                </dl>
                <div className="dt-card-foot">
                  <button type="button" className="btn btn-sm" aria-expanded={open === r.ticker} onClick={() => toggle(r.ticker)}>Why this score</button>
                  <span className="grow" />
                  <Actions r={r} />
                </div>
                {open === r.ticker && <Why r={r} />}
              </article>
            ))}
          </section>
        ) : (
          <section ref={listRef} className={`card flush${switching ? " dim" : ""}`}>
            <div className="table-wrap dt-scroll" tabIndex={0} role="region" aria-label="Day Trader candidates; scroll for more rows and columns">
              <table className="table dt-table">
                <thead><tr>
                  {COLS.map((c) => (
                    <th key={c.key} scope="col" className={`${c.key === "ticker" ? "" : "num"}${c.narrow ? " hide-narrow" : ""}`}
                      aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"}>
                      <button type="button" className="th-sort" onClick={() => sortBy(c.key)}>
                        {c.label}{sort.key === c.key ? (sort.desc ? " ↓" : " ↑") : ""}
                      </button>
                    </th>
                  ))}
                  <th scope="col" className="hide-narrow">Today</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr></thead>
                <tbody>
                  {shownRows.map((r) => (
                    <Fragment key={r.ticker}>
                      <tr data-t={r.ticker} className={flagged(r) ? "row-muted" : undefined}>
                        <td><span className="dt-ticker"><TickerLink ticker={r.ticker} /><RankMove n={moves[r.ticker]?.rank} /><Flags r={r} /></span></td>
                        <td className="num">
                          <button type="button" className="th-sort" aria-expanded={open === r.ticker} aria-label={`Why ${r.ticker} scores ${r.day_trade_score == null ? "no score" : Math.round(r.day_trade_score)}`}
                            onClick={() => toggle(r.ticker)}><Flash v={r.day_trade_score} dir={moves[r.ticker]?.score}><Score r={r} /></Flash></button>
                        </td>
                        <td className="num mono"><Flash v={r.last} dir={moves[r.ticker]?.last}>{price(r.last)}</Flash></td>
                        <td className="num mono"><Change r={r} offHours={offHours} /></td>
                        <td className="num mono hide-narrow">{pct(r.gap_pct)}</td>
                        <td className={`num mono hide-narrow ${tone(r.vs_vwap_pct)}`}>{pct(r.vs_vwap_pct)}</td>
                        <td className="num mono">{num(r.rvol)}</td>
                        <td className="num mono hide-narrow">{compact(r.volume)}</td>
                        <td className="hide-narrow"><Sparkline points={series[r.ticker]} ticker={r.ticker} /></td>
                        <td><Actions r={r} /></td>
                      </tr>
                      {open === r.ticker && <tr className="dt-why-row"><td colSpan={COLS.length + 2}><Why r={r} /></td></tr>}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      {d && d.missing > 0 && <p className="cap"><Pill tone="warn">{d.missing} missing</Pill> {d.missing} symbol{d.missing === 1 ? " has" : "s have"} no live quote right now.</p>}
      {rows.length > 15 && <div><button type="button" className="btn" onClick={() => setShowAll((v) => !v)}>{showAll ? "Show first 15" : `Show all ${rows.length} candidates`}</button></div>}
      {rows.length > 0 && <section id="stair-steppers"><StairSteppers symbols={rows.map((r) => r.ticker)} /></section>}
      <details className="card dt-help">
        <summary className="strong">How the DT score works</summary>
        <div className="stack-sm">
          <p className="cap">The DT score (0–100) rates how strong and consistent an intraday setup is, long or short. It is separate from the HSF Score and is not a price prediction.</p>
          <ul className="bullets">
            <li>Most of the weight goes to how many signals agree on direction: VWAP side, SuperTrend, EWO, the day&apos;s move and a gap that supports it.</li>
            <li>Trend strength (ADX) and relative volume confirm the setup. Extremes are capped, so a 500% move can&apos;t outscore everything else.</li>
            <li>Conflicts such as a fading gap or losing VWAP take points off. <strong>Strong</strong> needs a high score, broad agreement and confirmation.</li>
            <li>Quotes that look wrong (a 100%+ move, far from VWAP, or no recent trade) get a <em>Check quote</em> flag, show no tier and rank last.</li>
          </ul>
          <p className="cap">Click a score to see the signals behind it. Not financial advice.</p>
        </div>
      </details>
    </>
  );
}

function DayTraderHeading({ state }: { state?: Data["state"] }) {
  return <section className="page-head"><div><h1 className="h1">Day Trader</h1>
    <p className="cap">{state === "closed" ? "Last-session intraday rankings · market closed."
      : state === "afterhours" ? "Completed-session rankings with after-hours prices."
      : state === "premarket" ? "Pre-market quotes ranked by day-trade score."
      : state === "open" ? "Live intraday quotes ranked by day-trade score."
      : "Intraday quotes and rankings for the current market session."}</p>
  </div></section>;
}

export function DayTraderView() {
  const { can } = useSession();
  return <div className="stack">{can("can_day_trader") ? <Live /> : <>
    <DayTraderHeading />
    <Card><Locked title="Day Trader is part of Pro" plan="pro">Live quotes, VWAP and relative volume for movers or your watchlist, refreshed through the session.</Locked></Card>
  </>}</div>;
}
