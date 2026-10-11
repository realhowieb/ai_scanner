"use client";

// Day Trader (Pro): live quotes, gap, VWAP, relative volume and the 0-100 DT score for
// a symbol source, from /v1/day-trader. Refreshes every 45 s while a session is trading.
// Rows sort by any column, filter by side, price, RVOL and volume, explain their score,
// and flag quotes that look wrong (split, bad print, stale prior close).
import { Fragment, useEffect, useMemo, useState } from "react";

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
export const DT_REFRESH_MS = 45_000;
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
        {r.day_trade_score == null ? "Too little evidence to score this setup."
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

function FilterBar({ f, set, shown, total }: { f: Filters; set: (f: Filters) => void; shown: number; total: number }) {
  const band = PRICE_BANDS.find((b) => b.min === f.minPrice && b.max === f.maxPrice)?.id ?? "any";
  const active = JSON.stringify(f) !== JSON.stringify(NO_FILTERS);
  return (
    <section className="filters" aria-label="Filters">
      <div className="chips" role="group" aria-label="Side">
        {(["all", "long", "short"] as const).map((s) => (
          <button key={s} type="button" className="chip" aria-pressed={f.side === s} onClick={() => set({ ...f, side: s })}>
            {s === "all" ? "All" : s === "long" ? "Long setups" : "Short setups"}
          </button>
        ))}
      </div>
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
      {active && <>
        <span className="cap">Showing {shown} of {total}</span>
        <button type="button" className="btn btn-sm" onClick={() => set(NO_FILTERS)}>Clear</button>
      </>}
    </section>
  );
}

function Live() {
  const [source, setSource] = useState<Source>("movers");
  const [tick, setTick] = useState(0);
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "score", desc: true });
  const [open, setOpen] = useState<string | null>(null);
  const narrow = useNarrow();
  const dt = useApi(`dt:${source}:${tick}`, (signal) => unwrap(api.GET("/v1/day-trader", { params: { query: { source } }, signal })));
  const trading = !!dt.data && dt.data.state !== "closed";
  useEffect(() => {
    if (!trading) return;
    const t = setInterval(() => setTick((n) => n + 1), DT_REFRESH_MS);
    return () => clearInterval(t);
  }, [trading]);
  const d = dt.data;
  const offHours = !!d && (d.state === "afterhours" || d.state === "closed");
  const all = useMemo(() => d?.rows ?? [], [d]);
  const rows = useMemo(() => sortRows(applyFilters(all, filters), sort.key, sort.desc), [all, filters, sort]);
  // Sparklines for the top names; 1-minute bars change slowly, so reload every third quote refresh.
  const sparkSyms = all.slice(0, SPARK_MAX).map((r) => r.ticker).join(",");
  const spark = useApi(sparkSyms ? `dt-spark:${sparkSyms}:${Math.floor(tick / 3)}` : null, (signal) =>
    unwrap(api.GET("/v1/day-trader/sparklines", { params: { query: { symbols: sparkSyms } }, signal })));
  const series = spark.data?.series ?? {};
  const sortBy = (key: SortKey) => setSort((s) => (s.key === key ? { key, desc: !s.desc } : { key, desc: key !== "ticker" }));
  const toggle = (t: string) => setOpen((o) => (o === t ? null : t));

  return (
    <>
      <section className="filters" aria-label="Source">
        <label className="inline-field">Symbols
          <select value={source} onChange={(e) => setSource(e.target.value as Source)}>
            {SOURCES.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
        </label>
        <span className="grow" />
        {d && <span className="cap">{STATE[d.state] ?? d.state} · quotes {etTime(d.as_of)}{trading ? " · refreshes every 45s" : ""}</span>}
        <button type="button" className="btn btn-sm" onClick={() => setTick((n) => n + 1)} disabled={dt.loading}>Refresh</button>
      </section>
      {d && all.length > 0 && (
        <div className="metric-strip" aria-label="Day Trader summary">
          <span><strong>{metric(d.summary?.strong, all.filter((r) => r.dt_quality === "strong").length)}</strong> strong</span>
          <span><strong>{metric(d.summary?.developing, all.filter((r) => r.dt_quality === "developing").length)}</strong> developing</span>
          <span><strong>{metric(d.summary?.flagged, all.filter((r) => (r.quote_flags ?? []).length).length)}</strong> flagged</span>
          <span><strong>{metric(d.summary?.best_long, "—")}</strong> best long</span>
          <span><strong>{metric(d.summary?.best_short, "—")}</strong> best short</span>
        </div>
      )}
      {d && all.length > 0 && <FilterBar f={filters} set={setFilters} shown={rows.length} total={all.length} />}
      {offHours && all.length > 0 && <p className="cap">Market is {d!.state === "afterhours" ? "in after hours" : "closed"}: Chg % and DT score use the regular session; AH shows the move since the close.</p>}
      {dt.error && !d ? <ErrorState error={dt.error} onRetry={dt.reload} what="Day Trader" />
        : !d ? <Skeleton rows={8} label="Loading live quotes" />
        : all.length === 0 ? <Card><Empty title={d.symbols.length ? "No live quotes for these symbols right now." : "No symbols in this source."}>
            {source === "watchlist" ? "Add tickers to your default watchlist, or pick another source." : "Try another source."}</Empty></Card>
        : rows.length === 0 ? <Card><Empty title="No rows match these filters."><button type="button" className="btn btn-sm" onClick={() => setFilters(NO_FILTERS)}>Clear filters</button></Empty></Card>
        : narrow ? (
          <section className={`dt-cards${dt.loading ? " dim" : ""}`} aria-label="Day Trader symbols">
            <div className="chips" role="group" aria-label="Sort by">
              {(["score", "chg", "rvol", "volume"] as const).map((k) => (
                <button key={k} type="button" className="chip" aria-pressed={sort.key === k} onClick={() => sortBy(k)}>
                  {COLS.find((c) => c.key === k)!.label}{sort.key === k ? (sort.desc ? " ↓" : " ↑") : ""}
                </button>
              ))}
            </div>
            {rows.map((r) => (
              <article key={r.ticker} className="card dt-card">
                <div className="dt-card-head">
                  <TickerLink ticker={r.ticker} /><Flags r={r} />
                  <span className="grow" />
                  <Score r={r} />
                </div>
                <div className="dt-card-row">
                  <span className="mono">{price(r.last)}</span>
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
          <section className={`card flush${dt.loading ? " dim" : ""}`}>
            <div className="table-wrap">
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
                  {rows.map((r) => (
                    <Fragment key={r.ticker}>
                      <tr className={flagged(r) ? "row-muted" : undefined}>
                        <td><span className="dt-ticker"><TickerLink ticker={r.ticker} /><Flags r={r} /></span></td>
                        <td className="num">
                          <button type="button" className="th-sort" aria-expanded={open === r.ticker} aria-label={`Why ${r.ticker} scores ${r.day_trade_score == null ? "no score" : Math.round(r.day_trade_score)}`}
                            onClick={() => toggle(r.ticker)}><Score r={r} /></button>
                        </td>
                        <td className="num mono">{price(r.last)}</td>
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
      {rows.length > 0 && <StairSteppers symbols={rows.map((r) => r.ticker)} />}
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

export function DayTraderView() {
  const { can } = useSession();
  return (
    <div className="stack">
      <section className="page-head">
        <div>
          <h1 className="h1">Day Trader</h1>
          <p className="cap">Live intraday quotes ranked by day-trade score.</p>
        </div>
      </section>
      {can("can_day_trader") ? <Live /> : <Card><Locked title="Day Trader is part of Pro" plan="pro">Live quotes, VWAP and relative volume for movers or your watchlist, refreshed through the session.</Locked></Card>}
    </div>
  );
}
