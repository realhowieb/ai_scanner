"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Schemas } from "@/api/client";
import { compact, etDate, pct, price, shortDate } from "@/lib/format";

import { chartBars, emaCrosses } from "@/lib/chartBars";
import { InteractivePriceChart } from "./InteractivePriceChart";
export { emaCrosses } from "@/lib/chartBars";

type Bar = Schemas["Bar"];
const NO_EVENTS: ChartEvent[] = [];
/** Something that happened on a session, marked on the chart (an HSF signal firing). */
export type ChartEvent = { date: string; label: string };

const DEFAULT_W = 720; // drawing width until the figure is measured; then 1 unit = 1 CSS px
const NARROW_W = 520; // below this the chart opens on 3M so candles stay readable
const H = 330;
const VOL_H = 56;
const AXIS_H = 20;
const PAD = { l: 8, r: 60, t: 12 };
const GAP = 8;
const RANGES = [["1M", 21], ["3M", 63], ["6M", 126], ["1Y", 252]] as const;
type RangeKey = (typeof RANGES)[number][0];
const OVERLAYS = [
  { key: "ema", label: "EMA 9/21" },
  { key: "sma20", label: "SMA 20" },
  { key: "sma50", label: "SMA 50" },
] as const;
type OverlayKey = (typeof OVERLAYS)[number]["key"];
const MAX_EARNINGS_SLOTS = 10;

/** Round price levels (1, 2, 2.5 or 5 x 10^k apart) covering [min, max]. */
export function niceTicks(min: number, max: number, count = 5): { ticks: number[]; decimals: number } {
  const span = max - min || Math.abs(max) || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((m) => m * mag >= raw) ?? 10) * mag;
  let decimals = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
  if (!Number.isInteger(Number((step * 10 ** decimals).toFixed(6)))) decimals += 1;
  const ticks: number[] = [];
  for (let t = Math.ceil(min / step - 1e-9) * step; t <= max + 1e-9; t += step) ticks.push(Number(t.toFixed(decimals + 2)));
  return { ticks, decimals };
}

/** SVG path through the defined points, broken where a value is missing. */
function linePath(vals: (number | null | undefined)[], x: (i: number) => number, y: (v: number) => number): string {
  let d = "", pen = false;
  vals.forEach((v, i) => {
    if (v == null || !Number.isFinite(v)) { pen = false; return; }
    d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}

const day = (iso: string) => new Date(`${iso}T12:00:00Z`);
const dayLabel = (iso: string) => day(iso).toLocaleDateString("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" });

/** Weekday sessions after `last` up to and including `target` (holidays not known). */
function sessionsBetween(last: string, target: string): number {
  let n = 0;
  const d = day(last), end = day(target);
  while (d < end) {
    d.setUTCDate(d.getUTCDate() + 1);
    if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) n++;
  }
  return n;
}

/** Daily candles plus volume (completed sessions only): EMA 9/21 with their golden and
 * death crosses, optional SMA 20/50, HSF signal days, the next earnings date, a range
 * picker and a crosshair readout. Not a live chart. */
export function PriceChart({ bars, events = NO_EVENTS, earningsDate, volumeFeed }: {
  bars: Bar[];
  /** "iex": volume counts IEX exchange trades only, a small share of all U.S. volume. */
  volumeFeed?: string | null;
  events?: ChartEvent[];
  /** Next earnings date (YYYY-MM-DD), drawn past the last bar when it's close. */
  earningsDate?: string | null;
}) {
  const all = useMemo(() => chartBars(bars), [bars]);
  const ranges = RANGES.filter((_r, k) => k === 0 || all.length > RANGES[k - 1]![1]);
  const [rangePick, setRange] = useState<RangeKey | null>(null);
  const [on, setOn] = useState<Record<OverlayKey, boolean>>({ ema: true, sma20: false, sma50: false });
  const [hover, setHover] = useState<number | null>(null);
  const [W, setW] = useState(DEFAULT_W);
  const fig = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = fig.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => { const w = Math.round(e?.contentRect.width ?? 0); if (w > 0) setW(Math.max(300, w)); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fallback = W < NARROW_W ? "3M" : "6M";
  const range = ranges.find(([k]) => k === rangePick) ?? ranges.find(([k]) => k === fallback) ?? ranges[ranges.length - 1]!;
  const off = Math.max(0, all.length - range[1]);
  const data = useMemo(() => all.slice(off), [all, off]);
  const [interactive, setInteractive] = useState(false);
  const hoverDate = useCallback((date: string | null) => {
    const i = date ? data.findIndex((bar) => bar.date === date) : -1;
    setHover(i >= 0 ? i : null);
  }, [data]);
  if (all.length < 2) return null;
  const last = data[data.length - 1]!;
  const first = data[0]!;

  const earnSlots = earningsDate && earningsDate > last.date ? sessionsBetween(last.date, earningsDate) : 0;
  const showEarn = earnSlots > 0 && earnSlots <= MAX_EARNINGS_SLOTS;
  const slots = data.length + (showEarn ? earnSlots : 0);

  const lineVals = (b: Bar) => [on.ema ? b.ema9 : null, on.ema ? b.ema21 : null, on.sma20 ? b.sma20 : null, on.sma50 ? b.sma50 : null];
  const vals = data.flatMap((b) => [b.high ?? b.close, b.low ?? b.close, ...lineVals(b)]).filter((v): v is number => v != null && Number.isFinite(v));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo || hi || 1) * 0.04;
  const min = lo - pad, max = hi + pad, span = max - min;
  const { ticks, decimals } = niceTicks(min, max, 5);

  const vmax = Math.max(1, ...data.map((b) => b.volume ?? 0));
  const plotBottom = H - AXIS_H - VOL_H - GAP;
  const plotH = plotBottom - PAD.t;
  const step = (W - PAD.l - PAD.r) / slots;
  const x = (i: number) => PAD.l + i * step + step / 2;
  const y = (v: number) => PAD.t + (1 - (v - min) / span) * plotH;
  const bw = Math.max(1, step * 0.6);
  const volBase = H - AXIS_H;

  const crosses = on.ema ? emaCrosses(all).filter((c) => c.i >= off + 1).map((c) => ({ ...c, i: c.i - off })) : [];
  const hasEma = all.some((b) => b.ema9 != null);
  const allCrosses = emaCrosses(all);
  const lastGolden = [...allCrosses].reverse().find((c) => c.kind === "golden");
  const byDay = new Map<string, string[]>();
  for (const e of events) byDay.set(e.date, [...(byDay.get(e.date) ?? []), e.label]);
  const crossOn = new Map(crosses.map((c) => [c.i, c.kind]));

  // Month (or, under ~2 months, weekly) labels along the bottom.
  const dateTicks: { i: number; label: string }[] = [];
  data.forEach((b, i) => {
    const d = day(b.date), p = i > 0 ? day(data[i - 1]!.date) : null;
    const label = data.length <= 45
      ? (p && d.getUTCDay() < p.getUTCDay() ? shortDate(b.date) : null)
      : (p && d.getUTCMonth() !== p.getUTCMonth()
        ? (d.getUTCMonth() === 0 ? String(d.getUTCFullYear()) : d.toLocaleDateString("en-US", { timeZone: "UTC", month: "short" }))
        : null);
    if (label && (!dateTicks.length || x(i) - x(dateTicks[dateTicks.length - 1]!.i) > 44)
      && !(showEarn && x(slots - 1) - x(i) < 90)) dateTicks.push({ i, label });
  });

  const prev = data.length > 1 ? data[data.length - 2]! : null;
  const lastUp = !prev || last.close >= prev.close;
  const lastY = y(last.close);

  const hi_ = hover != null ? data[hover] : undefined;
  const shown = hi_ ?? last;
  const shownIdx = hover ?? data.length - 1;
  const shownPrev = shownIdx > 0 ? data[shownIdx - 1] : off > 0 ? all[off - 1] : undefined;
  const chg = shownPrev ? ((shown.close - shownPrev.close) / shownPrev.close) * 100 : null;
  const notes = [
    ...(crossOn.get(shownIdx) === "golden" ? ["Golden cross"] : crossOn.get(shownIdx) === "death" ? ["Death cross"] : []),
    ...(byDay.get(shown.date) ?? []),
  ];

  const pick = (clientX: number, el: SVGSVGElement) => {
    const r = el.getBoundingClientRect();
    if (!r.width) return;
    const px = ((clientX - r.left) * W) / r.width;
    const i = Math.floor((px - PAD.l) / step);
    setHover(i >= 0 && i < data.length ? i : null);
  };
  const onKey = (e: KeyboardEvent<Element>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const cur = hover ?? data.length - 1;
    setHover(Math.min(data.length - 1, Math.max(0, cur + (e.key === "ArrowLeft" ? -1 : 1))));
  };
  const tri = 5;

  return (
    <figure className="chart" ref={fig}>
      <div className="chart-tools">
        <div className="chips" role="group" aria-label="Chart range">
          {ranges.map(([k]) => (
            <button key={k} type="button" className="chip" aria-pressed={k === range[0]} onClick={() => { setRange(k); setHover(null); }}>{k}</button>
          ))}
        </div>
        <div className="chips" role="group" aria-label="Chart lines">
          {OVERLAYS.filter((o) => o.key !== "ema" || hasEma).map((o) => (
            <button key={o.key} type="button" className="chip" aria-pressed={on[o.key]} onClick={() => setOn({ ...on, [o.key]: !on[o.key] })}>
              <i className={`sw ${o.key === "ema" ? "ema-9" : o.key}`} aria-hidden="true" />{o.label}
            </button>
          ))}
        </div>
      </div>
      <div className="chart-readout mono" aria-live="polite">
        <span className="rd-date">{dayLabel(shown.date)}</span>
        {shown.open != null && <span>O {price(shown.open)}</span>}
        {shown.high != null && <span>H {price(shown.high)}</span>}
        {shown.low != null && <span>L {price(shown.low)}</span>}
        <span>C {price(shown.close)}</span>
        {chg != null && <span className={chg >= 0 ? "up" : "down"}>{pct(chg)}</span>}
        {shown.volume != null && <span>{volumeFeed === "iex" ? "IEX vol" : "Vol"} {compact(shown.volume)}</span>}
        {notes.map((n) => <span key={n} className="rd-note">{n}</span>)}
      </div>
      <div className={`interactive-price${interactive ? " is-ready" : ""}`}
        role="img" tabIndex={interactive ? 0 : -1} aria-hidden={!interactive}
        aria-label={`Interactive daily price from ${etDate(first.date)} to ${etDate(last.date)}, last close ${price(last.close)}. Drag to pan, pinch or scroll to zoom. Arrow keys step through sessions.`}
        onKeyDown={onKey} onBlur={() => setHover(null)}>
        <InteractivePriceChart bars={data} events={events} overlays={on} earningsDate={earningsDate}
          onReady={setInteractive} onHover={hoverDate} selectedDate={hover != null ? data[hover]?.date : null} />
      </div>
      <svg style={interactive ? { display: "none" } : undefined} aria-hidden={interactive} viewBox={`0 0 ${W} ${H}`} role="img" tabIndex={interactive ? -1 : 0} onKeyDown={onKey}
        aria-label={`Daily price from ${etDate(first.date)} to ${etDate(last.date)}, last close ${price(last.close)}. Arrow keys step through sessions.`}
        onPointerMove={(e: PointerEvent<SVGSVGElement>) => pick(e.clientX, e.currentTarget)}
        onPointerDown={(e: PointerEvent<SVGSVGElement>) => pick(e.clientX, e.currentTarget)}
        onPointerLeave={() => setHover(null)} onBlur={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="grid" />
            {Math.abs(y(t) - lastY) > 12 && <text x={W - PAD.r + 6} y={y(t) + 4} className="axis">{t.toFixed(decimals)}</text>}
          </g>
        ))}
        {dateTicks.map(({ i, label }) => (
          <g key={`d-${i}`}>
            <line x1={x(i)} x2={x(i)} y1={PAD.t} y2={volBase} className="grid grid-v" />
            <text x={x(i)} y={H - 5} className="axis" textAnchor="middle">{label}</text>
          </g>
        ))}
        {data.map((b, i) => {
          const cx = x(i);
          const o = b.open ?? b.close;
          const up = b.close >= o;
          const vh = ((b.volume ?? 0) / vmax) * (VOL_H - 4);
          return (
            <g key={b.date} className={up ? "c-up" : "c-down"}>
              <line x1={cx} x2={cx} y1={y(b.high ?? Math.max(o, b.close))} y2={y(b.low ?? Math.min(o, b.close))} />
              <rect x={cx - bw / 2} width={bw} y={y(Math.max(o, b.close))} height={Math.max(1, Math.abs(y(o) - y(b.close)))} />
              <rect className="vol" x={cx - bw / 2} width={bw} y={volBase - vh} height={vh} />
            </g>
          );
        })}
        {on.sma50 && <path className="ema sma50" d={linePath(data.map((b) => b.sma50), x, y)} />}
        {on.sma20 && <path className="ema sma20" d={linePath(data.map((b) => b.sma20), x, y)} />}
        {on.ema && <path className="ema ema-21" d={linePath(data.map((b) => b.ema21), x, y)} />}
        {on.ema && <path className="ema ema-9" d={linePath(data.map((b) => b.ema9), x, y)} />}
        {crosses.map(({ i, kind }) => {
          const b = data[i]!;
          const cx = x(i);
          const golden = kind === "golden";
          const yy = golden ? Math.min(plotBottom - tri * 1.6, y(b.low ?? b.close) + 6) : Math.max(PAD.t, y(b.high ?? b.close) - 6);
          const pts = golden ? `${cx},${yy} ${cx - tri},${yy + tri * 1.6} ${cx + tri},${yy + tri * 1.6}` : `${cx},${yy} ${cx - tri},${yy - tri * 1.6} ${cx + tri},${yy - tri * 1.6}`;
          return (
            <polygon key={`${kind}-${b.date}`} className={golden ? "x-golden" : "x-death"} points={pts}>
              <title>{`${golden ? "Golden" : "Death"} cross (EMA 9 ${golden ? "above" : "below"} EMA 21), ${etDate(b.date)}`}</title>
            </polygon>
          );
        })}
        {data.map((b, i) => byDay.has(b.date) && (
          <rect key={`s-${b.date}`} className="sig-mark" x={x(i) - 3.5} y={plotBottom + 1} width={7} height={7} transform={`rotate(45 ${x(i)} ${plotBottom + 4.5})`}>
            <title>{`${byDay.get(b.date)!.join(", ")}, ${etDate(b.date)}`}</title>
          </rect>
        ))}
        {showEarn && (
          <g className="earn-mark">
            <line x1={x(slots - 1)} x2={x(slots - 1)} y1={PAD.t} y2={volBase} />
            <text x={Math.min(x(slots - 1), W - PAD.r)} y={H - 5} textAnchor="end">{`Earnings ${shortDate(earningsDate)}`}</text>
          </g>
        )}
        <line className={`last-line ${lastUp ? "up" : "down"}`} x1={PAD.l} x2={W - PAD.r} y1={lastY} y2={lastY} />
        <rect className={`last-tag ${lastUp ? "up" : "down"}`} x={W - PAD.r + 2} y={lastY - 9} width={PAD.r - 4} height={18} rx={4} />
        <text className="last-tag-text" x={W - PAD.r + 6} y={lastY + 4}>{last.close.toFixed(Math.max(2, decimals))}</text>
        {hi_ && (
          <g className="crosshair" pointerEvents="none">
            <line x1={x(hover!)} x2={x(hover!)} y1={PAD.t} y2={volBase} />
            <line x1={PAD.l} x2={W - PAD.r} y1={y(hi_.close)} y2={y(hi_.close)} />
          </g>
        )}
      </svg>
      {(hasEma || events.length > 0 || showEarn) && (
        <div className="chart-legend" aria-hidden="true">
          {on.ema && <span><i className="mk x-golden">▲</i>Golden cross</span>}
          {on.ema && <span><i className="mk x-death">▼</i>Death cross</span>}
          {events.length > 0 && <span><i className="mk sig">{interactive ? "●" : "◆"}</i>HSF signal</span>}
          {showEarn && <span><i className="mk earn">┆</i>Earnings</span>}
        </div>
      )}
      <figcaption className="cap">
        {interactive && <span className="chart-help">Drag to pan · Pinch or scroll to zoom · Arrow keys read sessions. </span>}
        {data.length} daily bars, {etDate(first.date)} to {etDate(last.date)}.{hasEma ? (lastGolden ? ` Last golden cross ${shortDate(all[lastGolden.i]!.date)} (${all.length - 1 - lastGolden.i === 0 ? "latest session" : `${all.length - 1 - lastGolden.i} sessions ago`}).` : " No recent golden cross.") : ""} End-of-day prices, not live.{earningsDate ? ` Next earnings ${shortDate(earningsDate)}.` : ""}{volumeFeed === "iex" ? " Volume counts IEX exchange trades only, a small share of all U.S. volume." : ""}
        {" "}<a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">Charts by TradingView</a> · <a href="/licenses/lightweight-charts.txt" target="_blank" rel="noreferrer">License</a>
      </figcaption>
    </figure>
  );
}
