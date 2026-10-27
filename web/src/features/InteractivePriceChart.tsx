"use client";

import { useEffect, useRef, useState } from "react";
import type { IChartApi, ISeriesApi, ISeriesMarkersPluginApi, SeriesMarker, Time } from "lightweight-charts";
import type { Schemas } from "@/api/client";
import type { ChartEvent } from "./PriceChart";
import { emaCrosses } from "@/lib/chartBars";

type Bar = Schemas["Bar"];
type Props = {
  bars: Bar[];
  events: ChartEvent[];
  overlays: { ema: boolean; sma20: boolean; sma50: boolean; rsi: boolean; vol: boolean };
  /** RSI(14) per bar, aligned with `bars` (null while it warms up). */
  rsi?: (number | null)[];
  /** 20-session average volume per bar, aligned with `bars`. */
  avgVol?: (number | null)[];
  earningsDate?: string | null;
  onReady: (ready: boolean) => void;
  onHover: (date: string | null) => void;
  selectedDate?: string | null;
};
type Drawing = {
  chart: IChartApi;
  candles: ISeriesApi<"Candlestick">;
  volume: ISeriesApi<"Histogram">;
  lines: Record<"ema9" | "ema21" | "sma20" | "sma50", ISeriesApi<"Line">>;
  markers: ISeriesMarkersPluginApi<Time>;
  rsi: ISeriesApi<"Line">;
  avgVol: ISeriesApi<"Line">;
};

const BASE_H = 280;
const RSI_H = 110;
const VOL_H = 70;

/** Price pane first, then volume and RSI in their own panes when shown. Hidden ones park
 * in the price pane (invisible) so no empty strip is left behind. */
function arrangePanes(d: Drawing, showVol: boolean, showRsi: boolean) {
  const { chart } = d;
  if (typeof d.rsi.moveToPane !== "function" || typeof chart.panes !== "function") return;
  const want = { vol: showVol ? 1 : 0, rsi: showRsi ? (showVol ? 2 : 1) : 0 };
  const panes = 1 + (showVol ? 1 : 0) + (showRsi ? 1 : 0);
  const now = { vol: d.volume.getPane().paneIndex(), rsi: d.rsi.getPane().paneIndex() };
  if (now.vol !== want.vol || now.rsi !== want.rsi || chart.panes().length !== panes) {
    for (const s of [d.rsi, d.avgVol, d.volume]) s.moveToPane(0);
    for (let i = chart.panes().length - 1; i > 0; i--) if (typeof chart.removePane === "function") chart.removePane(i);
    if (showVol) { d.volume.moveToPane(1); d.avgVol.moveToPane(1); }
    if (showRsi) d.rsi.moveToPane(want.rsi);
  }
  if (showVol) d.volume.priceScale().applyOptions({ scaleMargins: { top: 0.12, bottom: 0 }, visible: true });
  if (showRsi) d.rsi.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.1 } });
  // Stretch factors share the height in proportion (setHeight on one pane resizes its neighbour).
  const all = chart.panes();
  all[0]?.setStretchFactor(BASE_H);
  if (showVol) all[1]?.setStretchFactor(VOL_H);
  if (showRsi) all[want.rsi]?.setStretchFactor(RSI_H);
}

/** Browser-only enhancement. The SVG stays available if Canvas or loading fails. */
export function InteractivePriceChart({ bars, events, overlays, rsi, avgVol, earningsDate, onReady, onHover, selectedDate }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const drawing = useRef<Drawing | null>(null);
  const earningsLabel = useRef<HTMLDivElement>(null);
  const previousDates = useRef("");
  const callbacks = useRef({ onReady, onHover });
  const [ready, setReady] = useState(false);
  useEffect(() => { callbacks.current = { onReady, onHover }; }, [onReady, onHover]);

  useEffect(() => {
    const el = host.current;
    if (!el || typeof ResizeObserver === "undefined" || typeof window.matchMedia !== "function") return;
    let disposed = false;
    let observer: ResizeObserver | undefined;
    let chart: IChartApi | undefined;
    void import("lightweight-charts").then((lc) => {
      if (disposed) return;
      const css = getComputedStyle(el);
      const color = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
      const up = color("--up", "#4ade80"), down = color("--down", "#f87171");
      chart = lc.createChart(el, {
        width: el.clientWidth || 720, height: BASE_H,
        layout: { background: { type: lc.ColorType.Solid, color: color("--bg-2", "#111822") },
          textColor: color("--text-2", "#aab6c5"), fontFamily: css.getPropertyValue("--font-sans").trim() || "sans-serif", attributionLogo: false,
          panes: { separatorColor: color("--line", "#26303c"), separatorHoverColor: color("--line-strong", "#34465f") } },
        grid: { vertLines: { color: color("--line-soft", "#202a35") }, horzLines: { color: color("--line-soft", "#202a35") } },
        rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.12, bottom: 0.08 } },
        timeScale: { borderVisible: false, rightOffset: 3, timeVisible: false },
        handleScroll: { vertTouchDrag: false, horzTouchDrag: true, mouseWheel: false, pressedMouseMove: true },
        handleScale: { pinch: true, mouseWheel: true, axisPressedMouseMove: true },
        crosshair: { mode: lc.CrosshairMode.Normal },
      });
      // TradingView attribution lives in the chart caption (PriceChart), so the library's own
      // logo link is off. That link sat inside this aria-hidden box, and axe flagged it as
      // focusable whenever the check ran before our tabIndex fix-up (aria-hidden-focus).
      const candles = chart.addSeries(lc.CandlestickSeries, { upColor: up, downColor: down,
        borderVisible: false, wickUpColor: up, wickDownColor: down });
      // Volume and its average sit on the right scale of their own pane (labelled in shares);
      // when Volume is off they're hidden in the price pane, where hidden series don't scale.
      const volume = chart.addSeries(lc.HistogramSeries, { priceFormat: { type: "volume" }, color: color("--line-strong", "#34465f"),
        priceLineVisible: false, lastValueVisible: true });
      const line = (c: string) => chart!.addSeries(lc.LineSeries, { color: c, lineWidth: 1,
        priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      const lines = { ema9: line(color("--accent", "#72d5ee")), ema21: line(color("--amber", "#e9b872")),
        sma20: line("#a995e7"), sma50: line("#ed9eaf") };
      const markers = lc.createSeriesMarkers(candles, []);
      // RSI(14) in its own pane under the candles, with the usual 70 / 30 guides.
      const rsiLine = chart.addSeries(lc.LineSeries, { color: color("--sma20", "#c792ea"), lineWidth: 1, priceLineVisible: false,
        lastValueVisible: true, crosshairMarkerVisible: false, autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }) }, 1);
      rsiLine.priceScale().applyOptions({ scaleMargins: { top: 0.1, bottom: 0.1 } });
      for (const level of [70, 30]) rsiLine.createPriceLine({ price: level, color: color("--line-strong", "#34465f"), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: "" });
      // 20-session average volume, drawn over the volume bars in the volume pane.
      const avgVolLine = chart.addSeries(lc.LineSeries, { color: color("--text-3", "#7d8a9b"), lineWidth: 1, lineStyle: 2,
        priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      chart.subscribeCrosshairMove((p) => {
        if (!p.time || !p.point || p.point.x < 0 || p.point.y < 0) { callbacks.current.onHover(null); return; }
        const t = p.time;
        const date = typeof t === "string" ? t : typeof t === "object"
          ? `${t.year}-${String(t.month).padStart(2, "0")}-${String(t.day).padStart(2, "0")}` : null;
        callbacks.current.onHover(date);
      });
      drawing.current = { chart, candles, volume, lines, markers, rsi: rsiLine, avgVol: avgVolLine };
      observer = new ResizeObserver(([entry]) => {
        if (!disposed && entry && entry.contentRect.width > 0) chart?.applyOptions({ width: Math.floor(entry.contentRect.width) });
      });
      observer.observe(el);
      setReady(true);
    }).catch(() => { if (!disposed) callbacks.current.onReady(false); });
    return () => {
      disposed = true;
      observer?.disconnect();
      chart?.remove();
      drawing.current = null;
    };
  }, []);

  useEffect(() => {
    const d = drawing.current;
    if (!ready || !d || bars.length < 2) return;
    try {
      const lastDate = bars.at(-1)!.date;
      const upcoming = earningsDate && /^\d{4}-\d{2}-\d{2}$/.test(earningsDate) && earningsDate > lastDate
        && Date.parse(earningsDate) - Date.parse(lastDate) <= 14 * 86400000 ? earningsDate : null;
      const oldRange = d.chart.timeScale().getVisibleLogicalRange();
      const datesKey = `${bars[0]!.date}:${lastDate}:${bars.length}:${upcoming ?? ""}`;
      d.candles.setData([...bars.map((b) => ({ time: b.date, open: b.open ?? b.close,
        high: b.high ?? b.close, low: b.low ?? b.close, close: b.close })), ...(upcoming ? [{ time: upcoming }] : [])]);
      d.volume.setData(bars.map((b) => ({ time: b.date, value: Math.max(0, b.volume ?? 0),
        color: b.close >= (b.open ?? b.close) ? "rgba(74,222,128,0.28)" : "rgba(248,113,113,0.28)" })));
      for (const key of ["ema9", "ema21", "sma20", "sma50"] as const) {
        d.lines[key].setData(bars.map((b) => b[key] != null && Number.isFinite(b[key])
          ? { time: b.date, value: b[key]! } : { time: b.date }));
        d.lines[key].applyOptions({ visible: key.startsWith("ema") ? overlays.ema : overlays[key as "sma20" | "sma50"] });
      }
      const dates = new Set(bars.map((b) => b.date));
      const goldenColor = host.current ? getComputedStyle(host.current).getPropertyValue("--gold").trim() : "#e9b872";
      const marks: SeriesMarker<Time>[] = events.filter((e) => dates.has(e.date)).map((e) => ({
        time: e.date, position: "belowBar", color: "#72d5ee", shape: "circle", text: "",
      }));
      if (overlays.ema) for (const cross of emaCrosses(bars)) {
        marks.push({ time: bars[cross.i]!.date, position: cross.kind === "golden" ? "belowBar" : "aboveBar",
          color: cross.kind === "golden" ? goldenColor : "#f87171", shape: cross.kind === "golden" ? "arrowUp" : "arrowDown",
          text: "" });
      }
      if (earningsDate && dates.has(earningsDate)) marks.push({ time: earningsDate, position: "aboveBar", color: "#e9b872", shape: "square", text: "Earnings" });
      d.markers.setMarkers(marks.sort((a, b) => String(a.time).localeCompare(String(b.time))));
      const showRsi = overlays.rsi && !!rsi?.some((v) => v != null);
      d.rsi.setData(bars.map((b, i) => { const v = rsi?.[i]; return v != null && Number.isFinite(v) ? { time: b.date, value: v } : { time: b.date }; }));
      d.rsi.applyOptions({ visible: showRsi });
      d.volume.applyOptions({ visible: overlays.vol });
      d.avgVol.setData(bars.map((b, i) => { const v = avgVol?.[i]; return v != null && Number.isFinite(v) ? { time: b.date, value: v } : { time: b.date }; }));
      d.avgVol.applyOptions({ visible: overlays.vol });
      arrangePanes(d, overlays.vol, showRsi);
      d.chart.applyOptions({ height: BASE_H + (overlays.vol ? VOL_H : 0) + (showRsi ? RSI_H : 0) });
      // Keep the user's zoom when toggling an overlay; a new date range fits its data.
      if (oldRange && previousDates.current === datesKey) d.chart.timeScale().setVisibleLogicalRange(oldRange);
      else d.chart.timeScale().fitContent();
      const positionEarnings = () => {
        const label = earningsLabel.current;
        if (!label) return;
        const x = upcoming ? d.chart.timeScale().timeToCoordinate(upcoming) : null;
        label.hidden = x == null || x < 0 || x > (host.current?.clientWidth ?? 0);
        if (x != null) label.style.left = `${x}px`;
      };
      positionEarnings();
      const frame = typeof requestAnimationFrame === "function" ? requestAnimationFrame(positionEarnings) : null;
      d.chart.timeScale().subscribeVisibleLogicalRangeChange(positionEarnings);
      d.chart.timeScale().subscribeSizeChange(positionEarnings);
      previousDates.current = datesKey;
      callbacks.current.onReady(true);
      return () => {
        if (frame != null) cancelAnimationFrame(frame);
        d.chart.timeScale().unsubscribeVisibleLogicalRangeChange(positionEarnings);
        d.chart.timeScale().unsubscribeSizeChange(positionEarnings);
      };
    } catch {
      callbacks.current.onReady(false);
    }
  }, [ready, bars, events, overlays, rsi, avgVol, earningsDate]);

  useEffect(() => {
    const d = drawing.current;
    if (!ready || !d) return;
    const bar = selectedDate ? bars.find((b) => b.date === selectedDate) : null;
    if (bar) d.chart.setCrosshairPosition(bar.close, bar.date, d.candles);
    else d.chart.clearCrosshairPosition();
  }, [ready, bars, selectedDate]);

  return <div className="lightweight-chart-frame"><div ref={host} className="lightweight-chart" aria-hidden="true" data-testid="interactive-price-chart" /><div className="chart-earnings-date" ref={earningsLabel} hidden aria-hidden="true"><span>Earnings</span></div></div>;
}
