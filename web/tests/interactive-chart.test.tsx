import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { PriceChart } from "@/features/PriceChart";
import { chartBars } from "@/lib/chartBars";
import type { Schemas } from "@/api/client";

const lc = vi.hoisted(() => {
  const scale = { fitContent: vi.fn(), getVisibleLogicalRange: vi.fn(() => null), setVisibleLogicalRange: vi.fn(), timeToCoordinate: vi.fn(() => null), subscribeVisibleLogicalRangeChange: vi.fn(), unsubscribeVisibleLogicalRangeChange: vi.fn(), subscribeSizeChange: vi.fn(), unsubscribeSizeChange: vi.fn() };
  const series: { setData: ReturnType<typeof vi.fn>; applyOptions: ReturnType<typeof vi.fn> }[] = [];
  const chart = { setCrosshairPosition: vi.fn(), clearCrosshairPosition: vi.fn(), remove: vi.fn(), applyOptions: vi.fn(), subscribeCrosshairMove: vi.fn((callback: (point: { time: string | null; point?: { x: number; y: number } }) => void) => { void callback; }), timeScale: () => scale,
    addSeries: vi.fn(() => { const s = { setData: vi.fn(), applyOptions: vi.fn(), priceScale: () => ({ applyOptions: vi.fn() }) }; series.push(s); return s; }) };
  return { chart, series, scale, createChart: vi.fn((container: unknown, options: unknown) => { void container; void options; return chart; }), markers: { setMarkers: vi.fn() } };
});
vi.mock("lightweight-charts", () => ({ createChart: lc.createChart, createSeriesMarkers: () => lc.markers,
  CandlestickSeries: "candles", HistogramSeries: "volume", LineSeries: "line", ColorType: { Solid: "solid" }, CrosshairMode: { Normal: 0 } }));

const bars: Schemas["Bar"][] = Array.from({ length: 90 }, (_, i) => {
  const date = new Date("2026-05-01T12:00:00Z"); date.setUTCDate(date.getUTCDate() + i);
  return { date: date.toISOString().slice(0, 10), open: 100 + i, high: 103 + i, low: 99 + i, close: 102 + i,
    volume: 1000 + i, ema9: i < 20 ? 100 : 110, ema21: 105, sma20: 102, sma50: 101 };
});
beforeEach(() => {
  vi.clearAllMocks(); lc.series.length = 0;
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: false, addEventListener() {}, removeEventListener() {} })));
});
afterEach(() => vi.unstubAllGlobals());

it("normalizes out-of-order sessions, duplicates and missing OHLC without changing source bars", () => {
  const first = { ...bars[0]!, open: null, high: null, low: null };
  const input = [bars[1]!, first, { ...first, close: 200 }, { ...bars[2]!, date: "2026-02-30" }, { ...bars[3]!, close: NaN }];
  const clean = chartBars(input);
  expect(clean).toHaveLength(2);
  expect(clean[0]).toMatchObject({ date: first.date, close: 200, open: 200, high: 200, low: 200 });
  expect(first.open).toBeNull();
});

it("renders interactive candles, volume, signal and cross markers with preserved controls and cleanup", async () => {
  const { unmount } = render(<PriceChart bars={bars} events={[{ date: bars[40]!.date, label: "Breakout" }]} earningsDate="2026-08-05" volumeFeed="iex" />);
  const image = await screen.findByRole("group", { name: /Interactive daily price/ });
  expect(lc.createChart).toHaveBeenCalledTimes(1);
  expect(lc.createChart.mock.calls[0]![1]).toMatchObject({ layout: { attributionLogo: true }, handleScroll: { vertTouchDrag: false } });
  expect(lc.series[0]!.setData.mock.lastCall![0]).toHaveLength(91);
  expect(lc.series[0]!.setData.mock.lastCall![0].at(-1)).toEqual({ time: "2026-08-05" });
  expect(lc.markers.setMarkers.mock.lastCall![0]).toEqual(expect.arrayContaining([
    expect.objectContaining({ time: bars[40]!.date, shape: "circle" }), expect.objectContaining({ shape: "arrowUp" }),
  ]));
  expect(screen.getByText(/Volume counts IEX/)).toBeInTheDocument();
  expect(screen.getByText(/Next earnings/)).toBeInTheDocument();
  const crosshair = lc.chart.subscribeCrosshairMove.mock.calls[0]![0];
  act(() => crosshair({ time: bars[40]!.date, point: { x: 10, y: 10 } }));
  expect(screen.getByText("Breakout")).toBeInTheDocument();
  act(() => crosshair({ time: null }));
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "SMA 50" }));
  expect(lc.series[5]!.applyOptions).toHaveBeenLastCalledWith({ visible: true });
  fireEvent.keyDown(image, { key: "ArrowLeft" });
  expect(screen.getByText("C $190.00")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "1M" }));
  expect(lc.series[0]!.setData.mock.lastCall![0]).toHaveLength(22);
  expect(lc.createChart).toHaveBeenCalledTimes(1);
  unmount(); expect(lc.chart.remove).toHaveBeenCalledTimes(1);
});

it("keeps the keyboard-accessible SVG if the interactive renderer fails", async () => {
  lc.createChart.mockImplementationOnce(() => { throw new Error("Canvas unavailable"); });
  render(<PriceChart bars={bars} />);
  await waitFor(() => expect(lc.createChart).toHaveBeenCalled());
  expect(screen.getByRole("img", { name: /^Daily price/ })).toHaveAttribute("tabindex", "0");
  expect(screen.queryByRole("group", { name: /Interactive daily price/ })).not.toBeInTheDocument();
});
