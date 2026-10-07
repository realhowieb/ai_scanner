import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { CustomScanView } from "@/features/CustomScanView";
import { ScannerView } from "@/features/ScannerView";
import { StockView } from "@/features/StockView";
import type { ScanApi } from "@/hooks/useScanJob";
import { SessionProvider } from "@/session/SessionProvider";

import { me, setup } from "./fixtures";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/scanner",
}));

const fetchMock = vi.fn();
beforeEach(() => {
  search = new URLSearchParams();
  replace.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

function urlOf(call: unknown[]): URL {
  const input = call[0] as Request | string;
  return new URL(typeof input === "string" ? input : input.url);
}

const latest = (over: Partial<Schemas["LatestScan"]> = {}): Schemas["LatestScan"] => ({
  scan_at: new Date().toISOString(), total: 140, max_results: 25, limited: true,
  setups: [setup("AAA", 88), setup("BBB", 70, { prob: 62.4 })], ...over,
});

describe("Scanner", () => {
  it("shows the plan cap from the server and an upgrade path", async () => {
    fetchMock.mockResolvedValue(jsonResponse(latest()));
    render(<SessionProvider initialMe={me("basic")}><ScannerView /></SessionProvider>);
    expect(await screen.findAllByRole("link", { name: "AAA" })).not.toHaveLength(0);
    const note = screen.getByRole("note");
    expect(note).toHaveTextContent("top 25 of 140 setups");
    expect(within(note).getByRole("button", { name: "Upgrade to Pro" })).toBeInTheDocument();
    expect(screen.queryByRole("navigation", { name: "Pages" })).not.toBeInTheDocument(); // 25 visible rows = one page
  });

  it("sends the filters to the API and pages within the plan cap", async () => {
    search = new URLSearchParams("signal=breakout&min=60&size=25&page=2");
    fetchMock.mockResolvedValue(jsonResponse(latest({ total: 80, max_results: 100, limited: false })));
    render(<SessionProvider initialMe={me("pro")}><ScannerView /></SessionProvider>);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const q = urlOf(fetchMock.mock.calls[0]!).searchParams;
    expect(Object.fromEntries(q)).toEqual({ limit: "25", offset: "25", min_score: "60", signal: "breakout" });
    expect(await screen.findByText("Page 2 of 4")).toBeInTheDocument();
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });

  it("locks PreBreakout below Premium without asking the API", async () => {
    search = new URLSearchParams("signal=prebreakout");
    render(<SessionProvider initialMe={me("pro")}><ScannerView /></SessionProvider>);
    expect(screen.getByText("PreBreakout is part of Premium")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("shows PreBreakout probability only for Premium", async () => {
    fetchMock.mockResolvedValue(jsonResponse(latest({ limited: false, max_results: 200 })));
    const { unmount } = render(<SessionProvider initialMe={me("premium")}><ScannerView /></SessionProvider>);
    expect((await screen.findAllByText("62%")).length).toBeGreaterThan(0);
    unmount();
    fetchMock.mockResolvedValue(jsonResponse(latest({ limited: false, setups: [setup("BBB", 70, { prob: null })] })));
    render(<SessionProvider initialMe={me("pro")}><ScannerView /></SessionProvider>);
    await screen.findAllByRole("link", { name: "BBB" });
    expect(screen.queryByText("62%")).not.toBeInTheDocument();
  });

  it("shows a service outage with the support code and a retry", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: "database unavailable" }, 503, { "retry-after": "30", "x-request-id": "web-feedbeef" }));
    render(<SessionProvider initialMe={me("pro")}><ScannerView /></SessionProvider>);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Couldn't load the Scanner right now");
    expect(alert).toHaveTextContent("Try again in 30s");
    expect(alert).toHaveTextContent("Support code: web-feedbeef");
  });
});

const idle: ScanApi = { list: async () => [], create: vi.fn(), get: vi.fn(), cancel: vi.fn() };

describe("Custom scan controls follow server entitlements", () => {
  it("Free: NASDAQ, Combo, US market, pre-market and Pro filters are locked; rows above the cap disabled", async () => {
    fetchMock.mockResolvedValue(jsonResponse(latest({ max_results: 25 })));
    render(<SessionProvider initialMe={me("basic")}><CustomScanView client={idle} /></SessionProvider>);
    expect(screen.getByRole("radio", { name: /S&P 500$/ })).toBeEnabled();
    for (const name of [/^NASDAQ/, /^Combo/, /^US market/]) expect(screen.getByRole("radio", { name })).toBeDisabled();
    expect(screen.getByRole("radio", { name: /Pre-market/ })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: /Unusual volume/ })).toBeDisabled();
    await waitFor(() => expect(screen.getByRole("option", { name: "50 (upgrade)" })).toBeDisabled());
  });

  it("Pro: NASDAQ unlocked, US market still Premium; Premium: everything", () => {
    fetchMock.mockResolvedValue(jsonResponse(latest({ max_results: 100 })));
    const { unmount } = render(<SessionProvider initialMe={me("pro")}><CustomScanView client={idle} /></SessionProvider>);
    expect(screen.getByRole("radio", { name: /^NASDAQ/ })).toBeEnabled();
    expect(screen.getByRole("radio", { name: /^US market/ })).toBeDisabled();
    unmount();
    render(<SessionProvider initialMe={me("premium")}><CustomScanView client={idle} /></SessionProvider>);
    expect(screen.getByRole("radio", { name: /^US market/ })).toBeEnabled();
  });

  it("shows the server's plan refusal with an upgrade button", async () => {
    fetchMock.mockResolvedValue(jsonResponse(latest()));
    const { ApiError } = await import("@/api/client");
    const refusing: ScanApi = { ...idle, create: async () => { throw new ApiError(403, "US market scans are part of Premium.", null, null, null); } };
    render(<SessionProvider initialMe={me("pro")}><CustomScanView client={refusing} /></SessionProvider>);
    screen.getByRole("button", { name: "Start scan" }).click();
    expect(await screen.findByText("US market scans are part of Premium.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upgrade to Premium" })).toBeInTheDocument();
  });
});

describe("Custom scan waiting and cancel", () => {
  const queuedJob = (createdAgoS: number): Schemas["ScanJob"] => ({
    scan_id: "d".repeat(32), status: "queued", universe: "sp500",
    params: { universe: "sp500", ticker: null, watchlist_id: null, score_all: false, profile: "regular", session_requested: "regular",
      session: "regular", min_price: 1, max_price: 1000, min_dollar_vol: 0, min_gap: 0, apply_gap_filter: false, unusual_volume: false,
      top_n: 25, max_results: 25, full_lists: false, max_nasdaq: null, max_combo: null },
    progress: { phase: "queued", symbols: null, elapsed_s: null }, result: null, error: null,
    created_at: new Date(Date.now() - createdAgoS * 1000).toISOString(), started_at: null, finished_at: null,
  });

  it("shows how long it has waited and explains a long wait", async () => {
    fetchMock.mockResolvedValue(jsonResponse(latest()));
    const slow: ScanApi = { ...idle, list: async () => [queuedJob(95)], get: async () => queuedJob(95) };
    render(<SessionProvider initialMe={me("pro")}><CustomScanView client={slow} /></SessionProvider>);
    expect(await screen.findByText(/Waiting for 1m 3\ds/)).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("taking longer than usual");
  });

  it("no long-wait note in the first minute", async () => {
    fetchMock.mockResolvedValue(jsonResponse(latest()));
    const fresh: ScanApi = { ...idle, list: async () => [queuedJob(5)], get: async () => queuedJob(5) };
    render(<SessionProvider initialMe={me("pro")}><CustomScanView client={fresh} /></SessionProvider>);
    expect(await screen.findByText(/Waiting for \ds/)).toBeInTheDocument();
    expect(screen.queryByText(/taking longer than usual/)).not.toBeInTheDocument();
  });

  it("Cancel scan stops it and offers a new scan", async () => {
    fetchMock.mockResolvedValue(jsonResponse(latest()));
    const cancelled = { ...queuedJob(10), status: "failed" as const, error: "Cancelled." };
    const api: ScanApi = { ...idle, list: async () => [queuedJob(10)], get: async () => queuedJob(10), cancel: vi.fn(async () => cancelled) };
    render(<SessionProvider initialMe={me("pro")}><CustomScanView client={api} /></SessionProvider>);
    (await screen.findByRole("button", { name: "Cancel scan" })).click();
    expect(await screen.findByText("Scan cancelled")).toBeInTheDocument();
    expect(api.cancel).toHaveBeenCalledWith("d".repeat(32));
    expect(screen.getByRole("button", { name: "Start scan" })).toBeEnabled();
  });
});

const stock = (over: Partial<Schemas["StockDetail"]> = {}): Schemas["StockDetail"] => ({
  ticker: "AAA", scan_at: new Date().toISOString(), in_latest_scan: true, has_setup: true, from_history: false, price: 12.5, change_pct: 1.2,
  hsf_score: 77, status: "STRONG", primary_setup: "breakout", signals: ["breakout"], score_components: { signals_component: 30 },
  movement: "RISING", score_change: 4, reasons: ["Breaking out"], risks: [], watch_next: [], breakout_score: 80, prob: 13.1,
  earnings_days: 3, history_summary: { observations: 4, matured: 3, positive: 2 }, historical_context: null, outcome_cohort: null,
  historical_locked: false, lifecycle: [], bars: [], bars_as_of: null, watchlists: [], alerts: [], ...over,
});

describe("Stock Intelligence", () => {
  it("labels the price as the scan's, not a live quote, and handles missing bars", () => {
    render(<StockView s={stock()} premium />);
    expect(screen.getByText(/not a live quote/)).toBeInTheDocument();
    expect(screen.getByText("No price history cached for this ticker.")).toBeInTheDocument();
    expect(screen.getByText("13%")).toBeInTheDocument();
  });

  it("explains the historical fallback when the ticker isn't in the latest scan", () => {
    render(<StockView s={stock({ in_latest_scan: false, from_history: true })} premium={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("wasn't in the latest market scan");
    expect(screen.getByRole("status")).toHaveTextContent("last recorded HSF observation");
    expect(screen.getByText(/from the last recorded observation/)).toBeInTheDocument();
  });

  it("shows a plain state when HSF has no score at all", () => {
    render(<StockView s={stock({ in_latest_scan: false, has_setup: false, hsf_score: null, status: null, price: null, signals: [], score_components: null })} premium={false} />);
    expect(screen.getByRole("status")).toHaveTextContent("HSF has no recorded score");
    expect(screen.getByText("No HSF Score")).toBeInTheDocument();
  });

  it("locks redacted fields: historical research below Pro, PreBreakout below Premium", () => {
    render(<StockView s={stock({ historical_locked: true, history_summary: null, prob: null })} premium={false} />);
    expect(screen.getByText("Historical research is part of Pro")).toBeInTheDocument();
    expect(screen.queryByText(/earlier HSF observation/)).not.toBeInTheDocument();
    expect(screen.getAllByText("Premium").length).toBeGreaterThan(0);
  });

  it("shows PreBreakout in the API's percent units", () => {
    render(<StockView s={stock()} premium />);
    expect(screen.getByText("13%")).toBeInTheDocument();
    expect(screen.queryByText("1310%")).not.toBeInTheDocument();
  });

  it("draws the chart from returned bars", () => {
    const bars = Array.from({ length: 30 }, (_, i) => ({ date: `2026-09-${String(i + 1).padStart(2, "0")}`, open: 10 + i, high: 11 + i, low: 9 + i, close: 10.5 + i, volume: 1e6 }));
    render(<StockView s={stock({ bars, bars_as_of: new Date().toISOString() })} premium />);
    expect(screen.getByRole("img", { name: /Daily price from/ })).toBeInTheDocument();
    expect(screen.getByText(/30 daily bars/)).toHaveTextContent("Not live prices");
  });
});
