import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DT_REFRESH_MS, DayTraderView, NO_FILTERS, applyFilters, diffRefresh, holdOrder, sortRows } from "@/features/DayTraderView";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/day-trader",
}));

const row = (o: Record<string, unknown> & { ticker: string }) => ({ dt_quality: "developing", dt_direction: "bullish", dt_reasons: [], dt_conflicts: [], quote_flags: [], ...o });
const ROWS = [
  row({ ticker: "WFF", day_trade_score: 92, last: 13, chg_pct: 516.1, session_chg_pct: 516.1, rvol: 18.4, volume: 1e6, quote_flags: ["Extreme move"] }),
  row({ ticker: "VEEA", day_trade_score: 71, dt_quality: "strong", last: 5.65, chg_pct: 46.8, session_chg_pct: 40.0, ext_chg_pct: 4.8, rvol: 2.6, volume: 351_500,
    dt_reasons: ["Above VWAP (+1.20%)"], dt_conflicts: ["Gap fading"] }),
  row({ ticker: "TMUS", day_trade_score: 55, dt_direction: "bearish", last: 148.57, chg_pct: -13.2, session_chg_pct: -13.2, rvol: 2.1, volume: 617_300 }),
  row({ ticker: "ATEX", day_trade_score: null, dt_quality: "insufficient", dt_direction: "neutral", last: 84.33, chg_pct: 10.4, rvol: 1.6, volume: 17_800 }),
];

const fetchMock = vi.fn();
beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockImplementation(async (req: Request) => {
    if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: ["VEEA"], series: { VEEA: [5, 5.2, 5.65] } });
    return jsonResponse({ state: "closed", source: "movers", symbols: ROWS.map((r) => r.ticker), missing: 0, as_of: new Date().toISOString(), rows: ROWS });
  });
});
afterEach(() => vi.unstubAllGlobals());

const tickers = () => screen.getAllByRole("row").slice(1).map((r) => within(r).queryAllByRole("link")[0]?.textContent).filter(Boolean);

describe("Day Trader table", () => {
  it("ranks flagged quotes and unscored rows last, and shows the session split", async () => {
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await waitFor(() => expect(tickers()).toEqual(["VEEA", "TMUS", "ATEX", "WFF"]));
    const wff = screen.getAllByRole("row")[4]!;
    expect(within(wff).getByText("Check quote")).toBeInTheDocument();
    expect(within(wff).queryByText("Developing")).not.toBeInTheDocument();   // no tier on a flagged quote
    expect(within(wff).getByText("92")).toHaveAttribute("title", "Check the quote before relying on this score");
    expect(within(screen.getAllByRole("row")[1]!).getByText("Strong")).toBeInTheDocument();
    expect(screen.getByText("AH +4.80%")).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /VEEA today: up/ })).toBeInTheDocument();
  });

  it("sorts by a column, filters by side and explains a score", async () => {
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await waitFor(() => expect(tickers()).toHaveLength(4));
    await u.click(screen.getByRole("button", { name: "RVOL" }));
    expect(tickers()).toEqual(["WFF", "VEEA", "TMUS", "ATEX"]);
    await u.click(screen.getByRole("button", { name: "Short setups" }));
    expect(tickers()).toEqual(["TMUS"]);
    expect(screen.getByText("Showing 1 of 4")).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Clear" }));
    await u.click(screen.getByRole("button", { name: "Why VEEA scores 71" }));
    expect(screen.getByText("Gap fading")).toBeInTheDocument();
    expect(screen.getByText("Above VWAP (+1.20%)")).toBeInTheDocument();
  });

  it("filters by volume and hides flagged quotes", () => {
    type R = Parameters<typeof applyFilters>[0][number];
    const rows = ROWS as unknown as R[];
    expect(applyFilters(rows, { ...NO_FILTERS, minVolume: 500_000 }).map((r) => r.ticker)).toEqual(["WFF", "TMUS"]);
    expect(applyFilters(rows, { ...NO_FILTERS, hideFlagged: true }).map((r) => r.ticker)).not.toContain("WFF");
    expect(sortRows(rows, "ticker", false).map((r) => r.ticker)).toEqual(["ATEX", "TMUS", "VEEA", "WFF"]);
  });
});

describe("Day Trader live refresh", () => {
  type D = Parameters<typeof diffRefresh>[0];
  const data = (rows: unknown[], as_of: string, source = "movers") => ({ state: "open", source, symbols: [], missing: 0, as_of, rows }) as unknown as D;

  it("reports rank moves and price/score direction between refreshes", () => {
    const a = data([row({ ticker: "A", day_trade_score: 80, last: 10 }), row({ ticker: "B", day_trade_score: 60, last: 20 })], "t1");
    const b = data([row({ ticker: "A", day_trade_score: 50, last: 9.5 }), row({ ticker: "B", day_trade_score: 70, last: 20 })], "t2");
    expect(diffRefresh(a, b)).toEqual({ A: { rank: -1, last: -1, score: -1 }, B: { rank: 1, last: 0, score: 1 } });
    expect(diffRefresh(a, data([], "t2", "watchlist"))).toEqual({});   // a source switch is not a move
    expect(diffRefresh(a, a)).toEqual({});
  });

  it("holds a paused order and appends new names", () => {
    type R = Parameters<typeof holdOrder>[0][number];
    const rows = [row({ ticker: "C" }), row({ ticker: "A" }), row({ ticker: "B" })] as unknown as R[];
    expect(holdOrder(rows, ["B", "A"]).map((r) => r.ticker)).toEqual(["B", "A", "C"]);
  });

  it("refreshes every 15 seconds while trading, shows rank moves, and stops while the tab is hidden", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let n = 0;
    fetchMock.mockImplementation(async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      n += 1;
      const rows = n === 1
        ? [row({ ticker: "AAA", day_trade_score: 80, last: 10 }), row({ ticker: "BBB", day_trade_score: 60, last: 5 })]
        : [row({ ticker: "AAA", day_trade_score: 50, last: 9 }), row({ ticker: "BBB", day_trade_score: 70, last: 5.5 })];
      return jsonResponse({ state: "open", source: "movers", symbols: ["AAA", "BBB"], missing: 0, as_of: new Date().toISOString(), rows });
    });
    try {
      render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
      await waitFor(() => expect(tickers()).toEqual(["AAA", "BBB"]));
      expect(screen.getByRole("status")).toHaveTextContent(/Live · updated \d+s ago/);
      await vi.advanceTimersByTimeAsync(DT_REFRESH_MS + 50);
      await waitFor(() => expect(tickers()).toEqual(["BBB", "AAA"]));
      expect(screen.getByLabelText("up 1 place")).toBeInTheDocument();
      expect(screen.getByLabelText("down 1 place")).toBeInTheDocument();

      const calls = () => fetchMock.mock.calls.filter((c) => new URL((c[0] as Request).url).pathname.endsWith("/v1/day-trader")).length;
      const before = calls();
      Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
      document.dispatchEvent(new Event("visibilitychange"));
      await vi.advanceTimersByTimeAsync(DT_REFRESH_MS * 3);
      expect(calls()).toBe(before);
      expect(screen.getByText(/paused while this tab is hidden/)).toBeInTheDocument();
    } finally {
      Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "visible" });
      vi.useRealTimers();
    }
  });

  it("keeps rows in place while the order is paused", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let n = 0;
    fetchMock.mockImplementation(async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      n += 1;
      const rows = n === 1
        ? [row({ ticker: "AAA", day_trade_score: 80 }), row({ ticker: "BBB", day_trade_score: 60 })]
        : [row({ ticker: "AAA", day_trade_score: 50 }), row({ ticker: "BBB", day_trade_score: 70 })];
      return jsonResponse({ state: "open", source: "movers", symbols: ["AAA", "BBB"], missing: 0, as_of: new Date().toISOString(), rows });
    });
    try {
      const u = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
      await waitFor(() => expect(tickers()).toEqual(["AAA", "BBB"]));
      await u.click(screen.getByRole("button", { name: "Pause order" }));
      await vi.advanceTimersByTimeAsync(DT_REFRESH_MS + 50);
      await waitFor(() => expect(screen.getByText("70")).toBeInTheDocument());   // the new scores are in
      expect(tickers()).toEqual(["AAA", "BBB"]);
      await u.click(screen.getByRole("button", { name: "Resume order" }));
      expect(tickers()).toEqual(["BBB", "AAA"]);
    } finally {
      vi.useRealTimers();
    }
  });
});
