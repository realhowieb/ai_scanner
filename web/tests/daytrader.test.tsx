import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_PREFS, DT_REFRESH_MS, DT_RETRY_MS, DayTraderView, MAX_CUSTOM, NO_FILTERS, applyFilters, diffRefresh, earningsLabel, holdOrder, parseTickers, readPrefs, sortRows, orderLists } from "@/features/DayTraderView";
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
// The Symbols menu loads the user's watchlists; answered here so tests count only quote calls.
let watchlists: unknown[] = [];
beforeEach(() => {
  fetchMock.mockReset();
  watchlists = [];
  localStorage.clear();
  vi.stubGlobal("fetch", (req: Request) =>
    new URL(req.url).pathname.endsWith("/v1/watchlists") ? Promise.resolve(jsonResponse(watchlists)) : fetchMock(req));
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

describe("Day Trader pre-market", () => {
  it("explains an empty Top movers list before pre-market trades start", async () => {
    fetchMock.mockImplementation(async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      const source = new URL(req.url).searchParams.get("source") ?? "watchlist";
      return jsonResponse({ state: "premarket", source, symbols: [], missing: 0, as_of: new Date().toISOString(), rows: [] });
    });
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await userEvent.selectOptions(screen.getByLabelText("Symbols"), "movers");
    expect(await screen.findByText("No pre-market gappers yet today.")).toBeInTheDocument();
    expect(screen.getByText(/about 8:00 AM ET/)).toBeInTheDocument();
    expect(screen.queryByText("No symbols in this source.")).not.toBeInTheDocument();
  });
});

describe("Day Trader watchlists", () => {
  it("keeps one My watchlist entry and picks the list in a second menu, default first", async () => {
    const asked: string[] = [];
    watchlists = [{ id: 3, name: "Swing ideas", is_default: false, symbol_count: 1 }, { id: 7, name: "Core", is_default: true, symbol_count: 1 }];
    fetchMock.mockImplementation(async (req: Request) => {
      const url = new URL(req.url);
      if (url.pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      const wid = url.searchParams.get("watchlist_id");
      asked.push(`${url.searchParams.get("source")}:${wid ?? ""}`);
      const t = wid === "3" ? "SWNG" : url.searchParams.get("source") === "watchlist" ? "CORE" : "MOVR";
      return jsonResponse({ state: "closed", source: url.searchParams.get("source"), symbols: [t], missing: 0, as_of: new Date().toISOString(),
        rows: [row({ ticker: t, day_trade_score: 50, last: 10 })] });
    });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    const menu = screen.getByLabelText("Symbols");
    await waitFor(() => expect(tickers()).toEqual(["MOVR"]));
    expect(within(menu).getByRole("option", { name: "My watchlist" })).toBeInTheDocument();
    expect(within(menu).queryByRole("option", { name: "Swing ideas" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Watchlist")).not.toBeInTheDocument();
    await u.selectOptions(menu, "My watchlist");
    await waitFor(() => expect(tickers()).toEqual(["CORE"]));
    const lists = screen.getByLabelText("Watchlist");
    expect(within(lists).getAllByRole("option").map((o) => o.textContent)).toEqual(["Core", "Swing ideas"]);
    expect(lists).toHaveValue("7");
    await u.selectOptions(lists, "Swing ideas");
    await waitFor(() => expect(tickers()).toEqual(["SWNG"]));
    expect(asked).toEqual(expect.arrayContaining(["watchlist:", "watchlist:3"]));
    await u.selectOptions(menu, "movers");
    expect(screen.queryByLabelText("Watchlist")).not.toBeInTheDocument();
  });

  it("orders watchlists default first, then by name", () => {
    const l = [{ id: 1, name: "B", is_default: false }, { id: 2, name: "A", is_default: false }, { id: 3, name: "Z", is_default: true }];
    expect(orderLists(l).map((w) => w.id)).toEqual([3, 2, 1]);
    expect(orderLists(null)).toEqual([]);
  });
});

describe("Day Trader waiting rows", () => {
  it("shows Waiting, not a score, for names with no trade yet today", async () => {
    fetchMock.mockImplementation(async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      return jsonResponse({ state: "premarket", source: "premarket", symbols: ["XP", "AAA"], missing: 0, as_of: new Date().toISOString(), rows: [
        row({ ticker: "AAA", day_trade_score: 64, last: 10, chg_pct: 3, traded_today: true }),
        row({ ticker: "XP", day_trade_score: null, dt_quality: "insufficient", dt_direction: "neutral", last: 31.38, chg_pct: null, traded_today: false }),
      ] });
    });
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await waitFor(() => expect(tickers()).toEqual(["AAA", "XP"]));
    const xp = screen.getAllByRole("row").find((r) => within(r).queryByRole("link", { name: "XP" }))!;
    expect(within(xp).getByText("Waiting")).toBeInTheDocument();
    expect(within(xp).queryByText("Strong")).not.toBeInTheDocument();
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
      expect(screen.getByRole("status")).toHaveTextContent(/Live · checked \d+s ago/);
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
  it("backs off after a failed refresh instead of retrying nonstop, and keeps the last quotes", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let n = 0;
    fetchMock.mockImplementation(async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      n += 1;
      if (n > 1) return jsonResponse({ detail: "down" }, 503);
      return jsonResponse({ state: "open", source: "movers", symbols: ["AAA"], missing: 0, as_of: new Date().toISOString(),
        rows: [row({ ticker: "AAA", day_trade_score: 80 })] });
    });
    try {
      render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
      await waitFor(() => expect(tickers()).toEqual(["AAA"]));
      await vi.advanceTimersByTimeAsync(DT_REFRESH_MS + 50);
      await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/Couldn.t refresh, retrying/));
      const failed = n;
      await vi.advanceTimersByTimeAsync(DT_RETRY_MS - 1000);
      expect(n).toBe(failed);
      expect(tickers()).toEqual(["AAA"]);
      await vi.advanceTimersByTimeAsync(2000);
      await waitFor(() => expect(n).toBe(failed + 1));
    } finally {
      vi.useRealTimers();
    }
  });

  it("times refreshes from this device's clock, so a server as_of in the past doesn't refresh nonstop", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let n = 0;
    fetchMock.mockImplementation(async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      n += 1;
      return jsonResponse({ state: "open", source: "movers", symbols: ["AAA"], missing: 0, as_of: new Date(Date.now() - 60_000).toISOString(),
        rows: [row({ ticker: "AAA", day_trade_score: 80 })] });
    });
    try {
      render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
      await waitFor(() => expect(tickers()).toEqual(["AAA"]));
      expect(screen.getByRole("status")).toHaveTextContent(/checked [0-2]s ago/);
      await vi.advanceTimersByTimeAsync(DT_REFRESH_MS / 2);
      expect(n).toBe(1);
      await vi.advanceTimersByTimeAsync(DT_REFRESH_MS);
      await waitFor(() => expect(n).toBe(2));
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("Day Trader earnings tag", () => {
  it("labels today, tomorrow and later days, and nothing without a date", () => {
    expect(earningsLabel({ earnings_days: 0, earnings_date: "2026-11-10" })).toBe("Earnings today");
    expect(earningsLabel({ earnings_days: 1, earnings_date: "2026-11-11" })).toBe("Earnings tomorrow");
    expect(earningsLabel({ earnings_days: 3, earnings_date: "2026-11-13" })).toBe("Earnings Fri");
    expect(earningsLabel({ earnings_days: null, earnings_date: null })).toBeNull();
    expect(earningsLabel({})).toBeNull();
  });

  it("shows the tag on the row and explains it with the score", async () => {
    fetchMock.mockImplementation(async (req: Request) => {
      if (new URL(req.url).pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      return jsonResponse({ state: "closed", source: "movers", symbols: ["NVDA", "AAA"], missing: 0, as_of: new Date().toISOString(), rows: [
        row({ ticker: "NVDA", day_trade_score: 70, last: 190, earnings_days: 0, earnings_date: "2026-11-10", earnings_time: "amc" }),
        row({ ticker: "AAA", day_trade_score: 60, last: 10 }),
      ] });
    });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await waitFor(() => expect(tickers()).toEqual(["NVDA", "AAA"]));
    const nvda = screen.getAllByRole("row").find((r) => within(r).queryByRole("link", { name: "NVDA" }))!;
    expect(within(nvda).getByText("Earnings today · AMC")).toBeInTheDocument();
    const aaa = screen.getAllByRole("row").find((r) => within(r).queryByRole("link", { name: "AAA" }))!;
    expect(within(aaa).queryByText(/Earnings/)).not.toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Why NVDA scores 70" }));
    expect(screen.getByText(/Reports 2026-11-10 after the close/)).toBeInTheDocument();
  });
});

describe("Day Trader typed tickers", () => {
  it("parses commas and spaces, drops junk and duplicates, and caps the list", () => {
    expect(parseTickers("nvda, tsla  $pltr;NVDA brk.b 123 !!")).toEqual(["NVDA", "TSLA", "PLTR", "BRK.B"]);
    expect(parseTickers(Array.from({ length: 80 }, (_, i) => `T${i}`).join(" "))).toHaveLength(MAX_CUSTOM);
  });

  it("asks for tickers, then checks only those", async () => {
    const asked: string[] = [];
    fetchMock.mockImplementation(async (req: Request) => {
      const url = new URL(req.url);
      if (url.pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      asked.push(`${url.searchParams.get("source")}:${url.searchParams.get("symbols") ?? ""}`);
      const syms = (url.searchParams.get("symbols") ?? "MOVR").split(",");
      return jsonResponse({ state: "closed", source: url.searchParams.get("source"), symbols: syms, missing: 0, as_of: new Date().toISOString(),
        rows: syms.map((t, i) => row({ ticker: t, day_trade_score: 60 - i, last: 10 })) });
    });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await waitFor(() => expect(tickers()).toEqual(["MOVR"]));
    await u.selectOptions(screen.getByLabelText("Symbols"), "custom");
    expect(screen.getByText("Type the tickers you want to watch.")).toBeInTheDocument();
    const before = asked.length;
    await u.type(screen.getByLabelText(/^Tickers/), "nvda tsla");
    await u.click(screen.getByRole("button", { name: "Show" }));
    await waitFor(() => expect(tickers()).toEqual(["NVDA", "TSLA"]));
    expect(asked.slice(before)).toEqual(["custom:NVDA,TSLA"]);
    expect(screen.getByLabelText(/^Tickers/)).toHaveValue("NVDA, TSLA");
  });
});

describe("Day Trader remembers your setup", () => {
  it("reopens with the saved source, tickers, filters and sort", async () => {
    const asked: string[] = [];
    fetchMock.mockImplementation(async (req: Request) => {
      const url = new URL(req.url);
      if (url.pathname.endsWith("/sparklines")) return jsonResponse({ checked: [], series: {} });
      asked.push(`${url.searchParams.get("source")}:${url.searchParams.get("symbols") ?? ""}`);
      return jsonResponse({ state: "closed", source: "custom", symbols: ["AMD", "INTC"], missing: 0, as_of: new Date().toISOString(),
        rows: [row({ ticker: "AMD", day_trade_score: 80, last: 150 }), row({ ticker: "INTC", day_trade_score: 40, last: 20, dt_direction: "bearish" })] });
    });
    const u = userEvent.setup();
    const first = render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await waitFor(() => expect(tickers().length).toBeGreaterThan(0));
    await u.selectOptions(screen.getByLabelText("Symbols"), "custom");
    await u.type(screen.getByLabelText(/^Tickers/), "amd,intc");
    await u.click(screen.getByRole("button", { name: "Show" }));
    await waitFor(() => expect(tickers()).toEqual(["AMD", "INTC"]));
    await u.click(screen.getByRole("button", { name: /^Ticker/ }));
    await waitFor(() => expect(tickers()).toEqual(["AMD", "INTC"]));
    first.unmount();
    const saved = readPrefs(localStorage.getItem("hsf.dayTrader.v1"));
    expect(saved.source).toBe("custom");
    expect(saved.custom).toEqual(["AMD", "INTC"]);
    expect(saved.sort).toEqual({ key: "ticker", desc: false });
    asked.length = 0;
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    expect(screen.getByLabelText("Symbols")).toHaveValue("custom");
    await waitFor(() => expect(tickers()).toEqual(["AMD", "INTC"]));
    expect(asked[0]).toBe("custom:AMD,INTC");
  });

  it("falls back to defaults for broken or unknown saved values", () => {
    expect(readPrefs(null)).toEqual(DEFAULT_PREFS);
    expect(readPrefs("not json")).toEqual(DEFAULT_PREFS);
    expect(readPrefs("[1,2]")).toEqual(DEFAULT_PREFS);
    const p = readPrefs(JSON.stringify({ source: "nope", listId: -3, custom: ["aapl", 5], filters: { side: "up", minRvol: 2, minPrice: "x" }, sort: { key: "bogus" } }));
    expect(p).toEqual({ ...DEFAULT_PREFS, custom: ["AAPL"], filters: { ...NO_FILTERS, minRvol: 2 } });
    // "Type your own tickers" with no tickers opens on Top movers instead of an empty page.
    expect(readPrefs(JSON.stringify({ source: "custom", custom: [] })).source).toBe("movers");
  });
});
