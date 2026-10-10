import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Schemas } from "@/api/client";
import { TodayView } from "@/features/TodayView";

const now = new Date().toISOString();
const base: Schemas["Today"] = {
  as_of: now, market: { phase: "premarket" }, errors: [],
  before_open: { scan_at: now, locked: false, movers: [{ ticker: "AAA", pct: 4.2, last: 10, score: 70 }] },
  top_setups: { state: "qualifying", threshold: 75, scan_at: now, setups: [{ ticker: "BBB", score: 81, primary_setup: "breakout", status: "STRONG", n_signals: 2, last: 20, chg_pct: 1, gap_pct: null, rvol: 2, prob: null }], also_ranked: [] },
  after_close: null,
  recap: { day: "2026-10-05", title: "Monday", scans: 3, premarket_scans: 1, postmarket_scans: 2, entered: ["CCC (65)"], left: [], standouts: [] },
};

describe("Today", () => {
  it("shows every section that loaded and a notice for the one that failed", () => {
    render(<TodayView data={{ ...base, recap: null, errors: [{ section: "recap", error: "OperationalError" }] }} />);
    expect(screen.getByRole("heading", { name: "BBB is the strongest setup on the board." })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Review BBB" })).toHaveAttribute("href", "/stocks/BBB");
    expect(screen.getByText("Pre-market")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "AAA" })).toHaveAttribute("href", "/stocks/AAA");
    expect(screen.getByRole("link", { name: "BBB" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Last session recap couldn't load");
    expect(screen.queryByText("OperationalError")).not.toBeInTheDocument();
  });

  it("locks pre-market movers below Pro with an upgrade button (server says locked)", () => {
    render(<TodayView data={{ ...base, before_open: { scan_at: now, locked: true, movers: [] } }} />);
    expect(screen.getByText("Before the open movers are part of Pro")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upgrade to Pro" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "AAA" })).not.toBeInTheDocument();
  });

  it("explains an empty scan and a scan with no qualifying setups", () => {
    const { rerender } = render(<TodayView data={{ ...base, top_setups: { state: "empty_scan", threshold: null, scan_at: null, setups: [], also_ranked: [] } }} />);
    expect(screen.getByText("No scan results yet.")).toBeInTheDocument();
    rerender(<TodayView data={{ ...base, top_setups: { state: "no_qualifying", threshold: 75, scan_at: now, setups: [], also_ranked: [] } }} />);
    expect(screen.getByText("No setup reached HSF 75 in the latest scan.")).toBeInTheDocument();
  });

  it("warns when the latest scan is stale", () => {
    const old = new Date(Date.now() - 9 * 3600_000).toISOString();
    render(<TodayView data={{ ...base, top_setups: { ...base.top_setups!, scan_at: old } }} />);
    expect(screen.getByRole("status")).toHaveTextContent(/may be out of date/);
  });

  it("trusts the API's schedule over the age rule", () => {
    const old = new Date(Date.now() - 18 * 3600_000).toISOString();
    const { rerender } = render(<TodayView data={{ ...base, market: { phase: "premarket", stale: false, latest_scan_at: old }, top_setups: { ...base.top_setups!, scan_at: old } }} />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();  // last night's final scan, nothing missed
    expect(screen.queryByText(/Stale/)).not.toBeInTheDocument();
    const missed = new Date(Date.now() - 2 * 3600_000).toISOString();
    rerender(<TodayView data={{ ...base, market: { phase: "open", stale: true, latest_scan_at: old, expected_scan_at: missed }, top_setups: { ...base.top_setups!, scan_at: old } }} />);
    expect(screen.getByRole("status")).toHaveTextContent(/Market data is delayed.*hasn't arrived yet/);
  });

  it("doesn't call last evening's after-hours scan stale", () => {
    const evening = new Date(Date.now() - 15 * 3600_000).toISOString();
    render(<TodayView data={{ ...base, after_close: { scan_at: evening, locked: false, movers: [] } }} />);
    expect(screen.getByText(/Updated 15h ago/)).not.toHaveTextContent("Stale");
  });

  it("fills Top setups with also-ranked names below the strong cutoff", () => {
    const also = { ...base.top_setups!.setups[0]!, ticker: "DDD", score: 53 };
    const { rerender } = render(<TodayView data={{ ...base, top_setups: { ...base.top_setups!, ranked_floor: 40, also_ranked: [also] } }} />);
    expect(screen.getByText("Also ranked")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "DDD" }).closest("tr")).toHaveClass("row-muted");
    expect(screen.getByText("Strong setups score HSF 75+. Also ranked: HSF 40 to 74.")).toBeInTheDocument();
    rerender(<TodayView data={{ ...base, top_setups: { state: "no_qualifying", threshold: 75, scan_at: now, setups: [], ranked_floor: 40, also_ranked: [also] } }} />);
    expect(screen.getByText(/No setup reached HSF 75 in the latest scan. These are the highest ranked names./)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "DDD" })).toBeInTheDocument();
  });

  it("mutes HSF badges below the ranked list", () => {
    render(<TodayView data={{ ...base, after_close: { scan_at: now, locked: false, movers: [{ ticker: "LEVI", pct: 7.4, last: 20.96, score: 3 }, { ticker: "EEE", pct: 5, last: 9, score: 62 }] } }} />);
    expect(screen.getByLabelText("HSF Score 3")).toHaveClass("score-weak");
    expect(screen.getByLabelText("HSF Score 62")).not.toHaveClass("score-weak");
  });

  it("links recap chips to the stock and shows standouts not already on Top setups", () => {
    render(<TodayView data={{ ...base, recap: { ...base.recap!, left: ["MYRG (65)"], standouts: [{ ticker: "BBB", score: 81, setup: null }, { ticker: "FFF", score: 77, setup: null }] } }} />);
    expect(screen.getByRole("link", { name: "CCC, HSF 65" })).toHaveAttribute("href", "/stocks/CCC");
    expect(screen.getByRole("link", { name: "MYRG, HSF 65" })).toHaveAttribute("href", "/stocks/MYRG");
    expect(screen.getByText("Left the ranked list, with their score from the day's first scan")).toBeInTheDocument();
    expect(screen.getByText("Strongest in the last scan")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "FFF, HSF 77" })).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /^BBB/ })).toHaveLength(1);  // already in Top setups
  });

  it("hides sessions the API leaves out (outside their window)", () => {
    render(<TodayView data={{ ...base, before_open: null, after_close: null }} />);
    expect(screen.queryByText("Before the open")).not.toBeInTheDocument();
    expect(screen.queryByText("After the close")).not.toBeInTheDocument();
  });
});

describe("Today market snapshot", () => {
  const snapshot: Schemas["Snapshot"] = {
    universe_symbols: 11553, ranked_count: 100, status: { level: "ok", label: "Operational" },
    indices: [{ symbol: "SPY", label: "S&P 500", last: 777.3, chg_pct: -0.23 }, { symbol: "QQQ", label: "Nasdaq 100", last: 757.84, chg_pct: null }],
    top_gainer: { ticker: "BSP", chg_pct: 20.36, last: 4.1, volume: 2e6 },
    most_active: { ticker: "AMD", chg_pct: 1.2, last: 160, volume: 379.1e6 },
  };

  it("shows the status strip and the four tiles", () => {
    render(<TodayView data={{ ...base, market: { phase: "closed", latest_scan_at: now }, snapshot }} />);
    const strip = screen.getByRole("region", { name: "Market data status" });
    expect(strip).toHaveTextContent("Universe 11,553 tradable stocks");
    expect(strip).toHaveTextContent("100 ranked setups");
    expect(strip).toHaveTextContent("System: Operational");
    expect(strip).toHaveTextContent(/Last scan .* ET \(0m ago\)/);
    expect(screen.getByText("777.30")).toBeInTheDocument();
    expect(screen.getByText("-0.23%")).toHaveClass("down");
    expect(screen.getByText("757.84").nextSibling).toHaveTextContent("—");  // no previous close, no made-up change
    expect(screen.getByRole("link", { name: "BSP" })).toHaveAttribute("href", "/stocks/BSP");
    expect(screen.getByText("+20.36%")).toHaveClass("up");
    expect(screen.getByText("379.1M shares")).toBeInTheDocument();
  });

  it("works with an API that doesn't send a snapshot yet, and when it fails", () => {
    const { rerender } = render(<TodayView data={{ ...base, market: { phase: "open", latest_scan_at: now } }} />);
    expect(screen.getByRole("region", { name: "Market data status" })).toHaveTextContent(/Last scan/);
    expect(screen.queryByText(/System:/)).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Market snapshot" })).not.toBeInTheDocument();
    rerender(<TodayView data={{ ...base, errors: [{ section: "snapshot", error: "RuntimeError" }] }} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Market snapshot couldn't load");
    expect(screen.getByRole("region", { name: "Market data status" })).toHaveTextContent("Latest market scan unavailable");
  });
});

describe("Today signed-in sections", () => {
  const mine: Schemas["TodayPersonal"] = {
    errors: [],
    new_since: { marker: "11:10", baseline_scan_at: now, tickers: [{ ticker: "TER", score: 76 }, { ticker: "ZZZ", score: null }], total: 2 },
    watchlist: { watchlist_id: 1, name: "Main", summary: { tracked: 40, needs_attention: 1, strengthening: 2, fading: 0 },
      in_scan: [{ ticker: "GNRC", score: 52 }, { ticker: "MPC", score: 17 }], missing: ["AEMD", "NVDA"] },
  };

  it("lists new names and the watchlist against the latest scan", () => {
    render(<TodayView data={base} mine={mine} />);
    expect(screen.getByRole("heading", { name: "New since your last visit" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "TER, HSF 76" })).toHaveAttribute("href", "/stocks/TER");
    expect(screen.getByRole("link", { name: "ZZZ" })).toBeInTheDocument();
    expect(screen.getByText("40 watched · 1 need attention · 2 strengthening · 0 fading")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "GNRC" })).toHaveAttribute("href", "/stocks/GNRC");
    expect(screen.getByLabelText("HSF Score 17")).toHaveClass("score-weak");
    expect(screen.getByText("Not in the latest scan: AEMD, NVDA")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Main" })).toHaveAttribute("href", "/watchlists");
  });

  it("explains a first visit and an empty watchlist", () => {
    render(<TodayView data={base} mine={{ errors: [], new_since: { marker: "11:", tickers: [], total: 0 },
      watchlist: { watchlist_id: null, in_scan: [], missing: [] } }} />);
    expect(screen.getByText(/Nothing new since you last looked/)).toBeInTheDocument();
    expect(screen.getByText("Your watchlist is empty.")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Choose your workflow" })).not.toBeInTheDocument();
  });

  it("guides a true first run toward one successful action", () => {
    render(<TodayView data={base} mine={{ errors: [], new_since: { tickers: [], total: 0 },
      watchlist: { watchlist_id: null, in_scan: [], missing: [] } }} />);
    const guide = screen.getByRole("region", { name: "Choose your workflow" });
    expect(guide).toBeInTheDocument();
    expect(within(guide).getByRole("link", { name: "Open Scanner" })).toHaveAttribute("href", "/scanner");
    expect(within(guide).getByRole("link", { name: "Review BBB" })).toHaveAttribute("href", "/stocks/BBB");
    expect(within(guide).getByRole("link", { name: "Add tickers" })).toHaveAttribute("href", "/watchlists");
    expect(within(guide).getByRole("link", { name: "Open Day Trader" })).toHaveAttribute("href", "/day-trader");
    expect(within(guide).getByText(/review one ticker, save it to a watchlist, then create one alert/i)).toBeInTheDocument();
  });

  it("shows a notice when the signed-in sections fail, and nothing while they load", () => {
    const { rerender } = render(<TodayView data={base} />);
    expect(screen.queryByRole("heading", { name: "Your watchlist" })).not.toBeInTheDocument();
    rerender(<TodayView data={{ ...base, recap: null }} mineFailed />);
    expect(screen.getAllByRole("alert").map((a) => a.textContent)).toEqual([
      "New since your last visit couldn't load right now. The rest of the page is current.",
      "Your watchlist couldn't load right now. The rest of the page is current.",
    ]);
  });
});
