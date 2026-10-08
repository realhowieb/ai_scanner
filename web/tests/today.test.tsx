import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Schemas } from "@/api/client";
import { TodayView } from "@/features/TodayView";

const now = new Date().toISOString();
const base: Schemas["Today"] = {
  as_of: now, market: { phase: "premarket" }, errors: [],
  before_open: { scan_at: now, locked: false, movers: [{ ticker: "AAA", pct: 4.2, last: 10, score: 70 }] },
  top_setups: { state: "qualifying", threshold: 75, scan_at: now, setups: [{ ticker: "BBB", score: 81, primary_setup: "breakout", status: "STRONG", n_signals: 2, last: 20, chg_pct: 1, gap_pct: null, rvol: 2, prob: null }] },
  after_close: null,
  recap: { day: "2026-10-05", title: "Monday", scans: 3, premarket_scans: 1, postmarket_scans: 2, entered: ["CCC (65)"], left: [], standouts: [] },
};

describe("Today", () => {
  it("shows every section that loaded and a notice for the one that failed", () => {
    render(<TodayView data={{ ...base, recap: null, errors: [{ section: "recap", error: "OperationalError" }] }} />);
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
    const { rerender } = render(<TodayView data={{ ...base, top_setups: { state: "empty_scan", threshold: null, scan_at: null, setups: [] } }} />);
    expect(screen.getByText("No scan results yet.")).toBeInTheDocument();
    rerender(<TodayView data={{ ...base, top_setups: { state: "no_qualifying", threshold: 75, scan_at: now, setups: [] } }} />);
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

  it("hides sessions the API leaves out (outside their window)", () => {
    render(<TodayView data={{ ...base, before_open: null, after_close: null }} />);
    expect(screen.queryByText("Before the open")).not.toBeInTheDocument();
    expect(screen.queryByText("After the close")).not.toBeInTheDocument();
  });
});
