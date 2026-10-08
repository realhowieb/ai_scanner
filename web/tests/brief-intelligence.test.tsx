import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Schemas } from "@/api/client";
import { MarketPulse, OpportunityRadar, RankMovement } from "@/features/BriefIntelligence";

const brief: Schemas["Brief"] = {
  available: true, snapshot_time: "2026-10-07T14:00:00Z", phase: "regular", has_previous_snapshot: true,
  market: [{ label: "SPY", last: 600, chg_pct: -1 }, { label: "QQQ", last: 500, chg_pct: 2 }],
  breadth: { advancers: 3, decliners: 1 }, sectors: [{ sector: "Tech", chg_pct: -2 }, { sector: "Energy", chg_pct: 1 }],
  opportunities: [{ ticker: "AAA", score: 81, score_delta: 5, primary_setup: "Momentum" }, { ticker: "BBB", score: 70, score_delta: -3 }],
  prebreakout_picks: [], prebreakout_locked: true, golden_crosses: [], top_breakout_scores: [], earnings_today: [], gappers: [], gainers: [], losers: [],
};

describe("Brief intelligence", () => {
  it("renders actual indexes, scoped counts, breadth and sector extremes", () => {
    render(<MarketPulse b={brief} />);
    expect(screen.getByText("-1.00%")).toBeInTheDocument();
    expect(screen.getByText("+2.00%")).toBeInTheDocument();
    expect(screen.getByText(/75% advancing/)).toBeInTheDocument();
    expect(screen.getByText("HSF 80+ in Radar").nextSibling).toHaveTextContent("1");
    expect(screen.queryByText("PreBreakout picks shown")).not.toBeInTheDocument();
    expect(screen.getByText("Energy")).toBeInTheDocument();
    expect(screen.getByText(/Snapshot Oct 7/)).toBeInTheDocument();
  });
  it("omits unavailable breadth without hiding indexes", () => {
    render(<MarketPulse b={{ ...brief, breadth: null }} />);
    expect(screen.queryByRole("region", { name: "HSF snapshot breadth" })).not.toBeInTheDocument();
    expect(screen.getByText("SPY")).toBeInTheDocument();
  });
  it("retains API ordering, score movement and stock navigation", () => {
    render(<OpportunityRadar b={brief} />);
    expect(screen.getAllByRole("listitem")[0]).toHaveTextContent("AAA");
    expect(screen.getByText("Rising +5")).toBeInTheDocument();
    expect(screen.getByText("Falling -3")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "AAA" })).toHaveAttribute("href", "/stocks/AAA");
    expect(screen.getByText("Brief #1")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Rank improved/)).not.toBeInTheDocument();
  });
  it("does not infer missing score movement and explains empty Radar", () => {
    const view = render(<OpportunityRadar b={{ ...brief, opportunities: [{ ticker: "X", score: 80 }] }} />);
    expect(screen.queryByText(/Rising|Falling|Unchanged/)).not.toBeInTheDocument();
    view.rerender(<OpportunityRadar b={{ ...brief, opportunities: [] }} />);
    expect(screen.getByText("No ranked opportunities in this snapshot.")).toBeInTheDocument();
  });
  it("treats lower numeric rank as an improvement, with no invented baseline", () => {
    const view = render(<RankMovement rank={5} previous={20} />);
    expect(screen.getByLabelText("Rank improved by 15")).toHaveTextContent("↑15");
    view.rerender(<RankMovement rank={20} previous={5} />);
    expect(screen.getByLabelText("Rank fell by 15")).toHaveTextContent("↓15");
    view.rerender(<RankMovement rank={20} />);
    expect(view.container).toBeEmptyDOMElement();
  });
});
