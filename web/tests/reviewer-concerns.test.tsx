// Answers to the outside review: market condition badge, public track record, last week's
// results in the recap, the execution-aware trade plan and the public explanations.
import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { samplePlan } from "@/features/Demo";
import { MarketConditionBadge } from "@/features/MarketCondition";
import { PublicTrackRecordView } from "@/features/PublicTrackRecord";
import { StockView } from "@/features/StockView";
import { WeeklyRecapView } from "@/features/WeeklyRecap";
import { SAMPLE_SETUPS } from "@/lib/sampleResearch";
import { marketingRoute } from "@/lib/siteHosts";
import { PUBLIC_PAGES } from "@/lib/seo";

import { stockDetail } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/",
}));

let handler: (path: string) => unknown = () => ({ detail: "Not Found" });
beforeEach(() => {
  handler = () => ({ detail: "Not Found" });
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost")) : input;
    const path = new URL(r.url).pathname.replace(/^\/api\/(hsf|public)/, "");
    const out = handler(path);
    return out === undefined ? jsonResponse({ detail: "Not Found" }, 404) : jsonResponse(out);
  }));
});
afterEach(() => vi.unstubAllGlobals());

const choppy = {
  available: true, condition: "choppy", label: "Choppy", advice: "The market has no clear trend. Breakouts fail more often in choppy markets.",
  why: "S&P 500 (SPY) is 0.4% below its 20-day average and 1.2% above its 50-day average, with 20-day volatility of 14% a year.",
  as_of: "2026-11-18", rule: "spy-sma20-50-vol20-v1",
};

describe("market condition badge", () => {
  it("shows the label, what it means and the rule behind it", async () => {
    handler = (p) => (p === "/v1/market/condition" ? choppy : undefined);
    render(<MarketConditionBadge />);
    const note = await screen.findByRole("note", { name: "Market condition: Choppy" });
    expect(note).toHaveTextContent("Breakouts fail more often");
    expect(note).toHaveTextContent("20-day average");
    expect(within(note).getByRole("link", { name: "More on market conditions" })).toHaveAttribute("href", "/how-hsf-works#method-market");
  });

  it("renders nothing when the condition can't be computed", async () => {
    handler = (p) => (p === "/v1/market/condition" ? { available: false, rule: "x" } : undefined);
    const { container } = render(<MarketConditionBadge />);
    await new Promise((r) => setTimeout(r, 20));
    expect(container).toBeEmptyDOMElement();
  });
});

const record = {
  horizon_days: 5, top_per_day: 10, min_score: 40, min_days: 20, verdict: "no_edge_yet",
  verdict_text: "So far these results don't show a proven edge over the S&P 500.",
  summary: { picks: 420, days: 42, pending: 50, unavailable: 3, first_day: "2026-09-14", last_day: "2026-11-11",
    avg_return: 0.004, median_return: 0.001, avg_spy: 0.003, avg_excess: 0.001, median_excess: -0.002,
    beat_rate: 0.48, up_rate: 0.52, excess_ci95: { low: -0.006, high: 0.008 } },
  by_condition: [
    { condition: "trending_up", label: "Trending up", picks: 300, days: 30, avg_return: 0.006, avg_excess: 0.002, median_excess: 0, beat_rate: 0.5 },
    { condition: "choppy", label: "Choppy", picks: 40, days: 4, avg_return: -0.01, avg_excess: -0.008, median_excess: -0.01, beat_rate: 0.4 },
  ],
  recent: [
    { ticker: "WIN", day: "2026-11-11", score: 80, setup: "breakout", status: "matured", return_5d: 0.12, spy_5d: 0.01, excess_5d: 0.11, condition: "trending_up" },
    { ticker: "LOSE", day: "2026-11-11", score: 75, setup: "gapper", status: "matured", return_5d: -0.09, spy_5d: 0.01, excess_5d: -0.1, condition: "trending_up" },
  ],
  generated_at: "2026-11-19T18:00:00Z",
};

describe("public track record (/results)", () => {
  it("states the verdict honestly and lists losers next to winners", () => {
    render(<PublicTrackRecordView record={record as never} />);
    expect(screen.getByRole("heading", { name: "No proven edge yet" })).toBeInTheDocument();
    expect(screen.getByText(/don't show a proven edge/)).toBeInTheDocument();
    expect(screen.getByText("likely range -0.60% to +0.80%")).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /LOSE/ })).toHaveTextContent("-9.0%");
    expect(screen.getByRole("row", { name: /WIN/ })).toHaveTextContent("+12.0%");
    expect(screen.getByText(/50 newer picks are still inside their 5 days/)).toBeInTheDocument();
    expect(screen.getByText(/No trading costs, slippage, stops or targets/)).toBeInTheDocument();
    // A condition with fewer than 5 days is muted, not hidden.
    expect(screen.getByRole("row", { name: /Choppy/ })).toHaveClass("row-muted");
  });

  it("says so when the API can't answer, with no numbers", () => {
    render(<PublicTrackRecordView record={null} />);
    expect(screen.getByText(/couldn't load just now/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/%/);
  });

  it("is a public page on the marketing host and in the sitemap", () => {
    expect(marketingRoute("/results")).toBe("serve");
    expect(PUBLIC_PAGES).toContain("/results");
  });
});

describe("weekly recap: last week's results", () => {
  it("shows the average against the S&P 500 with the best and worst three", () => {
    const recap = { week: "2026-W46", start: "2026-11-09", end: "2026-11-13", built_at: "2026-11-14T05:00:00Z", scan_days: 5, ranked_names: 0, ranked: [],
      earnings: null,
      last_week: { start: "2026-11-02", end: "2026-11-06", picks: 48, pending: 0, avg_return: 0.006, avg_spy: 0.01, avg_excess: -0.004, beat_rate: 0.44, up_rate: 0.55,
        best: [{ ticker: "UPP", day: "2026-11-03", score: 70, return_5d: 0.2, spy_5d: 0.01, excess_5d: 0.19 }],
        worst: [{ ticker: "DWN", day: "2026-11-04", score: 66, return_5d: -0.15, spy_5d: 0.01, excess_5d: -0.16 }] } };
    render(<WeeklyRecapView recap={recap as never} weeks={[]} />);
    const card = screen.getByRole("heading", { name: "How last week's top picks did" }).closest("section")!;
    expect(card).toHaveTextContent("44% beat the S&P 500");
    expect(within(card).getByRole("row", { name: /DWN/ })).toHaveTextContent("-15.0%");
    expect(within(card).getByRole("link", { name: /every pick on the track record/ })).toHaveAttribute("href", "/results");
  });
});

describe("trade plan: live entry and execution warnings", () => {
  it("shows the live entry, trigger, don't-chase line, liquidity and warnings, with trigger and stop alerts", async () => {
    const plan = { ticker: "AAA", entry: 13.1, entry_source: "live", scan_price: 12.5, live_price: 13.1, moved_since_scan_pct: 4.8,
      stop: 12.58, stop_pct: 4, targets: [13.89, 14.67], target_r: [1.5, 3], risk_per_share: 0.52, shares: 192, risk_budget: 100,
      trigger: 12.8, chase_limit: 12.75, liquidity: { dollar_volume_20d: 3_200_000, spread_pct: 0.8, position_value: 2515, position_pct_of_volume: 0.08 },
      warnings: ["Price is 4.8% above the scan price, past the don't-chase line (12.75).", "Thinly traded: about $3.2M a day changes hands."] };
    handler = (p) => (p.endsWith("/plan") ? plan : p === "/v1/market/condition" ? choppy : undefined);
    render(<StockView s={stockDetail()} premium={false} pro />);
    expect(await screen.findByText("Entry (live price)")).toBeInTheDocument();
    expect(screen.getByText(/scan \$12\.50, \+4\.8%/)).toBeInTheDocument();
    expect(screen.getByText("Don't chase above")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Before you trade" })).toHaveTextContent("Thinly traded");
    expect(screen.getByText("$3.2M")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Alert me at the trigger" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Alert me at the stop" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("note", { name: "Market condition: Choppy" })).toBeInTheDocument());
  });
});

describe("demo trade plan", () => {
  it("uses the real plan rules on the sample price", () => {
    const p = samplePlan(SAMPLE_SETUPS[0]!);
    expect(p.entry).toBe(188.4);
    expect(p.stop).toBeCloseTo(188.4 * 0.96);
    expect(p.chase).toBeCloseTo(188.4 * 1.02);
    expect(p.shares).toBe(Math.floor(100 / (188.4 * 0.04)));
  });
});
