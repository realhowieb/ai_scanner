import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { backLabel } from "@/components/AppShell";
import { StockView } from "@/features/StockView";

import { stockDetail } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/stocks/AAA",
}));

const fetchMock = vi.fn();
const urls = () => fetchMock.mock.calls.map((c) => new URL((c[0] as Request).url));
// The Historical card also asks /v1/outcomes/symbols/{ticker}; these tests are about the plan and AI
// note, so that call is answered here (404: an API without it) and never reaches fetchMock.
const outcomeCalls: string[] = [];
beforeEach(() => {
  fetchMock.mockReset();
  outcomeCalls.length = 0;
  vi.stubGlobal("fetch", (req: Request, ...rest: unknown[]) => {
    if (new URL(req.url).pathname.includes("/v1/outcomes/")) {
      outcomeCalls.push(req.url);
      return Promise.resolve(jsonResponse({ detail: "Not Found" }, 404));
    }
    return fetchMock(req, ...rest);
  });
});
afterEach(() => vi.unstubAllGlobals());

const plan = { ticker: "AAA", entry: 12.5, stop: 12.0, stop_pct: 4, targets: [13.25, 14.0], target_r: [1.5, 3], risk_per_share: 0.5, shares: 200, risk_budget: 100 };

describe("Stock detail: trade plan and AI note", () => {
  it("Pro: loads the API's plan, and asks again with the user's account size and risk", async () => {
    fetchMock.mockImplementation(async () => jsonResponse(plan));
    const u = userEvent.setup();
    render(<StockView s={stockDetail()} premium={false} pro />);
    expect(await screen.findByText("$12.00")).toBeInTheDocument();
    expect(screen.getByText("(−4.0%)")).toBeInTheDocument();
    expect(screen.getByText("(3R)")).toBeInTheDocument();
    expect(Object.fromEntries(urls()[0]!.searchParams)).toEqual({ account_size: "10000", risk_pct: "1" });
    await u.clear(screen.getByLabelText("Risk per trade (%)"));
    await u.type(screen.getByLabelText("Risk per trade (%)"), "12");
    expect(screen.getByRole("button", { name: "Update" })).toBeDisabled();
    await u.clear(screen.getByLabelText("Risk per trade (%)"));
    await u.type(screen.getByLabelText("Risk per trade (%)"), "2");
    await u.click(screen.getByRole("button", { name: "Update" }));
    await waitFor(() => expect(urls()).toHaveLength(2));
    expect(urls()[1]!.searchParams.get("risk_pct")).toBe("2");
  });

  it("Pro: a name the plan can't use shows the API's reason", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: "That ticker isn't in the latest market scan." }, 404));
    render(<StockView s={stockDetail()} premium={false} pro />);
    expect(await screen.findByText("That ticker isn't in the latest market scan.")).toBeInTheDocument();
  });

  it("Free: the plan is locked and never requested; names outside the scan get no plan card", () => {
    const { unmount } = render(<StockView s={stockDetail()} premium={false} />);
    expect(screen.getByText("Trade plans are part of Pro")).toBeInTheDocument();
    unmount();
    render(<StockView s={stockDetail({ in_latest_scan: false, from_history: true })} premium={false} pro />);
    expect(screen.queryByRole("heading", { name: "Trade plan" })).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(outcomeCalls).toEqual([]);            // Free: historical research is locked, so no evidence request
  });

  it("Premium: writes an AI note on request, and shows the daily limit message", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ ticker: "AAA", run_id: 1, snapshot_time: null, text: "AAA leads on volume." }))
      .mockResolvedValueOnce(jsonResponse({ detail: "You've used today's AI notes." }, 429));
    const u = userEvent.setup();
    render(<StockView s={stockDetail({ in_latest_scan: true, has_setup: false })} premium aiNotes />);
    await u.click(screen.getByRole("button", { name: "Write AI note" }));
    expect(await screen.findByText("AAA leads on volume.")).toBeInTheDocument();
    expect(urls()[0]!.pathname).toBe("/api/hsf/v1/ai/notes/AAA");
    await u.click(screen.getByRole("button", { name: "Write it again" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("You've used today's AI notes.");
  });

  it("goes back to where the user came from", () => {
    render(<StockView s={stockDetail()} premium={false} />);
    expect(screen.getByRole("link", { name: "← Back to Scanner" })).toHaveAttribute("href", "/scanner");
    expect(backLabel("/watchlists?id=3")).toBe("Back to Watchlists");
    expect(backLabel("/scanner/custom")).toBe("Back to Custom scan");
    expect(backLabel("/today")).toBe("Back to Today");
    expect(backLabel("/stocks/MSFT")).toBe("Back to the previous stock");
    expect(backLabel(null)).toBeNull();
  });
});
