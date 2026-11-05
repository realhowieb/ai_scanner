import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WatchlistPerformanceCard } from "@/features/WatchlistPerformance";
import { clearApiCache } from "@/hooks/useApi";
import { jsonResponse } from "./helpers";

const performance = { as_of: "2026-11-05", benchmark: "SPY", summary: null, items: [
  { ticker: "ABC", base_date: "2026-10-01", base_price: 10, last: 11, change_pct: 10, vs_benchmark: 8 },
] };
const fetchMock = vi.fn();
beforeEach(() => { clearApiCache(); fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

describe("Watchlist performance loading and recovery", () => {
  it("shows a load failure and retries successfully", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Service unavailable" }, 503))
      .mockResolvedValueOnce(jsonResponse(performance));
    render(<WatchlistPerformanceCard id={4} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load watchlist performance right now");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("shows missing history separately from a failed request", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ ...performance, items: [] }));
    render(<WatchlistPerformanceCard id={4} />);
    expect(await screen.findByText(/No price history is available/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("keeps cached results visible when a refresh fails", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(performance))
      .mockResolvedValueOnce(jsonResponse({ detail: "Service unavailable" }, 503));
    const view = render(<WatchlistPerformanceCard id={4} version={0} />);
    await screen.findByRole("table");
    view.rerender(<WatchlistPerformanceCard id={4} version={1} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't load watchlist performance");
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText(/Showing the last loaded results/)).toBeInTheDocument();
  });

  it("shows loading instead of disappearing", () => {
    fetchMock.mockReturnValue(new Promise(() => {}));
    render(<WatchlistPerformanceCard id={4} />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });
});
