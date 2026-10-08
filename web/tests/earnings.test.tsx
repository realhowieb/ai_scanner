import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { EarningsCard } from "@/features/EarningsCard";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

const fetchMock = vi.fn();
beforeEach(() => { fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

const url = (i: number) => new URL((fetchMock.mock.calls[i]![0] as Request).url);

describe("Earnings this week", () => {
  it("asks about the watchlist and top setups and lists who reports", async () => {
    fetchMock.mockResolvedValue(jsonResponse([
      { ticker: "MXL", earnings_date: "2026-10-08", days_until: 0, time: "amc" },
      { ticker: "TER", earnings_date: "2026-10-13", days_until: 5, time: null },
    ]));
    render(<SessionProvider initialMe={me("pro")}><EarningsCard tickers={["MXL", "TER"]} watched={new Set(["MXL"])} /></SessionProvider>);
    expect(await screen.findByRole("link", { name: "MXL" })).toHaveAttribute("href", "/stocks/MXL");
    expect(Object.fromEntries(url(0).searchParams)).toEqual({ days: "7", tickers: "MXL,TER" });
    expect(screen.getByText("Today")).toBeInTheDocument();
    expect(screen.getByText("After close")).toBeInTheDocument();
    expect(screen.getByText("Tue, Oct 13")).toBeInTheDocument();
    expect(screen.getByText("Time TBA")).toBeInTheDocument();
    expect(screen.getAllByText("Watchlist")).toHaveLength(1);
  });

  it("says when nobody reports, and waits for the names before asking", async () => {
    fetchMock.mockResolvedValue(jsonResponse([]));
    const { rerender } = render(<SessionProvider initialMe={me("pro")}><EarningsCard tickers={null} watched={new Set()} /></SessionProvider>);
    expect(fetchMock).not.toHaveBeenCalled();
    rerender(<SessionProvider initialMe={me("pro")}><EarningsCard tickers={["AAA"]} watched={new Set()} /></SessionProvider>);
    expect(await screen.findByText("No earnings this week for your watchlist or today's top setups.")).toBeInTheDocument();
  });

  it("locks below Pro without asking the API, and reports a failure", async () => {
    const { unmount } = render(<SessionProvider initialMe={me("basic")}><EarningsCard tickers={["AAA"]} watched={new Set()} /></SessionProvider>);
    expect(screen.getByText("Earnings timing is part of Pro")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    unmount();
    fetchMock.mockResolvedValue(jsonResponse({ detail: "down" }, 503));
    render(<SessionProvider initialMe={me("pro")}><EarningsCard tickers={["AAA"]} watched={new Set()} /></SessionProvider>);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Earnings this week couldn't load"));
  });
});
