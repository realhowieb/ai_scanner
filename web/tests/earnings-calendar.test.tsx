import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { EarningsCalendar } from "@/features/EarningsCalendar";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

const fetchMock = vi.fn();
beforeEach(() => { fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

const name = (ticker: string, extra: Record<string, unknown> = {}) => ({
  ticker, time: null, last: 100, chg_pct: 1, dollar_volume: 1e9, score: null, score_source: null,
  on_watchlist: false, logo_url: `https://example.test/${ticker}.png`, ...extra,
});
const week = {
  start: "2026-11-02", end: "2026-11-06", scan_at: null, per_day: 16,
  days: [
    { date: "2026-11-02", total: 100, shown: 2, names: [name("AAPL", { time: "amc", score: 62, score_source: "scan" }), name("PLTR", { time: "amc", on_watchlist: true })] },
    { date: "2026-11-03", total: 1, shown: 1, names: [name("SHOP", { time: "bmo" })] },
    { date: "2026-11-04", total: 0, shown: 0, names: [] },
    { date: "2026-11-05", total: 1, shown: 1, names: [name("NET")] },
    { date: "2026-11-06", total: 0, shown: 0, names: [] },
  ],
};

describe("Earnings week calendar", () => {
  it("shows five days of tiles with timing, scores, watchlist and the rest counted", async () => {
    fetchMock.mockResolvedValue(jsonResponse(week));
    render(<SessionProvider initialMe={me("pro")}><EarningsCalendar pro /></SessionProvider>);
    const aapl = await screen.findByRole("link", { name: "AAPL, after close, HSF Score 62" });
    expect(aapl).toHaveAttribute("href", "/stocks/AAPL");
    expect(screen.getByRole("link", { name: "PLTR, after close, on your watchlist" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "SHOP, before open" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "NET, time not announced" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Monday Nov 2" })).toHaveTextContent("100 reporting");
    expect(screen.getByText("+ 98 more")).toBeInTheDocument();
    expect(screen.getAllByText("No companies we track report this day.")).toHaveLength(2);
    expect(new URL((fetchMock.mock.calls[0]![0] as Request).url).searchParams.get("per_day")).toBe("16");
    // Logos load same-origin and fall back to the first letter when missing.
    const img = aapl.querySelector("img")!;
    expect(img.getAttribute("src")).toBe("/api/logo/AAPL");
    fireEvent.error(img);
    expect(aapl.querySelector("img")).toBeNull();
    expect(aapl).toHaveTextContent("A");
  });

  it("locks below Pro without asking the API", () => {
    render(<SessionProvider initialMe={me("basic")}><EarningsCalendar pro={false} /></SessionProvider>);
    expect(screen.getByText("The earnings calendar is part of Pro")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
