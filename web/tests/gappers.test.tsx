import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GappersCard } from "@/features/GappersCard";
import { clearApiCache } from "@/hooks/useApi";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

const fetchMock = vi.fn();
beforeEach(() => { fetchMock.mockReset(); clearApiCache(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

const now = new Date().toISOString();
const row = (ticker: string, pct: number, pm_volume: number | null = 12_500) => ({
  ticker, last: 100 + pct, previous_close: 100, pct, pm_volume, pm_dollar_volume: null, pm_vwap: null,
  pm_high: null, pm_low: null, trade_ts: now,
});
const card = (plan: "basic" | "pro" = "pro") => <SessionProvider initialMe={me(plan)}><GappersCard /></SessionProvider>;

describe("Pre-market gappers", () => {
  it("lists today's gappers with their move and pre-market volume", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ active: true, as_of: now, feed: "iex", rows: [row("UPP", 15), row("DWN", -8.5, null)] }));
    render(card());
    expect(await screen.findByRole("link", { name: "UPP" })).toHaveAttribute("href", "/stocks/UPP");
    expect(screen.getByText("+15.00%")).toBeInTheDocument();
    expect(screen.getByText("-8.50%")).toBeInTheDocument();
    expect(screen.getAllByText(/ sh$/)).toHaveLength(1);
    expect(screen.getByText(/Volume is from the IEX feed/)).toBeInTheDocument();
    expect(new URL((fetchMock.mock.calls[0]![0] as Request).url).searchParams.get("limit")).toBe("25");
  });

  it("says when nothing has gapped yet, and hides once pre-market is over", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ active: true, as_of: now, feed: "sip", rows: [] }));
    const { unmount } = render(card());
    expect(await screen.findByText("No pre-market gappers yet today.")).toBeInTheDocument();
    expect(screen.queryByText(/IEX feed/)).not.toBeInTheDocument();
    unmount();
    clearApiCache();
    fetchMock.mockResolvedValue(jsonResponse({ active: false, as_of: now, feed: "sip", rows: [] }));
    const { container } = render(card());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });

  it("locks below Pro without asking the API, and reports a failure", async () => {
    const { unmount } = render(card("basic"));
    expect(screen.getByText("Pre-market gappers are part of Pro")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    unmount();
    fetchMock.mockResolvedValue(jsonResponse({ detail: "down" }, 503));
    render(card());
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Pre-market gappers couldn't load"));
  });
});
