import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WatchlistsView, attentionReasons, fadingVsSpy } from "@/features/WatchlistsView";
import { SessionProvider } from "@/session/SessionProvider";

import { fakeApi } from "./fakeApi";
import { me } from "./fixtures";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  usePathname: () => "/watchlists",
}));

let api: ReturnType<typeof fakeApi>;
beforeEach(() => {
  api = fakeApi({ scan: [{ ticker: "AAPL", score: 81, last: 201.5 }] });
  vi.stubGlobal("fetch", vi.fn((input: Request | string, init?: RequestInit) => api.fetch(input, init)));
});
afterEach(() => vi.unstubAllGlobals());

const rows = () => [...document.querySelectorAll(".wl-table tbody tr")];

describe("watchlist summary chips", () => {
  it("attention reasons: Fading and/or a score drop", () => {
    expect(attentionReasons(true, -6)).toEqual(["Fading", "Score −6 since last scan"]);
    expect(attentionReasons(false, 3)).toEqual([]);
    expect(attentionReasons(undefined, null)).toEqual([]);
  });

  it("Fading means trailing SPY by 2 points or more; the scan's flag only without SPY", () => {
    expect(fadingVsSpy(-3.5, -3.1, true)).toBe(false); // red with the market: not fading
    expect(fadingVsSpy(-5.2, -3.1, false)).toBe(true);
    expect(fadingVsSpy(-1.5, 0.6, false)).toBe(true);
    expect(fadingVsSpy(-3, null, true)).toBe(true);
    expect(fadingVsSpy(null, -1, false)).toBe(false);
  });

  it("clicking a chip filters the list to the tickers it counts; clicking again shows all", async () => {
    const u = userEvent.setup();
    api.seedList("Main", ["AAPL", "MSFT"], true);
    render(<SessionProvider initialMe={me("pro")}><WatchlistsView /></SessionProvider>);
    const scanChip = await screen.findByRole("button", { name: /^1 in latest scan/ });
    expect(screen.getByRole("button", { name: /^2 tracked/ })).toHaveAttribute("aria-pressed", "true");
    expect(rows()).toHaveLength(2);

    await u.click(scanChip);
    expect(scanChip).toHaveAttribute("aria-pressed", "true");
    expect(rows()).toHaveLength(1);
    expect(rows()[0]).toHaveTextContent("AAPL");
    expect(screen.getByText(/Showing 1 of 2: in latest scan/)).toBeInTheDocument();

    await u.click(scanChip);
    expect(rows()).toHaveLength(2);
    expect(screen.getByRole("button", { name: /^0 with notes/ })).toBeDisabled();
  });

  it("need attention counts stocks trailing SPY, not every red stock on a down day", async () => {
    const at = new Date().toISOString();
    api = fakeApi({ scanState: true, scan: [{ ticker: "AAPL", score: 81, last: 201.5 }, { ticker: "NVDA", score: 70, last: 120 }],
      quotes: { AAPL: { last: 199, chg_pct: -4, at }, NVDA: { last: 112, chg_pct: -6, at }, SPY: { last: 600, chg_pct: -3, at } } });
    api.seedList("Main", ["AAPL", "NVDA"], true);
    render(<SessionProvider initialMe={me("pro")}><WatchlistsView /></SessionProvider>);
    expect(await screen.findByRole("button", { name: /^1 need attention/ })).toBeInTheDocument();
  });
});
