import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WatchlistsView, attentionReasons } from "@/features/WatchlistsView";
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

const rows = () => within(document.querySelector(".wl-items") as HTMLElement).getAllByRole("listitem");

describe("watchlist summary chips", () => {
  it("attention reasons: Fading and/or a score drop", () => {
    expect(attentionReasons(true, -6)).toEqual(["Fading", "Score −6 since last scan"]);
    expect(attentionReasons(false, 3)).toEqual([]);
    expect(attentionReasons(undefined, null)).toEqual([]);
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
});
