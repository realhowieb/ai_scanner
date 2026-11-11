import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import StockPage from "@/app/(app)/stocks/[ticker]/page";
import { WatchlistsView } from "@/features/WatchlistsView";
import { clearApiCache } from "@/hooks/useApi";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useParams: () => ({ ticker: "AAA" }), useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/watchlists", useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));
let paths: string[];
beforeEach(() => { clearApiCache(); paths = []; });
afterEach(() => vi.unstubAllGlobals());

it("starts stock news and filings while the stock detail is still loading", async () => {
  vi.stubGlobal("fetch", (request: Request) => {
    const path = new URL(request.url).pathname;
    paths.push(path);
    if (path.endsWith("/v1/stocks/AAA")) return new Promise(() => {});
    return Promise.resolve(jsonResponse({ available: false, names: {} }));
  });
  render(<SessionProvider initialMe={me("pro")}><StockPage /></SessionProvider>);
  await waitFor(() => {
    expect(paths.some((p) => p.endsWith("/v1/stocks/AAA/news"))).toBe(true);
    expect(paths.some((p) => p.endsWith("/v1/stocks/AAA/filings"))).toBe(true);
  });
  expect(screen.getByText(/Loading AAA/)).toBeInTheDocument();
});

it("starts performance while the watchlist detail is still loading", async () => {
  vi.stubGlobal("fetch", (request: Request) => {
    const path = new URL(request.url).pathname;
    paths.push(path);
    if (path.endsWith("/v1/watchlists")) return Promise.resolve(jsonResponse([
      { id: 4, name: "Main", is_default: true, symbol_count: 2 },
    ]));
    if (path.endsWith("/v1/watchlists/4")) return new Promise(() => {});
    if (path.endsWith("/performance")) return Promise.resolve(jsonResponse({ items: [], benchmark: "SPY" }));
    return Promise.resolve(jsonResponse({ items: [], changes: [], alerts: [], names: {} }));
  });
  render(<SessionProvider initialMe={me("pro")}><WatchlistsView /></SessionProvider>);
  await waitFor(() => expect(paths.some((p) => p.endsWith("/v1/watchlists/4/performance"))).toBe(true));
  expect(screen.getByText(/Loading watchlist/)).toBeInTheDocument();
});
