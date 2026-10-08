import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppShell } from "@/components/AppShell";
import { DayTraderView } from "@/features/DayTraderView";
import { JournalView } from "@/features/JournalView";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

let path = "/scanner/history";
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => path,
}));

const fetchMock = vi.fn();
beforeEach(() => {
  path = "/scanner/history";
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

const APP = join(__dirname, "..", "src", "app");
/** A route exists when a page file serves it (route groups like (app) don't add a segment). */
function routeExists(href: string): boolean {
  const segs = href.split("?")[0]!.split("/").filter(Boolean);
  return [join(APP, ...segs, "page.tsx"), join(APP, "(app)", ...segs, "page.tsx")].some(existsSync);
}

describe("navigation", () => {
  it("every main-nav and menu link opens a real page, and the section is marked current", async () => {
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AppShell><p>body</p></AppShell></SessionProvider>);
    const nav = screen.getByRole("navigation", { name: "Main" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((l) => l.textContent)).toEqual(["Today", "Brief", "Scanner", "Stock Intelligence", "Day Trader", "Watchlists", "Alerts", "Track record"]);
    for (const l of links) expect(routeExists(l.getAttribute("href")!)).toBe(true);
    expect(within(nav).getByRole("link", { name: "Scanner" })).toHaveAttribute("aria-current", "page");
    await u.click(screen.getByRole("button", { name: /Account/ }));
    const menu = screen.getByRole("menu");
    for (const l of within(menu).getAllByRole("menuitem").filter((x) => x.tagName === "A" && x.getAttribute("href")!.startsWith("/"))) {
      expect(routeExists(l.getAttribute("href")!)).toBe(true);
    }
    expect(within(menu).getByRole("menuitem", { name: "Account & billing" })).toHaveAttribute("href", "/account");
    expect(routeExists("/how-hsf-works")).toBe(true);
  });

  it("the sign-in guard covers every signed-in page", () => {
    const guard = readFileSync(join(__dirname, "..", "src", "proxy.ts"), "utf8");
    for (const p of ["today", "brief", "scanner", "day-trader", "watchlists", "alerts", "track-record", "journal", "account", "stocks", "paper"]) {
      expect(guard).toContain(`"/${p}/:path*"`);
    }
  });
});

describe("Journal", () => {
  const trade = { id: 1, ticker: "AAA", open: true, shares: 10, entry_price: 10, exit_price: null, mark: 11, pnl: 10, pnl_pct: 10, entered_at: null, closed_at: null, source: null };

  it("Free reads the journal but can't log; Pro logs a trade and the list reloads", async () => {
    fetchMock.mockImplementation(async () => jsonResponse({ trades: [], stats: null }));
    const { unmount } = render(<SessionProvider initialMe={me("basic")}><JournalView /></SessionProvider>);
    expect(screen.getByText("Logging trades is part of Pro")).toBeInTheDocument();
    expect(await screen.findByText("No trades logged yet.")).toBeInTheDocument();
    unmount();
    let logged = false;
    fetchMock.mockImplementation(async (req: Request) => {
      if (req.method === "POST") { logged = true; return new Response(null, { status: 201 }); }
      return jsonResponse({ trades: logged ? [trade] : [], stats: { closed: 0, wins: 0, avg_return_pct: null } });
    });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><JournalView /></SessionProvider>);
    await u.type(screen.getByLabelText("Ticker"), "aaa");
    await u.type(screen.getByLabelText("Entry price ($)"), "10");
    await u.type(screen.getByLabelText("Shares"), "10");
    await u.click(screen.getByRole("button", { name: "Log trade" }));
    expect(await screen.findByRole("link", { name: "AAA" })).toBeInTheDocument();
    const post = fetchMock.mock.calls.map((c) => c[0] as Request).find((r) => r.method === "POST")!;
    expect(await post.clone().json()).toEqual({ ticker: "AAA", entry_price: 10, shares: 10 });
    expect(screen.getByRole("button", { name: "Close" })).toBeInTheDocument();
  });
});

describe("Day Trader", () => {
  it("is locked below Pro, and ranks live rows by day-trade score for Pro", async () => {
    const { unmount } = render(<SessionProvider initialMe={me("basic")}><DayTraderView /></SessionProvider>);
    expect(screen.getByText("Day Trader is part of Pro")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    unmount();
    fetchMock.mockResolvedValue(jsonResponse({ state: "closed", source: "movers", symbols: ["A", "B", "C"], missing: 1, as_of: new Date().toISOString(),
      rows: [{ ticker: "A", day_trade_score: 40, last: 5 }, { ticker: "B", day_trade_score: 80, last: 7 }] }));
    render(<SessionProvider initialMe={me("pro")}><DayTraderView /></SessionProvider>);
    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(3));
    expect(screen.getAllByRole("row")[1]).toHaveTextContent("B");
    expect(screen.getByText(/1 symbol has no live quote/)).toBeInTheDocument();
    expect(new URL((fetchMock.mock.calls[0]![0] as Request).url).searchParams.get("source")).toBe("movers");
  });
});
