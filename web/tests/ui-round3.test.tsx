import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StarterAlerts, StarterWatchlistButton } from "@/features/Starters";
import { StockView } from "@/features/StockView";
import { WatchlistAINote } from "@/features/WatchlistAINote";
import { SessionProvider } from "@/session/SessionProvider";

import { me, stockDetail } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/stocks/AAA",
}));

const calls: { method: string; path: string; body: unknown }[] = [];
let routes: Record<string, (body: unknown) => unknown> = {};
beforeEach(() => {
  calls.length = 0;
  routes = {};
  vi.stubGlobal("fetch", async (req: Request) => {
    const url = new URL(req.url);
    const path = url.pathname.replace(/^\/api\/hsf/, "");
    const text = req.method === "GET" ? "" : await req.text();
    const body = text ? JSON.parse(text) : null;
    calls.push({ method: req.method, path, body });
    const h = routes[`${req.method} ${path}`];
    return h ? jsonResponse(h(body)) : jsonResponse({ detail: "Not Found" }, 404);
  });
});
afterEach(() => vi.unstubAllGlobals());

const ruleList = (rules: unknown[] = [], used = 0, limit = 5) => ({ limit, used, rules,
  capabilities: { tier: "pro", max_watchlists: 10, max_symbols_per_request: 50, max_active_alerts: limit, alert_rule_types: [], delivery_channels: ["in_app"] } });
const posted = (path: string) => calls.filter((c) => c.method === "POST" && c.path === path).map((c) => c.body);

describe("AI rundown upsell", () => {
  it("shows one upgrade button to Pro users, not two", () => {
    render(<SessionProvider initialMe={me("pro")}><WatchlistAINote id={1} name="Core" empty={false} /></SessionProvider>);
    expect(screen.getAllByRole("button", { name: "Upgrade to Premium" })).toHaveLength(1);
  });
});

describe("Unranked stock page", () => {
  it("adds a 'New HSF setup' alert in one tap", async () => {
    routes["GET /v1/alerts/rules"] = () => ruleList();
    routes["POST /v1/alerts/rules"] = (b) => ({ id: 3, ...(b as object) });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("basic")}>
      <StockView s={stockDetail({ in_latest_scan: false, has_setup: false, hsf_score: null, status: null, signals: [] })} premium={false} />
    </SessionProvider>);
    await u.click(await screen.findByRole("button", { name: "Alert me if AAA gets ranked" }));
    await waitFor(() => expect(posted("/v1/alerts/rules")).toEqual([
      { rule_type: "SETUP_APPEARED", ticker: "AAA", enabled: true, delivery_channels: ["in_app"] },
    ]));
    expect(await screen.findByText(/You'll get an alert when AAA becomes a ranked setup/)).toBeInTheDocument();
  });

  it("says it's already set when the rule exists", async () => {
    routes["GET /v1/alerts/rules"] = () => ruleList([{ id: 1, rule_type: "SETUP_APPEARED", ticker: "AAA", enabled: true, delivery_channels: ["in_app"] }], 1);
    render(<SessionProvider initialMe={me("basic")}>
      <StockView s={stockDetail({ in_latest_scan: false, has_setup: false, hsf_score: null, status: null, signals: [] })} premium={false} />
    </SessionProvider>);
    expect(await screen.findByText(/You'll get an alert when AAA becomes a ranked setup/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /gets ranked/ })).not.toBeInTheDocument();
  });

  it("keeps the smart alert form folded behind one button on stock pages", async () => {
    routes["GET /v1/alerts/rules"] = () => ruleList();
    routes["GET /v1/alerts/rules/types"] = () => [{ type: "HSF_SCORE_ABOVE", label: "HSF Score at or above", description: "Fires.", available: true,
      threshold: { min: 0, max: 100, default: 80 }, scope: "ticker" }];
    routes["GET /v1/watchlists"] = () => [];
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><StockView s={stockDetail()} premium={false} pro /></SessionProvider>);
    const open = await screen.findByRole("button", { name: "Add smart alert" });
    expect(screen.queryByRole("form", { name: "New smart alert" })).not.toBeInTheDocument();
    await u.click(open);
    expect(screen.getByRole("form", { name: "New smart alert" })).toBeInTheDocument();
  });
});

describe("Starters", () => {
  it("creates a default list from today's top names", async () => {
    routes["POST /v1/watchlists"] = () => ({ id: 7, name: "Today's top setups", is_default: true, symbol_count: 0 });
    routes["POST /v1/watchlists/7/tickers"] = () => ({ added: 3 });
    const done = vi.fn();
    const u = userEvent.setup();
    render(<StarterWatchlistButton tickers={["AAA", "BBB", "CCC"]} onDone={done} />);
    await u.click(screen.getByRole("button", { name: "Start with today's top 3" }));
    await waitFor(() => expect(done).toHaveBeenCalledWith(7));
    expect(posted("/v1/watchlists")).toEqual([{ name: "Today's top setups", make_default: true }]);
    expect(posted("/v1/watchlists/7/tickers")).toEqual([{ tickers: ["AAA", "BBB", "CCC"] }]);
  });

  it("fills an existing empty list instead of making a new one", async () => {
    routes["POST /v1/watchlists/4/tickers"] = () => ({ added: 1 });
    const u = userEvent.setup();
    render(<StarterWatchlistButton tickers={["AAA"]} listId={4} onDone={() => undefined} />);
    await u.click(screen.getByRole("button", { name: "Start with today's top 1" }));
    await waitFor(() => expect(posted("/v1/watchlists/4/tickers")).toHaveLength(1));
    expect(posted("/v1/watchlists")).toHaveLength(0);
  });

  it("offers ready-made alerts on the default list and a Pro scan alert", async () => {
    routes["GET /v1/watchlists"] = () => [{ id: 4, name: "Core", is_default: true, symbol_count: 3 }];
    routes["GET /v1/alerts/rules"] = () => ruleList();
    routes["POST /v1/alerts/rules"] = (b) => ({ id: 1, ...(b as object) });
    const done = vi.fn();
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><StarterAlerts onDone={done} /></SessionProvider>);
    expect(await screen.findByText("A stock on Core scores 75 or more")).toBeInTheDocument();
    expect(screen.getByText("A new Breakout setup scores 70 or more")).toBeInTheDocument();
    await u.click(screen.getAllByRole("button", { name: "Add" })[0]!);
    await waitFor(() => expect(posted("/v1/alerts/rules")).toEqual([
      { rule_type: "HSF_SCORE_CROSS_ABOVE", watchlist_id: 4, threshold: 75, enabled: true, delivery_channels: ["in_app"] },
    ]));
    expect(done).toHaveBeenCalledWith("A stock on Core scores 75 or more");
  });

  it("stays out of the way once the account has smart alerts", async () => {
    routes["GET /v1/watchlists"] = () => [{ id: 4, name: "Core", is_default: true, symbol_count: 3 }];
    routes["GET /v1/alerts/rules"] = () => ruleList([{ id: 1, rule_type: "RVOL_ABOVE", watchlist_id: 4, enabled: true, delivery_channels: ["in_app"] }], 1);
    const { container } = render(<SessionProvider initialMe={me("basic")}><StarterAlerts onDone={() => undefined} /></SessionProvider>);
    await waitFor(() => expect(calls.some((c) => c.path === "/v1/alerts/rules")).toBe(true));
    await new Promise((r) => setTimeout(r, 20));
    expect(container).toBeEmptyDOMElement();
  });
});
