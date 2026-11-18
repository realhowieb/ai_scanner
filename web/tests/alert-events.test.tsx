import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AlertsView } from "@/features/AlertsView";
import { groupFired, parseMessage, tickerStats } from "@/lib/alertEvents";
import { SessionProvider } from "@/session/SessionProvider";

import { fakeApi } from "./fakeApi";
import { jsonResponse } from "./helpers";
import { me } from "./fixtures";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  usePathname: () => "/alerts",
}));

const line = (tk: string, v: number, rule: number) => `${tk}: BreakoutScore ${v.toFixed(1)} (≥ ${rule})`;
const breakout = (rule: number, rows: [string, number][], more = 0) =>
  `Breakout alert: ${rows.map(([t, v]) => line(t, v, rule)).join("\n")}${more ? `\n…and ${more} more.` : ""}`;

const T = "2026-10-06T12:39:00Z";
const EVENTS = [
  { id: 2, event_id: "alert:2", source: "alert", ticker: "VICR", fired_at: T,
    message: breakout(30, [["VICR", 41.6], ["KOD", 50.2], ["PTC", 129.0], ["XP", 121.6]], 3) },
  { id: 1, event_id: "alert:1", source: "alert", ticker: "KOD", fired_at: "2026-10-06T12:39:20Z",
    message: breakout(50, [["KOD", 50.2], ["PTC", 129.0], ["XP", 121.6]]) },
  { id: 0, event_id: "alert:0", source: "alert", ticker: "NVDA", fired_at: "2026-10-05T08:00:00Z",
    message: "Move alert: NVDA -2.8% today (last 230.70, threshold ±2%)" },
];

describe("parseMessage", () => {
  it("splits ticker lines, the threshold, earnings flags and the cut-off count", () => {
    const p = parseMessage("Breakout alert: IOVA: BreakoutScore 118.9 (≥ 30) ⚠️ earnings in 3d\nKOD: BreakoutScore 69.5 (≥ 30)\n…and 7 more.");
    expect(p.label).toBe("Breakout alert");
    expect(p.rule).toBe("30");
    expect(p.extra).toBe(7);
    expect(p.rows).toEqual([
      { ticker: "IOVA", value: 118.9, detail: "BreakoutScore 118.9", earnings: "earnings in 3d" },
      { ticker: "KOD", value: 69.5, detail: "BreakoutScore 69.5", earnings: null },
    ]);
  });

  it("keeps a message that isn't ticker lines as text", () => {
    const p = parseMessage("Move alert: NVDA -2.8% today (last 230.70, threshold ±2%)");
    expect(p.rows).toEqual([]);
    expect(p.text).toBe("NVDA -2.8% today (last 230.70, threshold ±2%)");
  });
});

describe("groupFired", () => {
  it("merges same-scan breakout alerts, sorts strongest first and heads with the top scorer", () => {
    const items = groupFired(EVENTS);
    expect(items).toHaveLength(2);
    const [b, move] = items;
    expect(b!.ticker).toBe("PTC");
    expect(b!.rows.map((r) => r.ticker)).toEqual(["PTC", "XP", "KOD", "VICR"]);
    expect(b!.tiers).toEqual([{ rule: "50", count: 3 }, { rule: "30", count: 7 }]);
    expect(b!.total).toBe(7);
    expect(move!.title).toBe("Move");
    expect(move!.ticker).toBe("NVDA");
  });

  it("keeps the same rule firing on different scans apart", () => {
    const a = { id: 1, fired_at: "2026-10-06T12:39:00Z", message: breakout(30, [["A", 40]]) };
    const b = { id: 2, fired_at: "2026-10-06T17:12:00Z", message: breakout(50, [["A", 60]]) };
    expect(groupFired([a, b])).toHaveLength(2);
  });
});

describe("groupFired: new names", () => {
  it("marks names that weren't in the previous firing of the same kind", () => {
    const wl = (id: number, at: string, tks: string[]) => ({ id, fired_at: at, message: `Watchlist alert: ${tks.map((t) => `${t}: HSF 50`).join("\n")}` });
    const items = groupFired([wl(3, "2026-11-17T13:39:00Z", ["APP", "SEZL"]), wl(2, "2026-11-16T13:39:00Z", ["APP", "TTD"]), wl(1, "2026-11-13T13:39:00Z", ["APP"])]);
    expect([...items[0]!.fresh!]).toEqual(["SEZL"]);
    expect([...items[1]!.fresh!]).toEqual(["TTD"]);
    expect(items[2]!.fresh).toBeNull();
  });

  it("counts each ticker across alerts", () => {
    const stats = tickerStats(groupFired(EVENTS));
    expect(stats.find((s) => s.ticker === "PTC")).toMatchObject({ count: 1, best: 129, kinds: ["Breakout"] });
    expect(stats.find((s) => s.ticker === "NVDA")).toMatchObject({ count: 1, kinds: ["Move"] });
  });
});

const ago = (h: number) => new Date(Date.now() - h * 3600_000).toISOString();
const wlMsg = (tks: string[]) => `Watchlist alert: ${tks.map((t, i) => `${t}: HSF score ${50 - i}`).join("\n")}`;
const RECENT = [
  { id: 13, event_id: "alert:13", source: "alert", alert_id: 4, fired_at: ago(2), message: wlMsg(["APP", "BRZE", "SEZL"]) },
  { id: 12, event_id: "alert:12", source: "alert", alert_id: 5, fired_at: ago(2.1), message: breakout(75, [["CGON", 105.1], ["LFTO", 75.1]]) },
  { id: 11, event_id: "alert:11", source: "alert", alert_id: 4, fired_at: ago(26), message: wlMsg(["APP", "BRZE", "TTD"]) },
  { id: 10, event_id: "alert:10", source: "alert", alert_id: 4, fired_at: ago(24 * 20), message: wlMsg(["OLD"]) },
];

describe("Recently fired", () => {
  let api: ReturnType<typeof fakeApi>;
  beforeEach(() => {
    api = fakeApi({ events: EVENTS });
    vi.stubGlobal("fetch", vi.fn((input: Request | string, init?: RequestInit) => api.fetch(input, init)));
  });
  afterEach(() => vi.unstubAllGlobals());

  it("shows one row per scan with the top names; details show per-threshold counts", async () => {
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    // Nothing fired in the last 7 days, so it opens on everything loaded.
    expect(await screen.findByText("Breakout · 7 names")).toBeInTheDocument();
    const chips = within(screen.getByRole("list", { name: "Breakout matches" })).getAllByRole("link").map((a) => a.textContent);
    expect(chips).toEqual(["PTC", "XP", "KOD", "VICR"]);
    expect(screen.getByText("Move · NVDA")).toBeInTheDocument();
    expect(screen.getByText("NVDA -2.8% today (last 230.70, threshold ±2%)")).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Show details: Breakout · 7 names" }));
    expect(screen.getByText("≥ 50: 3")).toBeInTheDocument();
    expect(screen.getByText("≥ 30: 7")).toBeInTheDocument();
    expect(screen.getByText("7 matched · top 4 listed in the alert")).toBeInTheDocument();
  });

  it("shows the top 5 on the row and every listed name in the details", async () => {
    const rows = Array.from({ length: 12 }, (_, i) => [`T${i}`, 90 - i] as [string, number]);
    api = fakeApi({ events: [{ id: 9, event_id: "alert:9", source: "alert", fired_at: T, message: breakout(30, rows, 80) }] });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    const list = await screen.findByRole("list", { name: "Breakout matches" });
    expect(within(list).getAllByRole("link")).toHaveLength(5);
    expect(within(list).getByText("+87")).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Show details: Breakout · 92 names" }));
    expect(screen.getByText("92 matched · top 12 listed in the alert")).toBeInTheDocument();
    expect(within(screen.getByRole("list", { name: "All Breakout names" })).getAllByRole("link")).toHaveLength(12);
    expect(screen.getByText(/80 more matched but weren't listed/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open breakouts in Scanner →" })).toHaveAttribute("href", "/scanner?signal=breakout");
  });

  it("groups by day, filters by kind and range, marks new names and filters to a ticker from Fired most", async () => {
    api = fakeApi({ events: RECENT });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    const table = await screen.findByRole("table");
    // Last 7 days: three alerts, the 20-day-old one is hidden.
    expect(within(table).getAllByRole("list", { name: /matches$/ })).toHaveLength(3);
    expect(within(table).queryByText("OLD")).not.toBeInTheDocument();
    expect(within(screen.getByLabelText("Alert summary")).getByRole("button", { name: /3 fired in 7 days/ })).toHaveAttribute("aria-pressed", "true");
    // SEZL is new since the previous watchlist alert; APP repeats.
    expect(within(table).getAllByText("(new)").length).toBeGreaterThan(0);
    await u.click(screen.getByLabelText("New names only"));
    await u.click(screen.getByRole("radio", { name: "Watchlist" }));
    expect(within(screen.getByRole("table")).getAllByRole("list", { name: "Watchlist matches" })).toHaveLength(2);
    await u.click(screen.getByLabelText("New names only"));
    await u.click(screen.getByRole("radio", { name: "All" }));
    await u.click(screen.getByRole("radio", { name: "All loaded" }));
    expect(within(screen.getByRole("table")).getByText("OLD")).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "APP: fired 2 times" }));
    expect(screen.getByText(/Showing alerts with APP/)).toBeInTheDocument();
    expect(within(screen.getByRole("table")).getAllByRole("list", { name: /matches$/ })).toHaveLength(2);
  });

  it("Tickers tab lists how often each name fired; Show alerts filters Activity", async () => {
    api = fakeApi({ events: RECENT });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    await screen.findByRole("table");
    await u.click(screen.getByRole("tab", { name: "Tickers" }));
    const t = within(screen.getByRole("region", { name: "Tickers that fired" }));
    const app = t.getByRole("link", { name: "APP" }).closest("tr")!;
    expect(within(app).getByText("2")).toBeInTheDocument();
    await u.click(within(app).getByRole("button", { name: "Show APP alerts" }));
    expect(screen.getByRole("tab", { name: "Activity" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Filter by ticker")).toHaveValue("APP");
  });

  it("loads older alerts with the last cursor", async () => {
    const page = Array.from({ length: 100 }, (_, i) => ({ id: 500 - i, event_id: `alert:${500 - i}`, cursor: `c${i}`, source: "alert", fired_at: ago(1 + i * 0.1), message: wlMsg([`N${i}`]) }));
    const older = [{ id: 1, event_id: "alert:1", cursor: "z", source: "alert", fired_at: ago(24 * 40), message: wlMsg(["ANCIENT"]) }];
    const seen: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: Request) => {
      const url = new URL(input.url);
      if (url.pathname.endsWith("/v1/alerts/events")) { seen.push(url.search); return jsonResponse(url.searchParams.get("cursor") ? older : page); }
      return api.fetch(input);
    }));
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    await u.click(await screen.findByRole("button", { name: "Load older" }));
    expect(await within(screen.getByRole("table")).findByText("ANCIENT")).toBeInTheDocument();
    expect(seen[seen.length - 1]).toContain("cursor=c99");
    expect(screen.queryByRole("button", { name: "Load older" })).not.toBeInTheDocument();
  });
});

describe("Header chips and Rules table", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("a summary chip opens Rules filtered; bulk pause turns the selected alerts off", async () => {
    const api = fakeApi({ rules: true, events: [] });
    api.alertRows.push(
      { id: 7, type: "rvol", ticker: "TSLA", threshold: 2, direction: null, watchlist_only: false, enabled: true, last_fired_at: null, created_at: "" },
      { id: 8, type: "move", ticker: "AMD", threshold: 5, direction: null, watchlist_only: false, enabled: false, last_fired_at: null, created_at: "" },
      { id: 9, type: "price", ticker: "NVDA", threshold: 250, direction: "above", watchlist_only: false, enabled: true, last_fired_at: null, created_at: "" },
    );
    vi.stubGlobal("fetch", api.fetch);
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    const strip = await screen.findByLabelText("Alert summary");
    await u.click(await within(strip).findByRole("button", { name: /1 paused/ }));
    expect(screen.getByRole("tab", { name: "Rules" })).toHaveAttribute("aria-selected", "true");
    const rules = within(screen.getByRole("region", { name: "Your alert rules" }));
    expect(rules.getAllByRole("row")).toHaveLength(2);
    expect(rules.getByText("AMD moves ±5% in a day")).toBeInTheDocument();
    await u.click(screen.getByRole("radio", { name: /^Active/ }));
    await u.click(screen.getByRole("checkbox", { name: "Select all shown alerts" }));
    const bar = within(screen.getByRole("region", { name: "Selected alerts" }));
    expect(bar.getByText("2 selected")).toBeInTheDocument();
    await u.click(bar.getByRole("button", { name: "Pause" }));
    await waitFor(() => expect(api.alertRows.every((a) => !a.enabled)).toBe(true));
  });
});
