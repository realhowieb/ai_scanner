import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AlertsView } from "@/features/AlertsView";
import { groupFired, parseMessage } from "@/lib/alertEvents";
import { SessionProvider } from "@/session/SessionProvider";

import { fakeApi } from "./fakeApi";
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

describe("Recently fired", () => {
  let api: ReturnType<typeof fakeApi>;
  beforeEach(() => {
    api = fakeApi({ events: EVENTS });
    vi.stubGlobal("fetch", vi.fn((input: Request | string, init?: RequestInit) => api.fetch(input, init)));
  });
  afterEach(() => vi.unstubAllGlobals());

  it("shows one entry per scan with ticker chips and per-threshold counts", async () => {
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    expect(await screen.findByText("Breakout · 7 names")).toBeInTheDocument();
    expect(screen.getByText("≥ 50: 3")).toBeInTheDocument();
    expect(screen.getByText("≥ 30: 7")).toBeInTheDocument();
    const chips = within(screen.getByRole("list", { name: "Breakout matches" })).getAllByRole("link").map((a) => a.textContent);
    expect(chips).toEqual(["PTC", "XP", "KOD", "VICR"]);
    expect(screen.getByText("Move · NVDA")).toBeInTheDocument();
    expect(screen.getByText("NVDA -2.8% today (last 230.70, threshold ±2%)")).toBeInTheDocument();
  });

  it("shows the top 8 and expands to the rest", async () => {
    const rows = Array.from({ length: 12 }, (_, i) => [`T${i}`, 90 - i] as [string, number]);
    api = fakeApi({ events: [{ id: 9, event_id: "alert:9", source: "alert", fired_at: T, message: breakout(30, rows, 5) }] });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    const list = await screen.findByRole("list", { name: "Breakout matches" });
    expect(within(list).getAllByRole("link")).toHaveLength(8);
    await u.click(screen.getByRole("button", { name: "Show all 12" }));
    expect(within(list).getAllByRole("link")).toHaveLength(12);
    expect(screen.getByText("…and 5 more not listed in the alert.")).toBeInTheDocument();
  });
});
