import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Watchlist } from "@/api/userData";
import { WatchlistsView, dupKey, groupLists, lookAlikes, sinceAdded, sortRows } from "@/features/WatchlistsView";
import { SessionProvider } from "@/session/SessionProvider";

import { fakeApi } from "./fakeApi";
import { me } from "./fixtures";

let search = new URLSearchParams();
const replace = vi.fn((url: string) => { search = new URLSearchParams(url.split("?")[1] ?? ""); });
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/watchlists",
}));

let api: ReturnType<typeof fakeApi>;
beforeEach(() => {
  search = new URLSearchParams();
  localStorage.removeItem("hsf.watchlists.v1");
  api = fakeApi({ scanState: true, scan: [{ ticker: "NVDA", score: 90, last: 120 }, { ticker: "AAPL", score: 81, last: 201.5 }] });
  vi.stubGlobal("fetch", vi.fn((input: Request | string, init?: RequestInit) => api.fetch(input, init)));
});
afterEach(() => vi.unstubAllGlobals());

const wl = (id: number, name: string, extra: Partial<Watchlist> = {}): Watchlist => ({ id, name, is_default: false, symbol_count: 1, pinned: false, archived: false, ...extra });
const view = () => render(<SessionProvider initialMe={me("pro")}><WatchlistsView /></SessionProvider>);
const tickers = () => [...document.querySelectorAll(".wl-table tbody tr a.tk")].map((a) => a.textContent);

describe("watchlist list helpers", () => {
  it("spots look-alike names: trailing numbers, (2) and Copy, but not different years", () => {
    expect(dupKey("July Top List")).toBe(dupKey("july top list 2"));
    expect(dupKey("Focus July 2026")).toBe(dupKey("Focus July 2026 Copy"));
    expect(dupKey("Focus July 2026")).not.toBe(dupKey("Focus July 2025"));
    const groups = lookAlikes([wl(1, "July Top List", { symbol_count: 4 }), wl(2, "July Top List 2", { symbol_count: 9 }), wl(3, "Swing"), wl(4, "Swing (2)", { archived: true })]);
    expect(groups.map((g) => g.map((w) => w.id))).toEqual([[2, 1]]); // biggest first; archived lists aren't flagged
  });

  it("groups lists: default + pinned on top, earnings lists together, archived apart; search filters all groups", () => {
    const all = [wl(1, "Main", { is_default: true }), wl(2, "Focus", { pinned: true }), wl(3, "Earnings Nov 17"), wl(4, "Swing"), wl(5, "Old earnings", { archived: true })];
    const g = groupLists(all);
    expect([g.pinned, g.active, g.earnings, g.archived].map((x) => x.map((w) => w.id))).toEqual([[1, 2], [4], [3], [5]]);
    expect(groupLists(all, "earn").earnings.map((w) => w.id)).toEqual([3]);
    expect(groupLists(all, "earn").archived.map((w) => w.id)).toEqual([5]);
  });

  it("sorts with missing values last in either direction, and computes since-added", () => {
    const rows = [{ ticker: "B", v: 2 }, { ticker: "A", v: null }, { ticker: "C", v: 5 }];
    expect(sortRows(rows, (r) => r.v, true).map((r) => r.ticker)).toEqual(["C", "B", "A"]);
    expect(sortRows(rows, (r) => r.v, false).map((r) => r.ticker)).toEqual(["B", "C", "A"]);
    expect(sinceAdded(100, 110)).toBeCloseTo(10);
    expect(sinceAdded(null, 110)).toBeNull();
    expect(sinceAdded(0, 110)).toBeNull();
  });
});

describe("Watchlists page", () => {
  it("one table row per ticker with Since added; the note opens inline", async () => {
    api.seedList("Main", ["AAPL", "NVDA", "ZZZ"], true);
    const u = userEvent.setup();
    view();
    await screen.findByLabelText("HSF Score 90");
    expect(tickers()).toEqual(["NVDA", "AAPL", "ZZZ"]);
    expect(screen.getByRole("columnheader", { name: /Since added/ })).toBeInTheDocument();
    expect(screen.getByText("+1100.0%")).toBeInTheDocument(); // NVDA 120 vs 10 when added (fake data)
    await u.click(screen.getByRole("button", { name: "Add note for AAPL" }));
    await u.type(screen.getByLabelText("Note for AAPL"), "Breakout level 205");
    await u.click(screen.getByRole("button", { name: "Save note" }));
    expect(await screen.findByText("Breakout level 205")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit note for AAPL" })).toBeInTheDocument();
  });

  it("tabs hold the other cards", async () => {
    api.seedList("Main", ["AAPL"], true);
    const u = userEvent.setup();
    view();
    const perf = await screen.findByRole("tab", { name: "Performance" });
    expect(screen.getByRole("tab", { name: "Tickers" })).toHaveAttribute("aria-selected", "true");
    await u.click(perf);
    expect(perf).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: "Performance" })).toBeVisible();
    expect(screen.queryByRole("tabpanel", { name: "Tickers" })).not.toBeInTheDocument();
  });

  it("bulk: select tickers, move them to another list, then remove the rest after confirming", async () => {
    const main = api.seedList("Main", ["AAPL", "NVDA", "ZZZ"], true);
    const swing = api.seedList("Swing", []);
    const u = userEvent.setup();
    view();
    await screen.findByLabelText("HSF Score 90");
    await u.click(screen.getByRole("checkbox", { name: "Select AAPL" }));
    await u.click(screen.getByRole("checkbox", { name: "Select ZZZ" }));
    const bar = screen.getByRole("region", { name: "Selected tickers" });
    expect(bar).toHaveTextContent("2 selected");
    expect(within(bar).getByRole("link", { name: "Compare" })).toHaveAttribute("href", "/stocks/compare?symbols=AAPL%2CZZZ");
    await u.click(within(bar).getByRole("button", { name: "Move to…" }));
    const dlg = screen.getByRole("dialog", { name: "Move 2 tickers" });
    await u.click(within(dlg).getByRole("button", { name: "Move" }));
    expect(await screen.findByText(/Moved 2 to Swing\./)).toBeInTheDocument();
    expect(swing.items.map((i) => i.ticker)).toEqual(["AAPL", "ZZZ"]);
    await waitFor(() => expect(tickers()).toEqual(["NVDA"]));

    await u.click(screen.getByRole("checkbox", { name: "Select all shown tickers" }));
    await u.click(screen.getByRole("button", { name: "Remove selected" }));
    await u.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(main.items).toHaveLength(0));
  });

  it("pin, archive (not the default) and merge a look-alike list", async () => {
    api.seedList("Main", ["AAPL"], true);
    const july = api.seedList("July Top List", ["NVDA"]);
    const july2 = api.seedList("July Top List 2", ["NVDA", "AMD"]);
    const old = api.seedList("Old list", ["ZZZ"]);
    search = new URLSearchParams(`id=${old.id}`);
    const u = userEvent.setup();
    const { rerender } = view();
    await screen.findByRole("heading", { name: /Old list/ });
    await u.click(screen.getByRole("button", { name: "Archive" }));
    await waitFor(() => expect(old.archived).toBe(true));
    expect(await screen.findByText(/This list is archived/)).toBeInTheDocument();
    expect(await screen.findByText("Archived · 1")).toBeInTheDocument();

    // the look-alike hint opens Merge with both lists filled in
    const hint = screen.getByRole("note");
    expect(hint).toHaveTextContent("July Top List 2 + July Top List");
    await u.click(within(hint).getByRole("button", { name: "Merge" }));
    rerender(<SessionProvider initialMe={me("pro")}><WatchlistsView /></SessionProvider>);
    const dlg = await screen.findByRole("dialog", { name: "Merge into July Top List 2" });
    expect(within(dlg).getByLabelText("Merge this list in")).toHaveValue(String(july.id));
    await u.click(within(dlg).getByRole("button", { name: "Merge and delete" }));
    expect(await screen.findByText(/Merged July Top List into July Top List 2: 0 new tickers added\./)).toBeInTheDocument();
    expect(api.lists.map((w) => w.name)).not.toContain("July Top List");
    expect(july2.items.map((i) => i.ticker)).toEqual(["NVDA", "AMD"]);

    await u.click(await screen.findByRole("button", { name: "Pin to top" }));
    await waitFor(() => expect(july2.pinned).toBe(true));
  });
});
