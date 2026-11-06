import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { journalCsv } from "@/features/JournalView";
import { csvFile, ResearchCsvButton } from "@/features/ResearchCsv";
import { describeSchedule, ScheduledScans, statusText } from "@/features/ScheduledScans";
import { trackRecordCsv } from "@/features/TrackRecordView";
import { watchlistCsv } from "@/features/WatchlistsView";
import { SessionProvider } from "@/session/SessionProvider";

import { me, trackDefaults } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  usePathname: () => "/scanner/custom",
}));

const fetchMock = vi.fn();
beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", (request: Request) => fetchMock(request));
});
afterEach(() => vi.unstubAllGlobals());

const sched = (over: Partial<Schemas["ScheduledScan"]> = {}): Schemas["ScheduledScan"] => ({
  id: "a".repeat(32), name: "Morning gappers", run_at: "open", created_at: "2026-11-02T12:00:00+00:00",
  request: { universe: "nasdaq", filters: { session: "premarket", top_n: 50 } }, last_status: null, last_detail: null, last_run_at: null, last_scan_id: null, ...over,
});
const req = { universe: "sp500" as const, filters: { session: "regular" as const, top_n: 25 } };

describe("Scheduled scans", () => {
  it("is a locked Premium teaser for Pro and makes no API call", () => {
    render(<SessionProvider initialMe={me("pro")}><ScheduledScans current={() => ({ req })} /></SessionProvider>);
    expect(screen.getByText("Scheduled scans are part of Premium")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upgrade to Premium" })).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("lists schedules and saves the current form for Premium", async () => {
    fetchMock.mockImplementation((r: Request) => Promise.resolve(r.method === "POST"
      ? jsonResponse(sched({ id: "b".repeat(32), name: "Lunch", run_at: "midday" }), 201)
      : jsonResponse([sched({ last_status: "sent", last_detail: "12 setups", last_run_at: "2026-11-02T13:30:00+00:00" })])));
    render(<SessionProvider initialMe={me("premium")}><ScheduledScans current={() => ({ req })} /></SessionProvider>);
    expect(await screen.findByText("Morning gappers")).toBeInTheDocument();
    expect(screen.getByText("NASDAQ · pre-market · top 50 · every trading day before the open")).toBeInTheDocument();
    expect(screen.getByText("Emailed Nov 2 · 12 setups")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Lunch" } });
    fireEvent.change(screen.getByLabelText("When"), { target: { value: "midday" } });
    fireEvent.click(screen.getByRole("button", { name: "Email me this scan every trading day" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([r]) => (r as Request).method === "POST")).toBe(true));
    const post = fetchMock.mock.calls.map(([r]) => r as Request).find((r) => r.method === "POST")!;
    expect(new URL(post.url).pathname).toMatch(/\/v1\/scheduled-scans$/);
    expect(await post.json()).toEqual({ name: "Lunch", run_at: "midday", scan: req });
  });

  it("asks for a name and passes on form errors without saving", async () => {
    fetchMock.mockResolvedValue(jsonResponse([]));
    const { unmount } = render(<SessionProvider initialMe={me("premium")}><ScheduledScans current={() => ({ req })} /></SessionProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "Email me this scan every trading day" }));
    expect(screen.getByText(/Give this scan a name/)).toHaveAttribute("role", "alert");
    unmount();
    render(<SessionProvider initialMe={me("premium")}><ScheduledScans current={() => ({ error: "Choose a watchlist." })} /></SessionProvider>);
    fireEvent.click(await screen.findByRole("button", { name: "Email me this scan every trading day" }));
    expect(screen.getByText("Choose a watchlist.")).toHaveAttribute("role", "alert");
    expect(fetchMock.mock.calls.every(([r]) => (r as Request).method === "GET")).toBe(true);
  });

  it("hides the form at the limit of three", async () => {
    fetchMock.mockResolvedValue(jsonResponse([sched(), sched({ id: "b".repeat(32) }), sched({ id: "c".repeat(32) })]));
    render(<SessionProvider initialMe={me("premium")}><ScheduledScans current={() => ({ req })} /></SessionProvider>);
    expect(await screen.findByText(/most scheduled scans allowed/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Email me this scan every trading day" })).not.toBeInTheDocument();
  });

  it("explains each status in plain words", () => {
    expect(statusText({ last_status: null, last_detail: null, last_run_at: null }).text).toBe("Waiting for its first run");
    expect(statusText({ last_status: "unsubscribed", last_detail: null, last_run_at: null }).text).toMatch(/alert emails are off/);
    expect(statusText({ last_status: "paused", last_detail: "Scheduled scans are part of Premium.", last_run_at: null }).text).toBe("Scheduled scans are part of Premium.");
    expect(describeSchedule({ run_at: "close", request: { universe: "ticker", ticker: "AAPL", filters: {} } })).toBe("AAPL · top 25 · every trading day after the close");
  });
});

describe("Research CSV (Premium)", () => {
  it("shows a Premium link instead of the button on other plans", () => {
    render(<SessionProvider initialMe={me("pro")}><ResearchCsvButton what="your journal" filename="x.csv" build={() => ({ header: [], rows: [] })} /></SessionProvider>);
    expect(screen.getByRole("link", { name: "CSV export of your journal is part of Premium" })).toHaveAttribute("href", "/pricing");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("downloads for Premium", async () => {
    const build = vi.fn(() => ({ header: ["Ticker"], rows: [["AAA"]] }));
    const created = vi.fn(() => "blob:x");
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: created, revokeObjectURL: vi.fn() }));
    render(<SessionProvider initialMe={me("premium")}><ResearchCsvButton what="your journal" filename="x.csv" build={build} /></SessionProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Download your journal as CSV" }));
    await waitFor(() => expect(build).toHaveBeenCalled());
    await waitFor(() => expect(created).toHaveBeenCalled());
  });

  it("names files by kind, list and New York day", () => {
    expect(csvFile("watchlist", "Tech Leaders!", "2026-11-02")).toBe("hsf-watchlist-tech-leaders-2026-11-02.csv");
    expect(csvFile("journal", "", "2026-11-02")).toBe("hsf-journal-2026-11-02.csv");
  });

  it("builds the watchlist, journal and Track record tables", () => {
    const wl = watchlistCsv([
      { ticker: "AAA", note: "=cmd", added_at: "2026-10-01", price_when_added: 10, latest: { rank: 3 } },
      { ticker: "BBB", note: null },
    ] as never, new Map([["AAA", { score: 81, primary_setup: "breakout", last: 12, chg_pct: 1.234 }]]), { AAA: { last: 12.5, chg_pct: 2.5, source: "trade" } }, "2026-11-02T13:00:00Z");
    expect(wl.rows[0]).toEqual(["AAA", 81, "Breakout", 3, 12.5, 2.5, "latest trade", "2026-10-01", 10, "=cmd", "2026-11-02T13:00:00Z"]);
    expect(wl.rows[1]!.slice(0, 7)).toEqual(["BBB", null, "", null, null, null, ""]);
    const j = journalCsv([{ id: 1, ticker: "AAA", open: false, shares: 10, entry_price: 10, exit_price: 11, mark: 11, pnl: 10, pnl_pct: 10, source: "scan", entered_at: "a", closed_at: "b" }]);
    expect(j.rows[0]).toEqual(["AAA", "Closed", "scan", 10, 10, "a", 11, "b", 11, 10, 10]);
    const tr = trackRecordCsv([
      { ...trackDefaults, ranking: "breakout", ranking_label: "Breakout", horizon_days: 5, avg_excess_return: 0.0123, median_excess_return: 0.01, win_rate: 0.55, sample_size: 400, runs_used: 40, benchmark: "SPY", sufficient: true },
      { ...trackDefaults, ranking: "breakout", ranking_label: "Breakout", horizon_days: 20, avg_excess_return: 0.5, sample_size: 3, benchmark: "SPY", sufficient: false },
    ], [{ day: "2026-10-30", avg_excess_return: -0.002, methodology: "track-v2-next-open" }], 5);
    expect(tr.rows[0]!.slice(0, 7)).toEqual(["Summary", "Breakout", 5, "", 1.23, 1, 55]);
    expect(tr.rows[1]!.slice(4, 7)).toEqual([null, null, null]);   // still building: no rates
    expect(tr.rows[2]!.slice(0, 5)).toEqual(["Daily", "Breakout", 5, "2026-10-30", -0.2]);
    expect(tr.rows.every((row) => row.length === tr.header.length)).toBe(true);
    expect(tr.header).toContain("Coverage %");
    expect(tr.rows[0]![11]).toBe("track-v2-next-open");
    expect(tr.rows[0]![13]).toBe(-2); // absolute loss stays distinct from positive excess
  });
});
