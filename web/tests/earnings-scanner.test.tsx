import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { describeScanAlert } from "@/features/ScanAlert";
import { ScannerView } from "@/features/ScannerView";
import { SessionProvider } from "@/session/SessionProvider";

import { me, setup } from "./fixtures";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace, push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/scanner",
}));

const fetchMock = vi.fn();
const urls = () => fetchMock.mock.calls.map((c) => new URL((c[0] as Request).url));
const latest = { scan_at: new Date().toISOString(), total: 2, scan_total: 100, max_results: 100, limited: false, setups: [setup("AAA", 88), setup("BBB", 70)] };

const row = (ticker: string, date: string, days: number, extra: Partial<Schemas["EarningsSetup"]> = {}): Schemas["EarningsSetup"] => ({
  ...setup(ticker, 70), rank: 3, ranked: true, earnings_date: date, days_until: days, time: "amc", on_watchlist: false, ...extra,
});
const scan = (setups: Schemas["EarningsSetup"][], over: Partial<Schemas["EarningsScan"]> = {}): Schemas["EarningsScan"] => ({
  window: "week", start: "2026-10-21", end: "2026-10-28", scan_at: new Date().toISOString(), stale: false,
  total: setups.filter((s) => s.ranked).length, unranked_total: 0, max_results: 100, limited: false, setups, ...over,
});

let earnings: () => Response = () => jsonResponse(scan([]));
beforeEach(() => {
  search = new URLSearchParams("signal=earnings");
  replace.mockReset();
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (req: Request) => {
    const p = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
    if (p === "/v1/me/research-preferences") return jsonResponse({ scanner_views: [], custom_views: [], table: { columns: ["setup"], compact: false }, quiet_hours: { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" } });
    if (p === "/v1/scans/latest/earnings") return earnings();
    if (p === "/v1/scans/latest") return jsonResponse(latest);
    return jsonResponse({ detail: `no fake for ${p}` }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

const wrap = (plan: "basic" | "pro" | "premium") => render(<SessionProvider initialMe={me(plan)}><ScannerView /></SessionProvider>);
const earnCalls = () => urls().filter((u) => u.pathname.endsWith("/v1/scans/latest/earnings"));

describe("Earnings scanner", () => {
  it("groups reporting setups by day with timing and watchlist tags", async () => {
    search = new URLSearchParams("signal=earnings&when=week");
    earnings = () => jsonResponse(scan([
      row("TSLA", "2026-10-21", 0, { score: 78, time: "amc" }),
      row("T", "2026-10-21", 0, { score: 61, time: "bmo", on_watchlist: true }),
      row("INTC", "2026-10-22", 1, { time: null }),
    ]));
    wrap("pro");
    const table = await screen.findByRole("table");
    expect(within(table).getByText(/^Today · /)).toBeInTheDocument();
    expect(within(table).getByText(/^Tomorrow · /)).toBeInTheDocument();
    expect(within(table).getAllByText("After close")).toHaveLength(1);
    expect(within(table).getByText("Before open")).toBeInTheDocument();
    expect(within(table).getByText("Time TBA")).toBeInTheDocument();
    expect(within(table).getByText("Watchlist")).toBeInTheDocument();
    expect(screen.getByText("100 setups ranked · 3 report in the next 7 days")).toBeInTheDocument();
    expect(earnCalls()[0]!.searchParams.get("window")).toBe("week");
    // The main scan is fetched without a signal filter (earnings is not a scan signal).
    expect(urls().find((u) => u.pathname.endsWith("/v1/scans/latest"))!.searchParams.has("signal")).toBe(false);
  });

  it("keeps today free and locks the week ahead on Starter without calling the API", async () => {
    search = new URLSearchParams("signal=earnings&when=week");
    wrap("basic");
    expect(await screen.findByText("The earnings week ahead is part of Pro")).toBeInTheDocument();
    expect(earnCalls()).toHaveLength(0);
    expect(screen.getByRole("button", { name: /Next 7 days\s*· Pro/ })).toBeInTheDocument();
  });

  it("shows today on Starter and offers unranked reporters when none ranked", async () => {
    earnings = () => jsonResponse(scan([], { window: "today", unranked_total: 4 }));
    wrap("basic");
    expect(await screen.findByText("No ranked setups report today.")).toBeInTheDocument();
    expect(earnCalls()[0]!.searchParams.get("window")).toBe("today");
    expect(screen.getByRole("link", { name: "Score every reporter" })).toHaveAttribute("href", "/scanner/custom?universe=earnings&when=today&start=1");
    await userEvent.setup().click(screen.getByRole("button", { name: "Show all reporters" }));
    expect(replace).toHaveBeenCalledWith("/scanner?signal=earnings&all=1", { scroll: false });
  });

  it("marks unranked names and sends the timing filter", async () => {
    search = new URLSearchParams("signal=earnings&etime=bmo&all=1");
    earnings = () => jsonResponse(scan([row("ZZZQ", "2026-10-21", 0, { ranked: false, score: null, rank: null, time: "bmo" })], { window: "today" }));
    wrap("pro");
    expect((await screen.findAllByText("Not ranked")).length).toBeGreaterThan(0);
    const q = earnCalls()[0]!.searchParams;
    expect([q.get("time"), q.get("include_unranked")]).toEqual(["bmo", "true"]);
    await userEvent.setup().click(screen.getByRole("button", { name: "Tomorrow" }));
    expect(replace).toHaveBeenCalledWith("/scanner?signal=earnings&etime=bmo&all=1&when=tomorrow", { scroll: false });
  });

  it("describes earnings scan alerts", () => {
    expect(describeScanAlert(0, '{"earnings":"today"}')).toBe("HSF 0+, reporting today");
  });
});
