import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ScannerView } from "@/features/ScannerView";
import { sortWatchlistRows } from "@/features/WatchlistScanner";
import { SessionProvider } from "@/session/SessionProvider";

import { me, setup } from "./fixtures";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/scanner",
}));

const fetchMock = vi.fn();
const latest = { scan_at: new Date().toISOString(), total: 1, scan_total: 100, max_results: 100, limited: false, setups: [setup("AAA", 88)] };
const item = (ticker: string, extra: Record<string, unknown> = {}) => ({ ticker, added_at: null, note: null, price_when_added: null, latest: null, ...extra });
let detail: Record<string, unknown> = {};

beforeEach(() => {
  search = new URLSearchParams("signal=watchlist");
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (req: Request) => {
    const p = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
    if (p === "/v1/me/research-preferences") return jsonResponse({ scanner_views: [], custom_views: [], table: { columns: ["setup"], compact: false }, quiet_hours: { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" } });
    if (p === "/v1/watchlists") return jsonResponse([{ id: 4, name: "Core", is_default: true, symbol_count: 3 }]);
    if (p === "/v1/watchlists/4") return jsonResponse(detail);
    if (p === "/v1/quotes") return jsonResponse({ quotes: { KO: { last: 95, chg_pct: -0.5, gap_pct: 0, at: null, source: "trade" } }, feed: null });
    if (p === "/v1/scans/latest") return jsonResponse(latest);
    return jsonResponse({ ok: true });
  });
  vi.stubGlobal("fetch", fetchMock);
  detail = { id: 4, name: "Core", is_default: true, symbol_count: 3, scan_at: latest.scan_at, scan_total: 100, stale: false,
    items: [item("KO"), item("AAA", { latest: { ...setup("AAA", 88), rank: 1 } }), item("ZZZQ", { auto: setup("ZZZQ", 57) })] };
});
afterEach(() => vi.unstubAllGlobals());

const wrap = (plan: "basic" | "pro" | "premium") => render(<SessionProvider initialMe={me(plan)}><ScannerView /></SessionProvider>);

describe("Scanner Watchlist tab", () => {
  it("shows ranked, auto-scored and unscored stocks in that order", async () => {
    wrap("premium");
    const table = await screen.findByRole("table");
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows.map((r) => r.querySelector("a.tk")?.getAttribute("href"))).toEqual(["/stocks/AAA", "/stocks/ZZZQ", "/stocks/KO"]);
    expect(within(rows[0]!).getByText("#1")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("Auto")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("57")).toBeInTheDocument();
    expect(within(rows[2]!).getByText("Not ranked")).toBeInTheDocument();
    expect(screen.getByTestId("watchlist-scan-summary")).toHaveTextContent("1 more scored automatically after the scan (Auto)");
    expect(screen.queryByText(/scores every watchlist stock after each scan/)).not.toBeInTheDocument();
  });

  it("offers Premium's auto-score to other plans", async () => {
    detail = { ...detail, items: [item("KO"), item("AAA", { latest: { ...setup("AAA", 88), rank: 1 } })] };
    wrap("pro");
    await screen.findByRole("table");
    expect(screen.getByText(/scores every watchlist stock after each scan, including the 1 that didn/)).toBeInTheDocument();
  });

  it("shows when automatic scores were written and warns about a stale market scan", async () => {
    detail = { ...detail, stale: true, auto_scored_at: "2026-11-04T14:40:00Z" };
    wrap("premium");
    await screen.findByRole("table");
    expect(screen.getByTestId("watchlist-scan-summary")).toHaveTextContent(/Scored Nov 4, 9:40 AM ET/);
    expect(screen.getByText(/HSF Scores from the scan/)).toHaveTextContent("Scan delayed");
  });

  it("sorts ranked before auto-scored before unscored on HSF Score", () => {
    const rows = [setup("KO", 0), setup("ZZZQ", 90), setup("AAA", 50)];
    const out = sortWatchlistRows(rows, { KO: null, ZZZQ: null, AAA: 3 }, "score", new Set(["ZZZQ"]));
    expect(out.map((r) => r.ticker)).toEqual(["AAA", "ZZZQ", "KO"]);
  });
});
