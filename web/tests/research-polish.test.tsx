// Research polish pass (2026-11-01): data labels say what a price is, research counts never
// read missing as zero, failures show a retry, saved views keep every filter, Pro CSV export.
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LiveSetupsPreview, previewTitle } from "@/features/LiveSetupsPreview";
import { SimilarSetups } from "@/features/OutcomeEvidence";
import { ScannerCsvButton, csvName, scannerCsv } from "@/features/ScannerCsv";
import { ScannerView } from "@/features/ScannerView";
import { observationLine, sampleCoverage } from "@/features/StockView";
import { TrackRecordView } from "@/features/TrackRecordView";
import { WatchlistsView, priceNote } from "@/features/WatchlistsView";
import { clearApiCache } from "@/hooks/useApi";
import { csvCell, toCsv } from "@/lib/csv";
import { SessionProvider } from "@/session/SessionProvider";

import { fakeApi } from "./fakeApi";
import { me, setup } from "./fixtures";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
const replace = vi.fn();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace, push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/scanner",
}));

beforeEach(() => { clearApiCache(); search = new URLSearchParams(); replace.mockReset(); });
afterEach(() => vi.unstubAllGlobals());

const wrap = (ui: React.ReactNode, plan: "basic" | "pro" | "premium") => render(<SessionProvider initialMe={me(plan)}>{ui}</SessionProvider>);
const path = (r: Request) => new URL(r.url).pathname.replace(/^\/api\/hsf/, "");
const PREFS = { scanner_views: [], custom_views: [], table: { columns: ["setup", "spark", "last", "change", "gap", "rvol", "prob"], compact: false },
  quiet_hours: { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" } };

describe("CSV export", () => {
  it("escapes commas and quotes, leaves numbers alone and neutralises spreadsheet formulas", () => {
    expect(csvCell('a,"b"')).toBe('"a,""b"""');
    expect(csvCell(-1.25)).toBe("-1.25");
    expect(csvCell("=HYPERLINK(\"x\")")).toBe("\"'=HYPERLINK(\"\"x\"\")\"");
    expect(csvCell("@SUM")).toBe("'@SUM");
    expect(csvCell(Number.NaN)).toBe("");
    expect(csvCell(null)).toBe("");
    expect(toCsv(["a"], [[1]])).toBe("a\r\n1\r\n");
  });

  it("labels scan prices as scan prices, leaves missing numbers blank and keeps PreBreakout to Premium", () => {
    const rows = [setup("AAA", 88, { gap_pct: null, prob: 22, rvol: 1.4999999 }), setup("BBB", 70, { last: Number.NaN })];
    const pro = scannerCsv(rows, "2026-10-30T19:49:00Z", false).split("\r\n");
    expect(pro[0]).toBe("Rank,Ticker,HSF Score,Setup,Status,Signals,Scan price,Change % at scan,Gap %,Relative volume,Breakout score,Scanned at (UTC)");
    expect(pro[1]).toBe("1,AAA,88,Breakout,STRONG,Breakout,50,2,,1.5,80,2026-10-30T19:49:00Z");
    expect(pro[2]!.split(",")[6]).toBe("");                     // NaN price: blank, never 0
    expect(scannerCsv(rows, null, true).split("\r\n")[0]).toContain("PreBreakout % (model estimate)");
    expect(csvName("2026-10-30T19:49:00Z")).toBe("hsf-scanner-2026-10-30-1549-ET.csv");
    expect(csvName(null)).toBe("hsf-scanner.csv");
  });

  it("downloads every filtered row in pages of 200 with the same filters", async () => {
    const seen: URL[] = [];
    vi.stubGlobal("fetch", vi.fn(async (r: Request) => {
      const u = new URL(r.url);
      seen.push(u);
      const n = Number(u.searchParams.get("limit"));
      const off = Number(u.searchParams.get("offset"));
      return jsonResponse({ scan_at: "2026-10-30T19:49:00Z", total: 250, max_results: 500, limited: false,
        setups: Array.from({ length: n }, (_, i) => setup(`T${off + i}`, 50)) });
    }));
    let saved = "";
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: (b: Blob) => { void b.text().then((t) => { saved = t; }); return "blob:x"; }, revokeObjectURL: vi.fn() }));
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    render(<ScannerCsvButton query={{ min_score: 60, min_rvol: 2 }} total={250} scanAt="2026-10-30T19:49:00Z" premium={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Download 250 scanner results as CSV" }));
    await waitFor(() => expect(click).toHaveBeenCalled());
    expect(seen.map((u) => [u.searchParams.get("limit"), u.searchParams.get("offset"), u.searchParams.get("min_score"), u.searchParams.get("min_rvol")]))
      .toEqual([["200", "0", "60", "2"], ["50", "200", "60", "2"]]);
    await waitFor(() => expect(saved.split("\r\n").filter(Boolean)).toHaveLength(251));
    click.mockRestore();
  });

  it("offers the download to plans with CSV export only", async () => {
    const latest = { scan_at: new Date().toISOString(), total: 2, max_results: 100, limited: false, setups: [setup("AAA", 88), setup("BBB", 70)] };
    vi.stubGlobal("fetch", vi.fn(async (r: Request) => jsonResponse(path(r) === "/v1/me/research-preferences" ? PREFS : latest)));
    const pro = wrap(<ScannerView />, "pro");
    expect(await screen.findByRole("button", { name: /as CSV$/ })).toBeInTheDocument();
    pro.unmount();
    clearApiCache();
    wrap(<ScannerView />, "basic");
    await screen.findAllByText("AAA");
    expect(screen.queryByRole("button", { name: /as CSV$/ })).not.toBeInTheDocument();
  });
});

describe("Scanner saved views", () => {
  const latest = { scan_at: new Date().toISOString(), total: 1, max_results: 100, limited: false, setups: [setup("AAA", 88)] };

  it("saves the price band and relative-volume filter with the view", async () => {
    search = new URLSearchParams("price=5-20&rvol=2&min=60");
    const patches: unknown[] = [];
    vi.stubGlobal("fetch", vi.fn(async (r: Request) => {
      if (path(r) !== "/v1/me/research-preferences") return jsonResponse(latest);
      if (r.method === "PATCH") { const body = await r.json(); patches.push(body); return jsonResponse({ ...PREFS, ...body }); }
      return jsonResponse(PREFS);
    }));
    wrap(<ScannerView />, "pro");
    const u = userEvent.setup();
    await screen.findAllByText("AAA");
    await u.click(screen.getByText("Saved views", { selector: "summary" }));
    await u.type(screen.getByLabelText("View name"), "Mid caps, busy");
    await u.click(screen.getByRole("button", { name: "Save view" }));
    await waitFor(() => expect(patches).toHaveLength(1));
    expect(patches[0]).toEqual({ scanner_views: [expect.objectContaining({ name: "Mid caps, busy", min: 60, price: "5-20", rvol: 2 })] });
  });

  it("restores the price band and relative-volume filter when a view is chosen", async () => {
    const view = { name: "Cheap movers", signal: "", when: "today", min: 0, sort: "score", size: 25, price: "u5", rvol: 3 };
    vi.stubGlobal("fetch", vi.fn(async (r: Request) => jsonResponse(path(r) === "/v1/me/research-preferences" ? { ...PREFS, scanner_views: [view] } : latest)));
    wrap(<ScannerView />, "pro");
    const u = userEvent.setup();
    await screen.findAllByText("AAA");
    await u.click(screen.getByText("Saved views", { selector: "summary" }));
    await u.selectOptions(await screen.findByRole("combobox", { name: "Saved views" }), "Cheap movers");
    expect(replace).toHaveBeenLastCalledWith("/scanner?price=u5&rvol=3", { scroll: false });
  });
});

describe("Watchlist prices", () => {
  it("labels a daily-bar fallback as a daily close, not a trade", async () => {
    const api = fakeApi({ scanState: true, quotes: { AAA: { last: 10, chg_pct: 1, at: "2026-10-30T20:00:00Z" } } });
    const real = api.fetch;
    vi.stubGlobal("fetch", vi.fn(async (input: Request, init?: RequestInit) => {
      if (path(input) === "/v1/quotes") {
        return jsonResponse({ feed: "iex", quotes: {
          AAA: { last: 10, chg_pct: 1, at: "2026-10-30T19:59:00Z", source: "trade" },
          BBB: { last: 20, chg_pct: -1, at: "2026-10-30T04:00:00Z", source: "daily_bar" } } });
      }
      return real(input, init);
    }));
    api.seedList("Main", ["AAA", "BBB"]);
    wrap(<WatchlistsView />, "pro");
    expect(await screen.findByText(/last trade Oct 30, 3:59 PM ET/)).toBeInTheDocument();
    expect(screen.getByText("daily close Oct 30 · no recent trade")).toBeInTheDocument();
    expect(screen.queryByText(/last trade Oct 30, 12:00 AM ET/)).not.toBeInTheDocument();
  });

  it("never calls a trade 'up to a minute old' and doesn't claim prices are unavailable while loading", () => {
    expect(priceNote(true, false)).not.toMatch(/up to a minute old/);
    expect(priceNote(true, false)).toMatch(/latest trade, with its time/);
    expect(priceNote(false, true)).toMatch(/^Loading the latest trades/);
    expect(priceNote(false, false)).toMatch(/from the scan or the daily bar, as labelled/);
  });

  it("shows a retry when the list's alert rules fail, instead of dropping the card", async () => {
    const api = fakeApi({ scanState: true, rules: true });
    api.fail("GET", /^\/v1\/alerts\/rules$/, 503);
    vi.stubGlobal("fetch", vi.fn((input: Request, init?: RequestInit) => api.fetch(input, init)));
    api.seedList("Main", ["AAA"]);
    wrap(<WatchlistsView />, "pro");
    const card = (await screen.findByRole("heading", { name: "Alerts on this list" })).closest("section")!;
    expect(within(card).getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});

describe("Research wording", () => {
  it("doesn't report 0 positive outcomes before any observation has matured", () => {
    expect(observationLine("BTSG", { observations: 3, matured: 0, positive: 0 })).toBe("BTSG: 3 saved scan observations · none matured yet, so there is no outcome to report.");
    expect(observationLine("BTSG", { observations: 9, matured: 4, positive: 1 })).toBe("BTSG: 9 saved scan observations · 4 matured · 1 of 4 reached +4% within 5 trading days.");
    expect(observationLine("X", { observations: 1 })).toBe("X: 1 saved scan observation · outcome counts unavailable.");
  });

  it("describes sample size in words without calling it evidence", () => {
    expect(sampleCoverage(111)).toMatch(/Sample size: large \(100\+\); this describes how many observations there are, not how reliable/);
    expect(sampleCoverage(5)).toMatch(/too few to read/);
    expect(sampleCoverage(111)).not.toMatch(/evidence/i);
    expect(sampleCoverage(null)).toBe("");
  });

  it("puts a retry on Similar past setups when it fails", async () => {
    let calls = 0;
    vi.stubGlobal("fetch", vi.fn(async () => (++calls === 1 ? jsonResponse({ detail: "down" }, 503)
      : jsonResponse({ horizons: [{ horizon: 1, matured_count: 0, evidence_quality: "INSUFFICIENT" }] }))));
    render(<SimilarSetups score={78} setup="golden_cross" setupName="Golden cross" />);
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/No matured past Golden cross setups/)).toBeInTheDocument();
  });

  it("Track record says it is a historical description and drops the best-slice headline", async () => {
    vi.stubGlobal("fetch", vi.fn(async (r: Request) => {
      if (path(r) === "/v1/track-record") {
        return jsonResponse({ disclaimer: "Descriptive.", min_sample_size: 25, summary: { best_ranking: "BreakoutScore", best_horizon_days: 1, best_avg_excess_return: -0.001, ready_horizons: 5 },
          summaries: [{ ranking: "breakout", ranking_label: "BreakoutScore", horizon_days: 1, avg_excess_return: -0.001, median_excess_return: 0, win_rate: 0.49, sample_size: 148, runs_used: 30, sufficient: true }] });
      }
      return jsonResponse({ detail: "Not Found" }, 404);
    }));
    wrap(<TrackRecordView />, "pro");
    expect(screen.getByText(/A historical description, not validated forward evidence/)).toBeInTheDocument();
    expect(await screen.findByText(/Ranking study: top Breakout score picks/)).toBeInTheDocument();
    expect(screen.queryByText("First read")).not.toBeInTheDocument();
    expect(screen.queryByText(/highest observed average/)).not.toBeInTheDocument();
    expect(screen.getByText(/Historical description vs forward evidence/)).toBeInTheDocument();
  });
});

describe("Landing preview", () => {
  it("only says 'today' for an on-time scan from today", () => {
    const fri = "2026-10-30T19:49:00Z";
    expect(previewTitle(fri, false, Date.parse("2026-10-30T21:00:00Z"))).toEqual({ title: "Today's top setups", today: true });
    expect(previewTitle(fri, false, Date.parse("2026-11-01T16:00:00Z"))).toEqual({ title: "Top setups from the latest scan", today: false });
    expect(previewTitle(fri, true, Date.parse("2026-10-30T21:00:00Z")).today).toBe(false);
  });

  it("says its prices are from the scan", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ scan_at: "2026-10-30T19:49:00Z", stale: false, ranked_total: 40, free_rows: 25,
      setups: [{ ticker: "AAA", hsf_score: 91, setup: "breakout", change_pct: 2.1 }] })));
    render(<LiveSetupsPreview />);
    expect(await screen.findByText(/Prices and changes are from the scan, not live quotes/)).toBeInTheDocument();
  });
});
