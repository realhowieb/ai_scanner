import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync, writeFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { BriefView } from "@/features/BriefView";
import { SavedScanView, ScanHistoryView } from "@/features/ScanHistoryView";
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
beforeEach(() => {
  search = new URLSearchParams();
  replace.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

const wrap = (ui: React.ReactNode, plan: "basic" | "pro" | "premium") => render(<SessionProvider initialMe={me(plan)}>{ui}</SessionProvider>);
const route = (map: Record<string, () => Response>) => fetchMock.mockImplementation(async (req: Request) => {
  const p = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
  const hit = Object.entries(map).find(([k]) => p === k);
  return hit ? hit[1]() : jsonResponse({ detail: `no fake for ${p}` }, 404);
});

const brief = (over: Partial<Schemas["Brief"]> = {}): Schemas["Brief"] => ({
  available: true, snapshot_time: "2026-10-07T14:00:00+00:00", phase: "regular",
  market: [{ label: "S&P 500", last: 6012.3, chg_pct: 0.42 }], breadth: { advancers: 310, decliners: 190 },
  sectors: [{ sector: "Technology", chg_pct: 1.1 }], has_previous_snapshot: true,
  opportunities: [{ ticker: "NVDA", score: 82, status: "STRONG", movement_state: "RISING", score_delta: 5 }],
  gappers: [{ ticker: "XYZ", last: 10, chg_pct: 6, gap_pct: 5.5, earnings_days: 1 }], gainers: [{ ticker: "UP", chg_pct: 9 }], losers: [],
  golden_crosses: ["GC"], top_breakout_scores: [{ ticker: "BO", score: 91 }], prebreakout_picks: [], prebreakout_locked: true, earnings_today: [], ...over,
});

describe("Market Brief", () => {
  it.each([["premarket", "Pre-market"], ["regular", "Market open"], ["afterhours", "After hours"], ["closed", "Market closed"]])("uses the backend %s session", async (phase, label) => {
    route({ "/v1/brief": () => jsonResponse(brief({ phase })) });
    wrap(<BriefView />, "basic");
    expect(await screen.findByText(new RegExp(`${label} · snapshot`))).toBeInTheDocument();
  });
  it("renders the snapshot, keeps sectors as text, and locks Premium and Pro sections on Free", async () => {
    route({ "/v1/brief": () => jsonResponse(brief()) });
    wrap(<BriefView />, "basic");
    expect(await screen.findByText(/Market open · snapshot/)).toBeInTheDocument();
    expect(screen.getByText("S&P 500")).toBeInTheDocument();
    const opps = screen.getByRole("region", { name: "HSF Opportunity Radar" });
    expect(within(opps).getByRole("link", { name: "NVDA" })).toBeInTheDocument();
    expect(within(opps).getByText("Rising +5")).toBeInTheDocument();
    const sectors = screen.getByRole("region", { name: "Sectors" });
    expect(within(sectors).queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("PreBreakout is part of Premium")).toBeInTheDocument();
    expect(screen.getByText("The earnings calendar is part of Pro")).toBeInTheDocument();
    expect(urls().map((u) => u.pathname)).toEqual(["/api/hsf/v1/brief"]);            // no Pro calls on Free
    expect(screen.queryByRole("region", { name: "AI market narrative" })).not.toBeInTheDocument();
  });

  it("Pro gets the earnings calendar; not-ready and failed briefs are explained", async () => {
    route({ "/v1/brief": () => jsonResponse(brief()), "/v1/earnings": () => jsonResponse([{ ticker: "ERN", earnings_date: "2026-10-09", days_until: 1, time: "amc" }]) });
    const { unmount } = wrap(<BriefView />, "pro");
    expect(await screen.findByText("After close")).toBeInTheDocument();
    expect(screen.getByText("Tomorrow")).toBeInTheDocument();
    unmount();
    route({ "/v1/brief": () => jsonResponse({ available: false, snapshot_time: null }) });
    const second = wrap(<BriefView />, "pro");
    expect(await screen.findByText("Today's brief isn't ready yet.")).toBeInTheDocument();
    second.unmount();
    route({ "/v1/brief": () => jsonResponse({ detail: "x" }, 503) });
    wrap(<BriefView />, "pro");
    expect(await screen.findByText("Couldn't load the Market Brief right now")).toBeInTheDocument();
  });

  it("Premium: PreBreakout picks and an AI narrative on request", async () => {
    route({
      "/v1/brief": () => jsonResponse(brief({ prebreakout_locked: false, prebreakout_picks: [{ ticker: "PB", prob: 41.6 }] })),
      "/v1/earnings": () => jsonResponse([]),
      "/v1/ai/brief-narrative": () => jsonResponse({ text: "Tech led today.", run_id: null, snapshot_time: null, ticker: null }),
    });
    wrap(<BriefView />, "premium");
    expect(await screen.findByText("42%")).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: "Write narrative" }));
    expect(await screen.findByText("Tech led today.")).toBeInTheDocument();
  });

  it("keeps the Pulse and Radar available when earnings fails", async () => {
    route({ "/v1/brief": () => jsonResponse(brief()), "/v1/earnings": () => jsonResponse({ detail: "unavailable" }, 503) });
    wrap(<BriefView />, "pro");
    expect(await screen.findByText("Couldn't load the earnings calendar right now")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Market Pulse" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "View Scanner →" })).toHaveAttribute("href", "/scanner");
  });

  it("reserves separate loading panels for Pulse and Radar", () => {
    fetchMock.mockReturnValue(new Promise(() => {}));
    wrap(<BriefView />, "basic");
    expect(screen.getByText("Loading Market Pulse…")).toBeInTheDocument();
    expect(screen.getByText("Loading Opportunity Radar…")).toBeInTheDocument();
  });

  it("renders a representative responsive fixture without live services", async () => {
    route({ "/v1/brief": () => jsonResponse(brief({
      market: ["SPY", "QQQ", "IWM", "VIX"].map((label, i) => ({ label, last: 100 + i, chg_pct: i - 1.5 })),
      opportunities: Array.from({ length: 5 }, (_, i) => ({ ticker: `TEST${i}`, score: 85 - i * 4, score_delta: i - 2, primary_setup: i === 0 ? "Long setup name for responsive wrapping verification" : "Momentum", status: "WATCH" })),
      gainers: Array.from({ length: 6 }, (_, i) => ({ ticker: `GAIN${i}`, chg_pct: i + 1 })),
    })), "/v1/earnings": () => jsonResponse([]) });
    const view = wrap(<BriefView />, "premium");
    await screen.findByText("TEST0");
    await screen.findByText("No earnings on file for the next 7 days.");
    expect(view.container.querySelector(".brief-dashboard")?.firstElementChild).toHaveClass("brief-center");
    // Opt-in HTML export for browser layout QA. Fixtures are never shipped by the app.
    if (process.env.BRIEF_QA_HTML) writeFileSync(process.env.BRIEF_QA_HTML,
      `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Brief fixture QA</title><style>${readFileSync("src/app/globals.css", "utf8")}</style><main class="main">${view.container.innerHTML}</main></html>`);
  });
});

describe("Scan history", () => {
  it("is locked below Pro without calling the API", () => {
    wrap(<ScanHistoryView />, "basic");
    expect(screen.getByText("Scan history is part of Pro")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("lists saved scans and opens one", async () => {
    route({ "/v1/runs": () => jsonResponse([{ id: 7, name: "x", label: "SP500", row_count: 40, duration_s: 61.2, is_snapshot: false, created_at: "2026-10-07T14:00:00+00:00" }]) });
    const { unmount } = wrap(<ScanHistoryView />, "pro");
    expect(await screen.findByRole("link", { name: "SP500" })).toHaveAttribute("href", "/scanner/history/7");
    unmount();
    route({ "/v1/runs": () => jsonResponse([]) });
    const b = wrap(<ScanHistoryView />, "pro");
    expect(await screen.findByText("No saved scans yet.")).toBeInTheDocument();
    b.unmount();
    route({ "/v1/runs/7": () => jsonResponse({ id: 7, label: "SP500", name: null, row_count: 2, duration_s: 3, is_snapshot: false, created_at: null, total: 2, max_results: 100, limited: false, setups: [setup("AAA", 80)] }) });
    const c = wrap(<SavedScanView id={7} />, "pro");
    expect((await screen.findAllByRole("link", { name: "AAA" })).length).toBeGreaterThan(0);
    c.unmount();
    route({});
    wrap(<SavedScanView id={8} />, "pro");
    expect(await screen.findByText("Saved scan not found.")).toBeInTheDocument();
  });
});

describe("Scanner sort", () => {
  const latest = { scan_at: new Date().toISOString(), total: 2, max_results: 100, limited: false, setups: [setup("AAA", 88), setup("BBB", 70)] };

  it("sends the sort to the API, drops the rank numbers and says what is sorted", async () => {
    search = new URLSearchParams("sort=chg_pct");
    fetchMock.mockResolvedValue(jsonResponse(latest));
    wrap(<ScannerView />, "pro");
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(urls()[0]!.searchParams.get("sort")).toBe("chg_pct");
    expect(await screen.findByText(/Sorted by change % within your plan's top 2/)).toBeInTheDocument();
    expect(screen.queryByText("#1")).not.toBeInTheDocument();
  });

  it("ignores a PreBreakout sort below Premium and keeps HSF order by default", async () => {
    search = new URLSearchParams("sort=prob");
    fetchMock.mockResolvedValue(jsonResponse(latest));
    wrap(<ScannerView />, "pro");
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(urls()[0]!.searchParams.has("sort")).toBe(false);
    expect(screen.queryByRole("option", { name: "PreBreakout" })).not.toBeInTheDocument();
    await userEvent.setup().selectOptions(screen.getByLabelText("Sort by"), "rvol");
    expect(replace).toHaveBeenCalledWith("/scanner?sort=rvol", { scroll: false });
  });
});
