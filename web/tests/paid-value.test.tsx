import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SimilarSetups } from "@/features/OutcomeEvidence";
import { ScanAlertButton, describeScanAlert } from "@/features/ScanAlert";
import { describeRule } from "@/features/SmartAlerts";
import { WatchlistAINote } from "@/features/WatchlistAINote";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/scanner",
}));

const calls: { method: string; path: string; query: URLSearchParams; body: unknown }[] = [];
let routes: Record<string, (body: unknown) => unknown> = {};
beforeEach(() => {
  calls.length = 0;
  routes = {};
  vi.stubGlobal("fetch", async (req: Request) => {
    const url = new URL(req.url);
    const path = url.pathname.replace(/^\/api\/hsf/, "");
    const text = req.method === "GET" ? "" : await req.text();
    const body = text ? JSON.parse(text) : null;
    calls.push({ method: req.method, path, query: url.searchParams, body });
    const h = routes[`${req.method} ${path}`];
    return h ? jsonResponse(h(body)) : jsonResponse({ detail: "Not Found" }, 404);
  });
});
afterEach(() => vi.unstubAllGlobals());

const metric = (horizon: number, n: number, med: number, q = "MODERATE") => ({
  horizon, matured_count: n, pending_count: 0, median_return: med, median_excess_return: med / 2, benchmark_count: n,
  benchmark_beat_count: Math.round(n / 2), benchmark_beat_rate: 0.5, win_rate: 0.6, average_mfe: null, average_mae: null,
  evidence_quality: q,
});

describe("Similar past setups (Pro)", () => {
  it("asks for the same setup within 5 points and shows every horizon", async () => {
    routes["GET /v1/outcomes/horizons"] = () => ({ horizons: [metric(1, 40, 0.004), metric(3, 38, 0.011), metric(5, 6, 0.02, "INSUFFICIENT")] });
    render(<SimilarSetups score={77} setup="breakout" setupName="Breakout" />);
    expect(await screen.findByText(/how earlier Breakout setups scoring 72–82 moved/)).toBeInTheDocument();
    const q = calls[0]!.query;
    expect([q.get("setup"), q.get("min_score"), q.get("max_score")]).toEqual(["breakout", "72", "82"]);
    expect(screen.getByText("+1.10%")).toBeInTheDocument();
    expect(screen.getByText("5 days").closest("tr")).toHaveClass("row-muted");
  });

  it("says so when nothing like it has matured", async () => {
    routes["GET /v1/outcomes/horizons"] = () => ({ horizons: [metric(1, 0, 0), metric(3, 0, 0), metric(5, 0, 0)] });
    render(<SimilarSetups score={99} />);
    expect(await screen.findByText("No matured past HSF signals scoring 94–100 yet.")).toBeInTheDocument();
  });
});

const scanType = { type: "SCAN_MATCH", label: "New scanner matches", description: "d", operator: "new_matches", kind: "transition",
  threshold: { min: 0, max: 100, default: 75 }, takes_value: true, scope: "market", default_cooldown_seconds: 0, available: true };
const ruleList = (used = 1, limit = 5) => ({ limit, used, rules: [],
  capabilities: { tier: "pro", max_watchlists: 10, max_symbols_per_request: 50, max_active_alerts: limit, alert_rule_types: [], delivery_channels: ["in_app", "email"] } });

describe("Scan alerts (Pro)", () => {
  it("saves the Scanner's filters as a market-wide rule", async () => {
    routes["GET /v1/alerts/rules/types"] = () => [scanType];
    routes["GET /v1/alerts/rules"] = () => ruleList();
    routes["POST /v1/alerts/rules"] = (b) => ({ id: 9, ...(b as object) });
    const u = userEvent.setup();
    render(<ScanAlertButton setup="breakout" minScore={75} setupLabel="Breakout" />);
    await u.click(screen.getByRole("button", { name: "Alert me on new matches" }));
    await u.type(await screen.findByLabelText("Name (optional)"), "Big breakouts");
    await u.type(screen.getByLabelText("Min price ($)"), "5");
    await u.click(screen.getByLabelText("Also email me"));
    await u.click(screen.getByRole("button", { name: "Save scan alert" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.body).toEqual({ rule_type: "SCAN_MATCH", threshold: 75, enabled: true, delivery_channels: ["in_app", "email"],
      value: JSON.stringify({ name: "Big breakouts", setup: "breakout", min_price: 5 }) });
    expect(await screen.findByText(/Created: Big breakouts: HSF 75\+, Breakout, \$5\+/)).toBeInTheDocument();
  });

  it("is locked below Pro", async () => {
    routes["GET /v1/alerts/rules/types"] = () => [{ ...scanType, available: false }];
    routes["GET /v1/alerts/rules"] = () => ruleList(0, 1);
    const u = userEvent.setup();
    render(<ScanAlertButton setup="" minScore={0} />);
    await u.click(screen.getByRole("button", { name: "Alert me on new matches" }));
    expect(await screen.findByText("Scan alerts are part of Pro")).toBeInTheDocument();
  });

  it("describes saved scan alerts in the rule list", () => {
    expect(describeScanAlert(60, '{"setup":"gapper","min_price":2,"max_price":20,"min_rvol":3}')).toBe("HSF 60+, Gapper, $2 to $20, 3x volume");
    expect(describeRule({ rule_type: "SCAN_MATCH", threshold: 80, value: null }, [])).toBe("New scanner matches (HSF 80+)");
  });
});

describe("AI watchlist rundown (Premium)", () => {
  it("Premium: writes the rundown on request", async () => {
    routes["POST /v1/ai/watchlists/3/note"] = () => ({ snapshot_time: "2026-10-20T14:35:00+00:00", text: "AAA gained 12 points." });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("premium")}><WatchlistAINote id={3} name="Core" empty={false} /></SessionProvider>);
    await u.click(await screen.findByRole("button", { name: "Write AI rundown" }));
    expect(await screen.findByText("AAA gained 12 points.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh rundown" })).toBeInTheDocument();
  });

  it("Pro sees it locked", async () => {
    render(<SessionProvider initialMe={me("pro")}><WatchlistAINote id={3} name="Core" empty={false} /></SessionProvider>);
    expect(await screen.findByText("AI watchlist rundowns are part of Premium")).toBeInTheDocument();
    expect(calls.some((c) => c.path.includes("/v1/ai/"))).toBe(false);
  });
});
