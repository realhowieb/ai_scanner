import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { AppShell } from "@/components/AppShell";
import { AdminView } from "@/features/AdminView";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/admin",
}));

const admin = (): Schemas["Me"] => ({ ...me("basic"), email: "boss@example.com", is_admin: true });
let users: Schemas["AdminUser"][] = [];
const calls: { method: string; path: string; body: unknown }[] = [];

const day = (i: number) => `2026-10-${String(i).padStart(2, "0")}`;
const analytics = {
  loaded_at: "2026-10-14T12:00:00Z", days: 7, unavailable: [],
  overview: { total_users: 42, active_users_30d: 17, new_users_7d: 3, scans_today: 6, paying_users: 5 },
  systems: [{ system: "Scanner", status: "HEALTHY", reason: "All slots ran" }],
  users_daily: [8, 9, 10, 11, 12, 13, 14].map((d, i) => ({ day: day(d), new: i ? 1 : 0, total: 36 + i })),
  plans: [{ plan: "Basic", users: 30 }, { plan: "Pro", users: 4 }, { plan: "Premium", users: 1 }, { plan: "Admin", users: 0 }],
  scans_daily: [8, 9, 10, 11, 12, 13, 14].map((d) => ({ day: day(d), scheduled: 6, user: 1 })),
  conversions: { signups: 4, paid: 1, signup_to_paid: 0.25, destinations: [{ plan: "Pro", users: 1 }, { plan: "Premium", users: 0 }] },
  recovery: [{ started_at: "2026-10-13T12:00:00Z", incident: "SCAN_MISSED", action: "rerun scan", result: "recovered" }],
};
const group = (key: string, column: string, n: number, median: number) => ({ key, column, n, matured: n, sufficient: n >= 30,
  positive_rate: 0.6, average_return: median, median_return: median, average_mfe: 0.05, average_mae: -0.02 });
function signals(q: URLSearchParams) {
  const breakout = q.get("signal") === "Breakout";
  return { days: 30, min_n: Number(q.get("min_n") ?? 30), loaded_observations: 20000, truncated: true,
    options: { signals: ["Breakout", "Momentum"], cohorts: ["CANDIDATE"], horizons: ["1d", "5d"], sessions: ["REGULAR"], score_low: 40, score_high: 95 },
    summary: { observations: breakout ? 777 : 50, matured: 40, n: 40, sufficient: true, positive_rate: 0.6, average_return: 0.012, median_return: 0.008, average_mfe: 0.05, average_mae: -0.02 },
    by_signal_horizon: [group("Breakout", "5d", 35, 0.021), group("Momentum", "5d", 4, -0.01)],
    score_available: true, score_horizon: "1d", score_buckets: [group("80-89", "", 31, 0.015)], score_matrix: [group("80-89", "5d", 31, 0.015)] };
}

async function handle(input: Request | string, init?: RequestInit): Promise<Response> {
  const req = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
  const path = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
  const text = req.method === "GET" ? "" : await req.text();
  const body = text ? JSON.parse(text) : null;
  calls.push({ method: req.method, path, body });
  if (req.method === "GET" && path === "/v1/admin/users") return jsonResponse({ users });
  if (req.method === "GET" && path === "/v1/admin/audit") return jsonResponse({ events: [{ at: "2026-10-14T12:00:00Z", actor: "boss@example.com", action: "grant admin", target: "x@example.com", details: "" }] });
  if (path === "/v1/admin/users/update") {
    const b = body as { username: string; tier?: string };
    users = users.map((u) => (u.username === b.username ? { ...u, tier: b.tier ?? u.tier } : u));
    return jsonResponse({ ok: true, message: `Updated ${b.username}.` });
  }
  if (path === "/v1/admin/users/role") return jsonResponse({ detail: "You can't remove your own admin role." }, 422);
  if (path === "/v1/admin/health") return jsonResponse({ operations: { system_status: "DEGRADED", generated_at: "2026-10-14T12:00:00Z", stale: false, workflows: [] },
    health_score: 80, human_action: "NONE", next_expected_scan: null, subsystems: [{ name: "Scanner", status: "HEALTHY", label: "ok", reason: "All slots ran" }],
    incidents: [], email: { status: "HEALTHY", day: "2026-10-14", jobs: [{ job: "digest", label: "Morning digest", status: "HEALTHY", detail: "sent 3" }] } });
  if (path === "/v1/admin/providers") return jsonResponse({ checked_at: "2026-10-14T12:00:00Z", earnings_cache: { last_refresh: null, dated_rows: 10 },
    checks: [{ name: "Database", status: "ok", detail: "neon", ms: 5 }, { name: "Billing service", status: "error", detail: "down", ms: null }] });
  if (req.method === "GET" && path === "/v1/admin/earnings-refresh") return jsonResponse({ state: "idle" });
  if (req.method === "POST" && path === "/v1/admin/earnings-refresh") return jsonResponse({ state: "done", symbols: 600, dated: 420, finished_at: "2026-10-14T12:00:00Z" });
  if (path === "/v1/admin/analytics") return jsonResponse(analytics);
  if (path === "/v1/admin/signals") return jsonResponse(signals(new URL(req.url).searchParams));
  if (path === "/v1/admin/evidence") return jsonResponse({ days: 30, funnel: [{ stage: "Captured", count: 40, pct: 100 }, { stage: "Matured", count: 30, pct: 75 }],
    pending: 10, excluded: 5, cohorts: [{ cohort: "CANDIDATE", observations: 25 }], readiness: { state: "COLLECTING", reason: "Need 20 trading days", trading_days: 9, trading_days_preferred: 20 } });
  if (path === "/v1/admin/stair-stepper") return jsonResponse({ days: 30, observations: 0, horizons: ["+5m", "+10m"], windows: [], best_window: null, verdict_reason: "Requires 30 matured outcomes." });
  if (path === "/v1/admin/ml-readiness") return jsonResponse({ status: "NOT_READY", recommendation: "Keep collecting.", matured_signal_days: 134, matured_rows: 600,
    matured_trading_days: 12, matured_symbols: 80, outcome_coverage_recent: 0.9, walk_forward_folds: 0, blocking_reasons: ["Matured signal-days 134 < 1000"],
    gates: [{ gate: "Matured signal-days", kind: "Accumulating", value: 134, needs: ">= 1000", status: "FAIL", progress: 0.134 }] });
  return jsonResponse({ detail: `no fake for ${req.method} ${path}` }, 404);
}

beforeEach(() => {
  calls.length = 0;
  users = [
    { username: "boss@example.com", full_name: "Boss", tier: "basic", is_active: true, is_admin: true, email_verified: true, created_at: null },
    { username: "pro@example.com", full_name: "Pro User", tier: "pro", is_active: true, is_admin: false, email_verified: false, created_at: null },
  ];
  vi.stubGlobal("fetch", vi.fn(handle));
});
afterEach(() => vi.unstubAllGlobals());

const wrap = (m: Schemas["Me"]) => render(<SessionProvider initialMe={m}><AdminView /></SessionProvider>);

describe("Admin page", () => {
  it("is for admins only, and only admins see it in the account menu", async () => {
    const u = userEvent.setup();
    const { unmount } = wrap(me("pro"));
    expect(screen.getByText("Admins only.")).toBeInTheDocument();
    expect(calls).toHaveLength(0);
    unmount();
    const shell = (m: Schemas["Me"]) => render(<SessionProvider initialMe={m}><AppShell><p>body</p></AppShell></SessionProvider>);
    const a = shell(me("pro"));
    await u.click(screen.getByRole("button", { name: /Account/ }));
    expect(within(screen.getByRole("menu")).queryByRole("menuitem", { name: "Admin" })).not.toBeInTheDocument();
    a.unmount();
    shell(admin());
    await u.click(screen.getByRole("button", { name: /Account/ }));
    expect(within(screen.getByRole("menu")).getByRole("menuitem", { name: "Admin" })).toHaveAttribute("href", "/admin");
  });

  it("lists accounts, changes a plan and shows refusals", async () => {
    const u = userEvent.setup();
    wrap(admin());
    expect(await screen.findByText("pro@example.com")).toBeInTheDocument();
    expect(screen.getByText("2 accounts · 1 admins · 1 unverified")).toBeInTheDocument();
    await u.type(screen.getByRole("searchbox"), "pro");
    expect(screen.queryByText("boss@example.com", { selector: "strong" })).not.toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Manage pro@example.com" }));
    const dialog = screen.getByRole("dialog");
    await u.selectOptions(within(dialog).getByRole("combobox"), "premium");
    await u.click(within(dialog).getByRole("button", { name: "Save changes" }));
    expect(await within(dialog).findByText("Updated pro@example.com.")).toBeInTheDocument();
    expect(calls.find((c) => c.path === "/v1/admin/users/update")?.body).toEqual({ username: "pro@example.com", tier: "premium" });
    expect(within(dialog).getByRole("button", { name: "Mark email verified" })).toBeInTheDocument();
    await u.click(within(dialog).getByRole("button", { name: "Make admin" }));
    await u.click(screen.getAllByRole("button", { name: "Make admin" }).at(-1)!);
    expect(await screen.findByText("You can't remove your own admin role.")).toBeInTheDocument();
  });

  it("shows product analytics and recovery", async () => {
    const u = userEvent.setup();
    wrap(admin());
    await u.click(screen.getByRole("tab", { name: "Analytics" }));
    expect(await screen.findByText("Total users")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("25.0%")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Scans per day: Scheduled 42, User 7/ })).toBeInTheDocument();
    expect(screen.getByText("SCAN_MISSED")).toBeInTheDocument();
    await u.selectOptions(screen.getByRole("combobox", { name: "Period" }), "90");
    await waitFor(() => expect(calls.some((c) => c.path === "/v1/admin/analytics" && c.method === "GET")).toBe(true));
  });

  it("shows signal performance, filters it, and the research evidence", async () => {
    const u = userEvent.setup();
    wrap(admin());
    await u.click(screen.getByRole("tab", { name: "Research" }));
    expect(await screen.findByText("+2.1%")).toBeInTheDocument();
    expect(screen.getByText("n=4")).toBeInTheDocument();
    expect(screen.getByText(/Based on the newest 20,000 observations/)).toBeInTheDocument();
    await u.selectOptions(screen.getByRole("combobox", { name: "Signal" }), "Breakout");
    expect(await screen.findByText("777")).toBeInTheDocument();
    expect(await screen.findByText("Need 20 trading days")).toBeInTheDocument();
    expect(screen.getByText("No Stair-Stepper outcomes have matured in this period.")).toBeInTheDocument();
    expect(await screen.findByText("Keep collecting.")).toBeInTheDocument();
    expect(screen.getByText(">= 1000")).toBeInTheDocument();
  });

  it("shows health, runs service checks and the earnings refresh", async () => {
    const u = userEvent.setup();
    wrap(admin());
    await u.click(screen.getByRole("tab", { name: "Health" }));
    expect(await screen.findByText("All slots ran")).toBeInTheDocument();
    expect(screen.getByText("Morning digest")).toBeInTheDocument();
    expect(calls.some((c) => c.path === "/v1/admin/providers")).toBe(false);
    await u.click(screen.getByRole("button", { name: "Run checks" }));
    expect(await screen.findByText("Billing service")).toBeInTheDocument();
    await u.click(screen.getByRole("tab", { name: "Tools" }));
    await u.click(await screen.findByRole("button", { name: "Refresh earnings now" }));
    await waitFor(() => expect(screen.getByText(/dates found for 420 of 600 symbols/)).toBeInTheDocument());
  });
});
