import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { AccountView } from "@/features/AccountView";
import { TrackRecordView, signedPct } from "@/features/TrackRecordView";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/account",
}));

type Route = (req: Request, body: unknown) => Response | Promise<Response> | undefined;
let routes: Route[] = [];
const calls: { method: string; path: string; search: string; body: unknown }[] = [];
const assign = vi.fn();

async function handle(input: Request | string, init?: RequestInit): Promise<Response> {
  const req = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
  const text = req.method === "GET" ? "" : await req.text();
  const body = text ? JSON.parse(text) : null;
  const path = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
  calls.push({ method: req.method, path, search: new URL(req.url).search, body });
  for (const r of routes) {
    const out = await r(req, body);
    if (out) return out;
  }
  return jsonResponse({ detail: `no fake for ${req.method} ${path}` }, 404);
}

const on = (method: string, path: string | RegExp, fn: (body: unknown, req: Request) => Response): Route => (req, body) => {
  const p = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
  return req.method === method && (typeof path === "string" ? p === path : path.test(p)) ? fn(body, req) : undefined;
};

let prefs = { digest: true, evening: false, alerts: true };
beforeEach(() => {
  calls.length = 0;
  assign.mockReset();
  prefs = { digest: true, evening: false, alerts: true };
  routes = [
    on("GET", "/v1/me/email-preferences", () => jsonResponse(prefs)),
    on("PATCH", "/v1/me/email-preferences", (b) => { Object.assign(prefs, b); return jsonResponse(prefs); }),
  ];
  vi.stubGlobal("fetch", vi.fn(handle));
  vi.stubGlobal("location", { ...window.location, origin: "http://localhost", pathname: "/account", search: "", assign });
});
afterEach(() => vi.unstubAllGlobals());

const user = () => userEvent.setup();
const wrap = (ui: React.ReactNode, m: Schemas["Me"]) => render(<SessionProvider initialMe={m}>{ui}</SessionProvider>);

describe("Account", () => {
  it("Free: identity, plan from /v1/me, what is included and both upgrades", async () => {
    wrap(<AccountView />, me("basic"));
    expect(screen.getByRole("heading", { name: "Account" })).toBeInTheDocument();
    expect(screen.getByText("basic@example.invalid")).toBeInTheDocument();
    const plan = screen.getByRole("region", { name: /Plan/ });
    expect(within(plan).getByText("Free")).toBeInTheDocument();
    expect(within(plan).getByRole("button", { name: "Upgrade to Pro" })).toBeInTheDocument();
    expect(within(plan).getByRole("button", { name: "Upgrade to Premium" })).toBeInTheDocument();
    expect(within(plan).queryByRole("button", { name: "Manage subscription" })).not.toBeInTheDocument();
    expect(within(plan).getByText("S&P 500 scans")).toBeInTheDocument();
    expect(within(plan).getByText("PreBreakout probability")).toBeInTheDocument();   // listed under "Not on your plan"
    expect(within(plan).getByText(/Up to 1 alert\./)).toBeInTheDocument();
  });

  it("Pro: manage subscription opens the Stripe portal; no subscription yet is explained", async () => {
    routes.unshift(on("POST", "/v1/billing/portal", (b) => (b && (b as { flow?: string }).flow === "cancel"
      ? jsonResponse({ detail: "No subscription to manage yet. Choose a plan first." }, 404)
      : jsonResponse({ url: "https://billing.stripe.com/p/x", mode: "portal" }))));
    const u = user();
    wrap(<AccountView />, me("pro"));
    const plan = screen.getByRole("region", { name: /Plan/ });
    expect(within(plan).queryByRole("button", { name: "Upgrade to Pro" })).not.toBeInTheDocument();
    await u.click(within(plan).getByRole("button", { name: "Manage subscription" }));
    await waitFor(() => expect(assign).toHaveBeenCalledWith("https://billing.stripe.com/p/x"));
    await u.click(within(plan).getByRole("button", { name: "Cancel subscription" }));
    expect(await within(plan).findByRole("alert")).toHaveTextContent("No subscription to manage yet");
  });

  it("unverified: says why and resends the verification email", async () => {
    routes.unshift(on("POST", "/v1/me/verify-email", () => jsonResponse({ ok: true, message: "Verification email sent. Check your inbox (and spam)." })));
    const u = user();
    wrap(<AccountView />, { ...me("basic"), email_verified: false });
    expect(screen.getByText("Not verified")).toBeInTheDocument();
    expect(screen.getByText("Verify your email before upgrading.")).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Send verification email" }));
    expect(await screen.findByText(/Verification email sent/)).toBeInTheDocument();
  });

  it("email settings show the server's answer and keep it on a failed save", async () => {
    const u = user();
    wrap(<AccountView />, me("pro"));
    const alerts = await screen.findByRole("checkbox", { name: /Alert emails/ });
    expect(alerts).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /Evening market wrap/ })).not.toBeChecked();
    await u.click(alerts);
    await waitFor(() => expect(alerts).not.toBeChecked());
    expect(calls.find((c) => c.method === "PATCH")!.body).toEqual({ alerts: false });
    routes.unshift(on("PATCH", "/v1/me/email-preferences", () => jsonResponse({ detail: "The HSF service is unavailable right now." }, 503)));
    await u.click(alerts);
    expect(await screen.findByText("The HSF service is unavailable right now.")).toBeInTheDocument();
    expect(alerts).not.toBeChecked();
  });

  it("email settings: loading, then an error with retry", async () => {
    routes.unshift(on("GET", "/v1/me/email-preferences", () => jsonResponse({ detail: "boom" }, 503)));
    wrap(<AccountView />, me("pro"));
    expect(screen.getByRole("status")).toHaveTextContent("Loading email settings");
    expect(await screen.findByText("Couldn't load your email settings right now")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });

  it("changes the password through this site's auth route, never the proxy", async () => {
    let n = 0;
    routes.unshift(on("POST", "/api/auth/password", () => (++n === 1 ? jsonResponse({ detail: "Current password is incorrect." }, 400) : jsonResponse({ ok: true }))));
    const u = user();
    wrap(<AccountView />, me("pro"));
    const card = screen.getByRole("region", { name: "Password" });
    await u.type(within(card).getByLabelText("Current password"), "old");
    await u.type(within(card).getByLabelText("New password"), "new-pass-1");
    await u.type(within(card).getByLabelText("Repeat new password"), "new-pass-2");
    expect(within(card).getByText("The new passwords don't match.")).toBeInTheDocument();
    expect(within(card).getByRole("button", { name: "Change password" })).toBeDisabled();
    await u.clear(within(card).getByLabelText("Repeat new password"));
    await u.type(within(card).getByLabelText("Repeat new password"), "new-pass-1");
    await u.click(within(card).getByRole("button", { name: "Change password" }));
    expect(await within(card).findByRole("alert")).toHaveTextContent("Current password is incorrect.");
    await u.click(within(card).getByRole("button", { name: "Change password" }));
    expect(await within(card).findByText(/Password changed/)).toBeInTheDocument();
    expect(calls.filter((c) => c.path === "/api/auth/password")[1]!.body).toEqual({ current_password: "old", new_password: "new-pass-1" });
    expect(calls.some((c) => c.path === "/v1/me/password")).toBe(false);
  });

  it("deletes the account only with the password and DELETE typed, and explains a refusal", async () => {
    let n = 0;
    routes.unshift(
      on("DELETE", "/v1/me", () => (++n === 1 ? jsonResponse({ detail: "Cancel your paid subscription first." }, 409) : new Response(null, { status: 204 }))),
      on("POST", "/api/auth/logout", () => new Response(null, { status: 204 })),
    );
    const u = user();
    wrap(<AccountView />, me("pro"));
    await u.click(screen.getByRole("button", { name: "Delete account…" }));
    const dlg = screen.getByRole("dialog", { name: "Delete your account?" });
    const go = within(dlg).getByRole("button", { name: "Delete account" });
    await u.type(within(dlg).getByLabelText("Password"), "pw");
    expect(go).toBeDisabled();
    await u.type(within(dlg).getByLabelText("Type DELETE to confirm"), "DELETE");
    await u.click(go);
    expect(await within(dlg).findByRole("alert")).toHaveTextContent("Cancel your paid subscription first.");
    expect(assign).not.toHaveBeenCalled();
    await u.click(go);
    await waitFor(() => expect(assign).toHaveBeenCalledWith("/login"));
    expect(calls.find((c) => c.method === "DELETE")!.body).toEqual({ password: "pw", confirm: "DELETE" });
  });
});

const summary = (horizon: number, ranking: "breakout" | "prebreakout", n: number, extra: Partial<Schemas["TrackRecordSummary"]> = {}): Schemas["TrackRecordSummary"] => ({
  ranking, ranking_label: ranking === "breakout" ? "BreakoutScore" : "PreBreakoutProb", horizon_days: horizon,
  avg_excess_return: 0.0123, median_excess_return: -0.004, win_rate: 0.56, sample_size: n, runs_used: 40, top_n: 10,
  benchmark: "SPY", computed_at: "2026-10-07T22:00:00+00:00", sufficient: n >= 25, ...extra,
});

describe("Track record", () => {
  const record = { disclaimer: "Historical research: backtested on saved scan snapshots.", min_sample_size: 25,
    summaries: [summary(5, "breakout", 300), summary(1, "breakout", 310), summary(20, "breakout", 12), summary(5, "prebreakout", 80, { avg_excess_return: 0.031 })] };

  it("is locked below Pro without asking the API", () => {
    wrap(<TrackRecordView />, me("basic"));
    expect(screen.getByText("Historical research is part of Pro")).toBeInTheDocument();
    expect(calls).toHaveLength(0);
  });

  it("shows rates only where the sample is large enough, with the disclaimer and the daily series", async () => {
    routes.unshift(
      on("GET", "/v1/track-record", () => jsonResponse(record)),
      on("GET", "/v1/track-record/daily", () => jsonResponse([{ day: "2026-10-01", avg_excess_return: 0.01 }, { day: "2026-10-02", avg_excess_return: -0.02 }, { day: "2026-10-03", avg_excess_return: null }])),
    );
    const u = user();
    wrap(<TrackRecordView />, me("pro"));
    expect(await screen.findByText(/backtested on saved scan snapshots/)).toBeInTheDocument();
    const rows = screen.getAllByRole("row").slice(1).map((r) => r.textContent);
    expect(rows[0]).toMatch(/^1 trading day/);                    // sorted by horizon
    expect(rows[1]).toContain("+1.23%");
    expect(rows[1]).toContain("56%");
    expect(rows[2]).toContain("Still building (12 of 25 needed)");
    expect(rows[2]).not.toContain("56%");
    expect(await screen.findByText(/1 of 2 days were above SPY/)).toBeInTheDocument();
    const q = new URLSearchParams(calls.find((c) => c.path === "/v1/track-record/daily")!.search);
    expect(Object.fromEntries(q)).toEqual({ ranking: "breakout", horizon: "5", days: "120" });
    await u.click(screen.getByRole("button", { name: "PreBreakout" }));
    expect(await screen.findByText("+3.10%")).toBeInTheDocument();
  });

  it("empty and failed answers", async () => {
    routes.unshift(on("GET", "/v1/track-record", () => jsonResponse({ ...record, summaries: [] })));
    const { unmount } = wrap(<TrackRecordView />, me("pro"));
    expect(await screen.findByText("No track record computed for this ranking yet.")).toBeInTheDocument();
    unmount();
    routes.unshift(on("GET", "/v1/track-record", () => jsonResponse({ detail: "The HSF service is unavailable right now." }, 503)));
    wrap(<TrackRecordView />, me("pro"));
    expect(await screen.findByText("Couldn't load the track record right now")).toBeInTheDocument();
  });

  it("formats fractions as signed percentages", () => {
    expect(signedPct(0.012)).toBe("+1.20%");
    expect(signedPct(-0.0005)).toBe("-0.05%");
    expect(signedPct(null)).toBe("—");
  });
});
