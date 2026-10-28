import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FoundingRefundsTab } from "@/features/AdminFoundingRefunds";
import { FounderRefunds } from "@/features/FounderRefunds";

import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/account/founder",
}));

type Call = { method: string; path: string; search: string; body: unknown };
const fetchMock = vi.fn();
let calls: Call[] = [];
function serve(routes: Record<string, Response | ((c: Call) => Response)>) {
  fetchMock.mockImplementation(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const url = new URL(r.url);
    const text = r.method === "GET" ? "" : await r.text();
    const c = { method: r.method, path: url.pathname.replace(/^\/api\/(hsf|public)/, ""), search: url.search, body: text ? JSON.parse(text) : null };
    calls.push(c);
    const hit = routes[`${c.method} ${c.path}`];
    if (!hit) return jsonResponse({ detail: `no route ${c.method} ${c.path}` }, 500);
    return (typeof hit === "function" ? hit(c) : hit).clone();
  });
}

const payment = (over: Record<string, unknown> = {}) => ({
  kind: "purchase", tier: "founder_pro", amount_cents: 49900, currency: "usd", paid_at: "2026-11-02T15:00:00+00:00",
  refunded_cents: 0, deadline: "2026-11-16T15:00:00+00:00", within_guarantee: true, refundable_cents: 49900, ...over,
});
const mine = (over: Record<string, unknown> = {}) => ({
  payments: [payment()], requests: [], guarantee_days: 14, start_within_business_days: 7,
  support_email: "[support email to be added]", ...over,
});
const request = (over: Record<string, unknown> = {}) => ({
  id: 4, kind: "purchase", status: "requested", within_guarantee: true, member_reason: null, amount_cents: null,
  requested_at: "2026-11-05T15:00:00+00:00", due_at: "2026-11-14T15:00:00+00:00", decided_at: null,
  initiated_at: null, completed_at: null, decision_reason: null, ...over,
});
const adminRow = (over: Record<string, unknown> = {}) => ({
  ...request(), user_id: "member@example.com", basis: null, decided_by: null, provider_refund_id: null, failure_reason: null,
  tier: "founder_pro", payment_reference: "pi_1", paid_cents: 49900, refunded_cents: 0, paid_at: "2026-11-02T15:00:00+00:00",
  number: "017", membership_status: "active", activated_at: "2026-11-02T15:00:00+00:00", expires_at: "2031-11-02T15:00:00+00:00",
  upgraded_at: null, updated_at: null, ...over,
});

beforeEach(() => {
  calls = [];
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("member refunds", () => {
  it("shows the guarantee date and sends only the payment kind", async () => {
    let requested = false;
    serve({
      "GET /v1/founding/refunds/me": () => jsonResponse(mine(requested ? { requests: [request()] } : {})),
      "POST /v1/founding/refunds/request": () => { requested = true; return jsonResponse(request()); },
    });
    const u = userEvent.setup();
    render(<FounderRefunds />);
    expect(await screen.findByText(/Full refund guarantee until/)).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Request refund" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/full refund of \$499\.00/)).toBeInTheDocument();
    await u.type(within(dialog).getByRole("textbox"), "not for me");
    await u.click(within(dialog).getByRole("button", { name: "Send refund request" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ kind: "purchase", reason: "not for me" });
    expect(await screen.findByText("Waiting for review")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Request refund" })).not.toBeInTheDocument();
  });

  it("still lets a member ask after 14 days, for review", async () => {
    serve({ "GET /v1/founding/refunds/me": jsonResponse(mine({ payments: [payment({ within_guarantee: false })] })) });
    const u = userEvent.setup();
    render(<FounderRefunds />);
    expect(await screen.findByText(/14-day guarantee ended/)).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Request refund" }));
    expect(await screen.findByText(/a person reviews it against the refund policy/)).toBeInTheDocument();
  });

  it("shows a denial reason", async () => {
    serve({ "GET /v1/founding/refunds/me": jsonResponse(mine({ requests: [request({ status: "denied", decision_reason: "Outside the window." })] })) });
    render(<FounderRefunds />);
    expect(await screen.findByText("Not approved")).toBeInTheDocument();
    expect(screen.getByText("Reason: Outside the window.")).toBeInTheDocument();
  });
});

describe("admin refunds", () => {
  it("lists waiting requests and approves one with the guarantee", async () => {
    serve({
      "GET /v1/admin/founding/refunds": jsonResponse({ requests: [adminRow()], refunds_live: false, prorated_live: false }),
      "GET /v1/admin/founding/refunds/4": jsonResponse({ request: adminRow(), accepted_policies: [{ policy_type: "founding_terms", version: 1, content_hash: "a".repeat(64), accepted_at: "2026-11-02T14:59:00+00:00" }], audit: [{ at: "2026-11-05T15:00:00+00:00", actor: "member@example.com", action: "refund_requested", detail: null }], refunds_live: false, prorated_live: false }),
      "POST /v1/admin/founding/refunds/4/approve": jsonResponse({ request: adminRow({ status: "approved", basis: "guarantee", amount_cents: 49900 }), sent: false, note: "Refunds are switched off (FOUNDING_REFUNDS_ENABLED). Approved, not sent." }),
    });
    const u = userEvent.setup();
    render(<FoundingRefundsTab />);
    expect(await screen.findByText("member@example.com")).toBeInTheDocument();
    expect(calls[0]?.search).toBe("?status=requested");
    await u.click(screen.getByRole("button", { name: "Review" }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText(/nothing is sent to Stripe/)).toBeInTheDocument();
    expect(within(dialog).getByText("founding_terms v1")).toBeInTheDocument();
    expect(within(dialog).getByRole("option", { name: /Permanent discontinuation \(prorated\) \(off\)/ })).toBeDisabled();
    expect(within(dialog).getByRole("button", { name: "Deny" })).toBeDisabled();
    await u.click(within(dialog).getByRole("button", { name: "Approve" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
    expect(calls.find((c) => c.method === "POST")!.body).toEqual({ basis: "guarantee", reason: "" });
    expect(await within(dialog).findByText(/Approved, not sent/)).toBeInTheDocument();
  });

  it("denies with a reason", async () => {
    serve({
      "GET /v1/admin/founding/refunds": jsonResponse({ requests: [adminRow({ within_guarantee: false })], refunds_live: true, prorated_live: true }),
      "GET /v1/admin/founding/refunds/4": jsonResponse({ request: adminRow({ within_guarantee: false }), accepted_policies: [], audit: [], refunds_live: true, prorated_live: true }),
      "POST /v1/admin/founding/refunds/4/deny": jsonResponse({ request: adminRow({ status: "denied" }), sent: false }),
    });
    const u = userEvent.setup();
    render(<FoundingRefundsTab />);
    await u.click(await screen.findByRole("button", { name: "Review" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).queryByRole("option", { name: "14-day guarantee" })).not.toBeInTheDocument();
    await u.type(await within(dialog).findByRole("textbox", { name: /Note/ }), "No exception applies.");
    await u.click(within(dialog).getByRole("button", { name: "Deny" }));
    await waitFor(() => expect(calls.find((c) => c.method === "POST")?.body).toEqual({ reason: "No exception applies." }));
  });
});
