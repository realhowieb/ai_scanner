import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FounderView } from "@/features/FounderView";
import { PolicyPage } from "@/features/PolicyPage";

import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/account/founder",
}));

type Call = { method: string; path: string; body: unknown };
const fetchMock = vi.fn();
let calls: Call[] = [];
function serve(routes: Record<string, Response | ((c: Call) => Response)>) {
  fetchMock.mockImplementation(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const url = new URL(r.url);
    const text = r.method === "GET" ? "" : await r.text();
    const c = { method: r.method, path: url.pathname.replace(/^\/api\/(hsf|public)/, ""), body: text ? JSON.parse(text) : null };
    calls.push(c);
    const hit = routes[`${c.method} ${c.path}`];
    if (!hit) return jsonResponse({ detail: `no route ${c.method} ${c.path}` }, 500);
    return (typeof hit === "function" ? hit(c) : hit).clone();
  });
}

const disclosure = (name: string, price: string) => ({
  name, payment: `One-time payment: ${price}`, duration: "Membership duration: Five years",
  recurring: "Recurring payments: None", refund: "Refund guarantee: 14 days",
});
const POLICIES = [
  { policy_type: "founding_terms", version: 1, content_hash: "a".repeat(64), title: "Terms", path: "/founding-100/terms", status: "proposed" },
  { policy_type: "founding_refunds", version: 1, content_hash: "b".repeat(64), title: "Refunds", path: "/founding-100/refunds", status: "proposed" },
  { policy_type: "privacy", version: 1, content_hash: "c".repeat(64), title: "Privacy", path: "/privacy", status: "proposed" },
];
const ACK = "I understand that I am purchasing a five-year HSF AI Founding Membership for a one-time payment. My membership does not automatically renew. I have reviewed the Founding Membership Terms, Refund Policy, and Privacy Policy.";
function offer(over: Record<string, unknown> = {}) {
  return {
    open: true, checkout_open: true, preview: false, missing_details: [], acknowledgment: ACK, policies: POLICIES, membership: null,
    tiers: [
      { tier: "founder_pro", label: "Founder Pro", plan: "pro", price_cents: 49900, capacity: 75, remaining: 75, on_sale: true, disclosure: disclosure("HSF AI Founder Pro", "$499") },
      { tier: "founder_elite", label: "Founder Elite", plan: "premium", price_cents: 99900, capacity: 25, remaining: 0, on_sale: false, disclosure: disclosure("HSF AI Founder Elite", "$999") },
    ],
    upgrade: { price_cents: 50000, eligible: false, on_sale: false, disclosure: { ...disclosure("HSF AI Founder Elite upgrade", "$500"), duration: "Membership duration: Keeps your current end date" } },
    ...over,
  };
}

const assign = vi.fn();
beforeEach(() => {
  search = new URLSearchParams();
  calls = [];
  fetchMock.mockReset();
  assign.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  Object.defineProperty(window, "location", { configurable: true, value: { ...window.location, assign } });
});
afterEach(() => vi.unstubAllGlobals());

describe("founder page", () => {
  it("shows each tier's disclosures and needs the unticked acknowledgment before checkout", async () => {
    serve({
      "GET /v1/founding/offer": jsonResponse(offer()),
      "POST /v1/founding/checkout": jsonResponse({ checkout_url: "https://checkout.stripe.com/c/1", expires_at: "2026-11-02T15:35:00+00:00" }),
    });
    const u = userEvent.setup();
    render(<FounderView />);
    expect(await screen.findByText("HSF AI Founder Pro")).toBeInTheDocument();
    for (const text of ["$499", "$999", "Five years", "None", "14 days"]) expect(screen.getAllByText(text).length).toBeGreaterThan(0);
    expect(screen.getByText("75 of 75 left")).toBeInTheDocument();
    const box = screen.getByRole("checkbox", { name: ACK });
    expect(box).not.toBeChecked();
    for (const name of ["Founding Membership Terms", "Refund Policy", "Privacy Policy"]) {
      expect(screen.getByRole("link", { name: new RegExp(name) })).toHaveAttribute("href", expect.stringMatching(/^\/(founding-100|privacy)/));
    }
    const pay = screen.getByRole("button", { name: "Continue to payment: $499" });
    expect(pay).toBeDisabled();
    expect(screen.getByRole("button", { name: "Sold out" })).toBeDisabled();
    await u.click(box);
    expect(pay).toBeEnabled();
    await u.click(pay);
    await waitFor(() => expect(assign).toHaveBeenCalledWith("https://checkout.stripe.com/c/1"));
    const sent = calls.find((c) => c.path === "/v1/founding/checkout")!.body as Record<string, unknown>;
    expect(sent).toEqual({ tier: "founder_pro", upgrade: false, acknowledged: true,
      policies: POLICIES.map(({ policy_type, version, content_hash }) => ({ policy_type, version, content_hash })) });
  });

  it("offers to reload when the policies changed", async () => {
    serve({
      "GET /v1/founding/offer": jsonResponse(offer()),
      "POST /v1/founding/checkout": jsonResponse({ detail: "Our policies changed since this page loaded.", code: "policy_changed", changed: ["privacy"] }, 409),
    });
    const u = userEvent.setup();
    render(<FounderView />);
    await u.click(await screen.findByRole("checkbox"));
    await u.click(screen.getByRole("button", { name: "Continue to payment: $499" }));
    expect(await screen.findByText(/Our policies changed/)).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Load the current policies" }));
    expect(screen.getByRole("checkbox")).not.toBeChecked();
    expect(assign).not.toHaveBeenCalled();
  });

  it("shows a member's badge and the upgrade", async () => {
    serve({ "GET /v1/founding/offer": jsonResponse(offer({
      membership: { tier: "founder_pro", label: "Founder Pro", status: "active", number: "017", badge: "HSF AI Founder #017", activated_at: "2026-11-02T15:00:00+00:00", expires_at: "2031-11-02T15:00:00+00:00" },
      upgrade: { ...offer().upgrade, eligible: true, on_sale: true },
    })) });
    render(<FounderView />);
    expect(await screen.findByText("HSF AI Founder #017")).toBeInTheDocument();
    expect(screen.getByText("Access until")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /\$499/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue to payment: $500" })).toBeDisabled();
  });

  it("says the program isn't open for members while it's off", async () => {
    serve({ "GET /v1/founding/offer": jsonResponse({ detail: "The Founding 100 isn't open yet.", code: "founding_closed" }, 404) });
    render(<FounderView />);
    expect(await screen.findByText("The Founding 100 isn't open yet.")).toBeInTheDocument();
  });

  it("marks the admin preview", async () => {
    serve({ "GET /v1/founding/offer": jsonResponse(offer({ preview: true, checkout_open: false, missing_details: ["HSF_SUPPORT_EMAIL"],
      tiers: offer().tiers.map((t) => ({ ...t, on_sale: false })) })) });
    render(<FounderView />);
    expect(await screen.findByText(/Admin preview.*HSF_SUPPORT_EMAIL/)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Not on sale yet" })[0]).toBeDisabled();
  });
});

describe("policy pages", () => {
  const policy = {
    policy_type: "founding_refunds" as const, version: 1, title: "HSF AI Founding Membership Refund Policy", effective_date: "Not yet in effect",
    status: "proposed" as const, content_hash: "d".repeat(64), path: "/founding-100/refunds",
    sections: [{ heading: null, blocks: [{ p: "Intro." }] }, { heading: "14-day refund guarantee", blocks: [{ list: ["No reason needed."] }, { p: "After list." }] }],
  };

  it("renders sections with the review banner, version and fingerprint", () => {
    render(<PolicyPage loaded={{ policy, preview: false }} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Refund Policy");
    expect(screen.getByText("Proposed terms, pending legal review. Not yet in effect.")).toBeInTheDocument();
    expect(screen.getByText("Version 1 · Not yet in effect")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "14-day refund guarantee" })).toBeInTheDocument();
    expect(screen.getByRole("listitem")).toHaveTextContent("No reason needed.");
    expect(screen.getByText("d".repeat(64))).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Founding Membership Terms" })).toHaveAttribute("href", "/founding-100/terms");
    expect(screen.queryByRole("link", { name: "Refund Policy" })).not.toBeInTheDocument();
  });

  it("marks an admin preview and handles an unpublished page", () => {
    const { unmount } = render(<PolicyPage loaded={{ policy: { ...policy, status: "approved" }, preview: true }} />);
    expect(screen.getByText(/Admin preview/)).toBeInTheDocument();
    expect(screen.queryByText(/pending legal review/)).not.toBeInTheDocument();
    unmount();
    render(<PolicyPage loaded={null} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Not published yet");
  });
});
