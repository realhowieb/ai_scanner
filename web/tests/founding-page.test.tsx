import { render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { FoundingStrip } from "@/features/FoundingBanner";
import { CLAIM_HREF, FAQ, FoundingPageView } from "@/features/FoundingPage";
import { usageRoute } from "@/lib/usageAnalytics";

import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/founding-100",
}));

const tiers = [
  { tier: "founder_pro", label: "Founder Pro", price_cents: 49900, capacity: 75, remaining: 73, on_sale: true,
    five_year: { monthly_cents: 2500, five_years_monthly_cents: 150000, price_cents: 49900, saved_cents: 100100 } },
  { tier: "founder_elite", label: "Founder Elite", price_cents: 99900, capacity: 25, remaining: 24, on_sale: true,
    five_year: { monthly_cents: 4000, five_years_monthly_cents: 240000, price_cents: 99900, saved_cents: 140100 } },
];
const status = { open: true, checkout_open: true, remaining: 97, tiers };

function serve(founding: unknown) {
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string) => {
    const url = String(typeof input === "string" ? input : input.url);
    if (url.includes("/v1/public/founding")) return jsonResponse(founding);
    return jsonResponse({ signed_in: false });
  }));
}
afterEach(() => vi.unstubAllGlobals());

describe("public Founding 100 page", () => {
  it("shows seats, the five-year math, both tiers and the FAQ before sign-in", () => {
    serve(status);
    render(<FoundingPageView initial={status} />);
    expect(screen.getByRole("heading", { level: 1, name: /100 founding memberships/ })).toBeInTheDocument();
    expect(screen.getByText(/97 of 100 seats left/)).toBeInTheDocument();
    expect(screen.getByText("73")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Founder Pro" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Founder Elite" })).toBeInTheDocument();
    expect(screen.getByText("You keep $1,001")).toBeInTheDocument();
    for (const [q] of FAQ) expect(screen.getByText(q)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Founding terms" })).toHaveAttribute("href", "/founding-100/terms");
  });

  it("sends visitors to sign-up, then straight on to checkout", () => {
    serve(status);
    render(<FoundingPageView initial={status} />);
    expect(CLAIM_HREF).toBe("/signup?next=%2Faccount%2Ffounder");
    const ctas = screen.getAllByRole("link", { name: "Create your account and claim a seat" });
    expect(ctas.length).toBeGreaterThan(1);
    for (const a of ctas) expect(a).toHaveAttribute("href", CLAIM_HREF);
  });

  it("says so when sold out, with no buy buttons", () => {
    const sold = { ...status, remaining: 0, tiers: tiers.map((t) => ({ ...t, remaining: 0, on_sale: false })) };
    serve(sold);
    render(<FoundingPageView initial={sold} />);
    expect(screen.getByText(/All 100 seats are taken/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /claim a seat/i })).toBeNull();
  });

  it("says so when the program is closed", () => {
    serve({ open: false, checkout_open: false, tiers: [], remaining: 0 });
    render(<FoundingPageView initial={null} />);
    expect(screen.getByText(/isn.t open right now/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /claim a seat/i })).toBeNull();
  });
});

describe("Founding strip and tracking", () => {
  it("links the weekly recap strip to the public page", async () => {
    serve(status);
    render(<FoundingStrip surface="weekly-test" />);
    const h = await screen.findByRole("heading", { name: "97 of 100 founding seats left" });
    const card = h.closest("section")!;
    expect(within(card).getByRole("link", { name: "See the Founding 100" })).toHaveAttribute("href", "/founding-100");
  });

  it("counts views of the public Founding and weekly pages", () => {
    expect(usageRoute("/founding-100")).toBe("/founding-100");
    expect(usageRoute("/founders")).toBe("/founders");
    expect(usageRoute("/weekly/2026-W44")).toBe("/weekly/:week");
    expect(usageRoute("/account/founder")).toBe("/account/founder");
  });
});
