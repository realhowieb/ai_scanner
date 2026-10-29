import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { FoundingBanner } from "@/features/FoundingBanner";

import { jsonResponse } from "./helpers";

const status = (over: Record<string, unknown> = {}) => ({
  open: true, checkout_open: true, remaining: 97,
  tiers: [
    { tier: "founder_pro", label: "Founder Pro", price_cents: 49900, capacity: 75, remaining: 73, on_sale: true },
    { tier: "founder_elite", label: "Founder Elite", price_cents: 99900, capacity: 25, remaining: 24, on_sale: true },
  ],
  ...over,
});

function serve(body: unknown) {
  const fetchMock = vi.fn(async (input: Request | string) => (void input, jsonResponse(body)));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
afterEach(() => vi.unstubAllGlobals());

describe("FoundingBanner", () => {
  it("shows seats left, prices and a link to the founder page when open", async () => {
    const f = serve(status());
    render(<FoundingBanner surface="test-open" />);
    expect(await screen.findByRole("heading", { name: "97 of 100 founding seats left" })).toBeTruthy();
    expect(screen.getByText(/Founder Pro \$499 \(73 left\) · Founder Elite \$999 \(24 left\)/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "See the Founding 100" }).getAttribute("href")).toBe("/account/founder");
    const req = f.mock.calls[0]![0];
    expect(String(typeof req === "string" ? req : req.url)).toContain("/api/public/v1/public/founding");
  });

  it("shows nothing while the program is closed", async () => {
    const f = serve({ open: false, checkout_open: false, tiers: [], remaining: 0 });
    const { container } = render(<FoundingBanner surface="test-closed" />);
    await waitFor(() => expect(f).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 20));
    expect(container.textContent).toBe("");
  });

  it("says sold out with no buy link", async () => {
    serve(status({ remaining: 0, tiers: status().tiers.map((t) => ({ ...t, remaining: 0, on_sale: false })) }));
    render(<FoundingBanner surface="test-sold" />);
    expect(await screen.findByRole("heading", { name: "The Founding 100 is sold out" })).toBeTruthy();
    expect(screen.queryByRole("link", { name: /Founding 100/ })).toBeNull();
  });
});
