import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FounderBenefits } from "@/features/FounderBenefits";

import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/account/founder",
}));

const fetchMock = vi.fn();
const posts: { path: string; body: unknown }[] = [];
const benefits = (over: Record<string, unknown> = {}) => ({
  status: "active", tier: "founder_elite", badge: "HSF AI Elite Founder #004", show_badge: true,
  alumni_discount_percent: 50, alumni_rate_ready: false, early_access: [],
  invites: { days: 60, total: 3, left: 2 }, elite_limits: { alert_limit: 50, ai_multiplier: 2 },
  poll: { open: true, choice: null, options: [{ key: "api", label: "Personal API access" }, { key: "crypto", label: "Crypto scanner" }] },
  ...over,
});

beforeEach(() => {
  posts.length = 0;
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const path = new URL(r.url).pathname.replace(/^\/api\/hsf/, "");
    if (r.method === "GET") return jsonResponse(benefits());
    const body = JSON.parse(await r.text());
    posts.push({ path, body });
    if (path === "/v1/founding/poll") return jsonResponse(benefits({ poll: { ...benefits().poll, choice: body.option } }));
    return jsonResponse(benefits({ show_badge: body.show }));
  });
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

describe("founder benefits", () => {
  it("lists the benefits, toggles the badge and votes", async () => {
    const u = userEvent.setup();
    render(<FounderBenefits />);
    expect(await screen.findByText(/any HSF plan is 50% off/)).toBeInTheDocument();
    expect(screen.getByText(/Up to 50 active alerts/)).toBeInTheDocument();
    expect(screen.getByText(/2 of 3 left/)).toBeInTheDocument();
    await u.click(screen.getByRole("checkbox", { name: /Show my founder badge/ }));
    await waitFor(() => expect(posts[0]).toEqual({ path: "/v1/founding/benefits/badge", body: { show: false } }));
    expect(screen.getByRole("checkbox", { name: /Show my founder badge/ })).not.toBeChecked();
    await u.click(screen.getByRole("radio", { name: "Crypto scanner" }));
    await u.click(screen.getByRole("button", { name: "Vote" }));
    await waitFor(() => expect(posts[1]).toEqual({ path: "/v1/founding/poll", body: { option: "crypto" } }));
    expect(await screen.findByText("Your vote: Crypto scanner")).toBeInTheDocument();
  });
});
