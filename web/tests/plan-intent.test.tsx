import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CompanyFacts } from "@/features/StockView";
import { AutoCheckout, needsPlan } from "@/features/PlanIntent";
import { readPlanIntent, savePlanIntent, upgradeHref } from "@/lib/planIntent";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
vi.mock("next/navigation", () => ({ useSearchParams: () => search, useRouter: () => ({ replace: vi.fn(), push: vi.fn() }), usePathname: () => "/account" }));

const fetchMock = vi.fn();
const assign = vi.fn();
beforeEach(() => {
  search = new URLSearchParams();
  fetchMock.mockReset();
  assign.mockReset();
  localStorage.clear();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("location", { ...window.location, assign });
});
afterEach(() => vi.unstubAllGlobals());

describe("plan picked on Pricing", () => {
  it("is kept for a week and points at the Account checkout link", () => {
    savePlanIntent("pro", "year", 1_000);
    expect(readPlanIntent(2_000)).toEqual({ plan: "pro", interval: "year", at: 1_000 });
    expect(readPlanIntent(1_000 + 8 * 24 * 3600 * 1000)).toBeNull();
    expect(upgradeHref("premium", "month")).toBe("/account?upgrade=premium&interval=month#plan");
  });

  it("a free Pro trial still counts as needing Pro", () => {
    expect(needsPlan({ plan: "pro", pro_trial_until: "2026-11-17T00:00:00Z" }, "pro")).toBe(true);
    expect(needsPlan({ plan: "pro", pro_trial_until: null }, "pro")).toBe(false);
    expect(needsPlan({ plan: "pro", pro_trial_until: null }, "premium")).toBe(true);
  });

  it("Account opens checkout for ?upgrade= once the email is confirmed", async () => {
    search = new URLSearchParams("upgrade=premium&interval=year");
    savePlanIntent("premium", "year");
    fetchMock.mockImplementation(async (req: Request) => new URL(req.url).pathname.endsWith("/v1/billing/checkout")
      ? jsonResponse({ url: "https://checkout.stripe.com/c/test", mode: "checkout" })
      : jsonResponse({ detail: "no fake" }, 404));
    const m = { ...me("basic"), email_verified: true };
    render(<SessionProvider initialMe={m}><AutoCheckout me={m} /></SessionProvider>);
    await waitFor(() => expect(assign).toHaveBeenCalledWith("https://checkout.stripe.com/c/test"));
    const call = fetchMock.mock.calls.find((c) => new URL((c[0] as Request).url).pathname.endsWith("/v1/billing/checkout"))!;
    expect(await (call[0] as Request).clone().json()).toEqual({ plan: "premium", interval: "year" });
    expect(readPlanIntent()).toBeNull();
  });

  it("waits for the email confirmation before opening checkout", () => {
    search = new URLSearchParams("upgrade=pro");
    const m = { ...me("basic"), email_verified: false };
    render(<SessionProvider initialMe={m}><AutoCheckout me={m} /></SessionProvider>);
    expect(screen.getByText(/Confirm your email first/)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("company facts", () => {
  it("shows market cap, size, sector and industry when known", () => {
    render(<CompanyFacts facts={{ sector: "Technology", industry: "Semiconductors", market_cap: 3.2e12, size: "Mega" }} />);
    expect(screen.getByLabelText("Company facts")).toHaveTextContent("Market cap$3.2T · Mega capSectorTechnologyIndustrySemiconductors");
  });
  it("shows nothing when the screener doesn't list the stock", () => {
    const { container } = render(<CompanyFacts facts={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
