import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FounderBriefingCard } from "@/features/AdminFoundingRefunds";
import { FounderBenefits } from "@/features/FounderBenefits";
import { FounderCardView, FoundersWallView } from "@/features/FoundersPublic";
import { FiveYearMath, fiveYearLine, FoundingNudge } from "@/features/FoundingBanner";
import { monthYear } from "@/lib/founders";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/account/founder",
}));

const tiers = [
  { tier: "founder_pro", label: "Founder Pro", price_cents: 49900, capacity: 75, remaining: 73, on_sale: true,
    five_year: { monthly_cents: 2500, five_years_monthly_cents: 150000, price_cents: 49900, saved_cents: 100100 } },
  { tier: "founder_elite", label: "Founder Elite", price_cents: 99900, capacity: 25, remaining: 24, on_sale: true,
    five_year: { monthly_cents: 4000, five_years_monthly_cents: 240000, price_cents: 99900, saved_cents: 140100 } },
];
const value = { months_used: 3, months_left: 57, term_months: 60, monthly_cents: 4000, paid_cents: 99900,
  monthly_cost_so_far_cents: 12000, ahead_cents: 0, break_even_month: 25, alerts_used: 7 };
const benefits = (over: Record<string, unknown> = {}) => ({
  status: "active", tier: "founder_elite", badge: "HSF AI Elite Founder #004", show_badge: true,
  alumni_discount_percent: 50, alumni_rate_ready: false, early_access: [],
  invites: null, elite_limits: { alert_limit: 50, ai_multiplier: 2 },
  poll: { open: false, choice: null, options: [] }, value, wall_name: null, card_path: "/founders/card/abcd1234",
  reply_target: "one business day", ...over,
});

const posts: { path: string; body: unknown }[] = [];
let handler: (path: string, method: string, body: unknown) => unknown = () => ({});
beforeEach(() => {
  posts.length = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const path = new URL(r.url).pathname.replace(/^\/api\/(hsf|public)/, "");
    const body = r.method === "GET" ? null : JSON.parse((await r.text()) || "null");
    if (body !== null) posts.push({ path, body });
    return jsonResponse(handler(path, r.method, body));
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe("5-year math", () => {
  it("compares five years paying monthly with the one-time price", () => {
    expect(fiveYearLine(tiers[0] as never)).toBe("5 years of Pro: $1,500 paying monthly, $499 once.");
    render(<FiveYearMath tiers={tiers as never} />);
    expect(screen.getByText("$2,400")).toBeInTheDocument();
    expect(screen.getByText("You keep $1,001")).toBeInTheDocument();
  });

  it("nudges free members with seats left, but never founders", async () => {
    handler = () => ({ open: true, checkout_open: true, remaining: 97, tiers });
    const { unmount } = render(<SessionProvider initialMe={me("basic")}><FoundingNudge surface="t1" /></SessionProvider>);
    expect(await screen.findByRole("link", { name: /Founding 100: 97 seats left\. Founder Pro is \$499 once/ })).toHaveAttribute("href", "/account/founder");
    unmount();
    const { container } = render(<SessionProvider initialMe={{ ...me("pro"), founder_tier: "founder_pro" }}><FoundingNudge surface="t2" /></SessionProvider>);
    await new Promise((r) => setTimeout(r, 30));
    expect(container.textContent).toBe("");
  });
});

describe("founder page extras", () => {
  it("shows the value meter, share card and priority support", async () => {
    handler = () => benefits();
    render(<FounderBenefits />);
    expect(await screen.findByRole("meter", { name: "Months of your membership used" })).toHaveAttribute("aria-valuenow", "3");
    expect(screen.getByText("3 of 60")).toBeInTheDocument();
    expect(screen.getByText("7 of 50")).toBeInTheDocument();
    expect(screen.getByText("Your seat pays for itself in month 25.")).toBeInTheDocument();
    expect(screen.getByText(/aim to reply within one business day/)).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Your founder card link" })).toHaveValue("http://localhost:3000/founders/card/abcd1234");
    expect(screen.getByRole("link", { name: /Post to X/ }).getAttribute("href")).toContain("twitter.com/intent/tweet");
  });

  it("joins the Founders Wall only after opting in with a name", async () => {
    const u = userEvent.setup();
    handler = (path, _m, body) => path === "/v1/founding/wall" ? benefits({ wall_name: (body as { name?: string }).name ?? null }) : benefits();
    render(<FounderBenefits />);
    const box = await screen.findByRole("checkbox", { name: /List me on the public Founders Wall/ });
    expect(box).not.toBeChecked();
    await u.click(box);
    await u.type(screen.getByRole("textbox", { name: "Display name" }), "Pro Trader");
    await u.click(screen.getByRole("button", { name: "Add me to the wall" }));
    await waitFor(() => expect(posts[0]).toEqual({ path: "/v1/founding/wall", body: { show: true, name: "Pro Trader" } }));
    expect(await screen.findByText('Listed as "Pro Trader".')).toBeInTheDocument();
    await u.click(screen.getByRole("checkbox", { name: /List me on the public Founders Wall/ }));
    await waitFor(() => expect(posts[1]).toEqual({ path: "/v1/founding/wall", body: { show: false } }));
  });
});

describe("public founder pages", () => {
  it("renders a founder card with no name or email, and a sign-up link with the invite code", () => {
    render(<FounderCardView card={{ badge: "HSF AI Founder #017", number: "017", tier: "founder_pro", label: "Founder Pro",
      status: "active", since: "2026-10-31T17:00:00+00:00", until: "2031-10-31T17:00:00+00:00", code: "abcd1234" }} />);
    expect(screen.getByText("#017")).toBeInTheDocument();
    expect(screen.getByText("October 2026")).toBeInTheDocument();
    expect(screen.getByText("October 2031")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Create a free account" })).toHaveAttribute("href", "/signup?ref=abcd1234");
  });

  it("lists every seat on the wall", () => {
    const seats = Array.from({ length: 100 }, (_, i) => ({ number: String(i + 1).padStart(3, "0"), taken: i < 2,
      tier: i === 1 ? "founder_elite" : i === 0 ? "founder_pro" : null, label: i === 1 ? "Founder Elite" : i === 0 ? "Founder Pro" : null,
      name: i === 1 ? "Bea" : null, since: i < 2 ? "2026-10" : null }));
    render(<FoundersWallView wall={{ open: true, taken: 2, seats }} />);
    expect(screen.getByText("2 of 100 seats taken.", { exact: false })).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(100);
    expect(screen.getByText("Founder")).toBeInTheDocument();
    expect(screen.getByText("Bea")).toBeInTheDocument();
    expect(screen.getAllByText("Open")).toHaveLength(98);
    expect(screen.getByRole("link", { name: "Claim a seat" })).toHaveAttribute("href", "/account/founder");
  });

  it("formats months", () => {
    expect(monthYear("2026-01-05")).toBe("January 2026");
    expect(monthYear("bad")).toBeNull();
  });
});

describe("Founder Briefing (admin)", () => {
  it("previews, then sends once after confirming", async () => {
    const u = userEvent.setup();
    let sent = false;
    handler = (path) => path.endsWith("/send") ? (sent = true, { month: "2026-11", sent: 3, failed: 0 })
      : { month: "2026-11", subject: "Your Founder Briefing: November 2026", html: "<p>hi</p>", text: "hi", recipients: 3,
        already_sent: sent, sent_at: null, history: [] };
    render(<FounderBriefingCard />);
    const send = screen.getByRole("button", { name: "Send to founders" });
    expect(send).toBeDisabled();
    await u.type(screen.getByRole("textbox", { name: /What shipped this month/ }), "Shipped the Founders Wall and value meter.");
    await u.click(screen.getByRole("button", { name: "Preview" }));
    expect(await screen.findByText("Your Founder Briefing: November 2026")).toBeInTheDocument();
    expect(screen.getByTitle("Founder Briefing preview")).toBeInTheDocument();
    await u.click(screen.getByRole("button", { name: "Send to founders" }));
    await u.click(await screen.findByRole("button", { name: "Send now" }));
    expect(await screen.findByText("Sent to 3 founders.")).toBeInTheDocument();
    expect(posts.find((p) => p.path.endsWith("/send"))?.body).toEqual({ note: "Shipped the Founders Wall and value meter.", confirm: true });
    expect(screen.getByRole("button", { name: "Send to founders" })).toBeDisabled();
  });
});
