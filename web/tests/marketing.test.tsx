import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MarketingTab, SocialCard, taggedLink } from "@/features/AdminMarketing";
import { EmailSignup, ListConfirmView } from "@/features/EmailSignup";
import { Landing } from "@/features/Landing";
import { WeeklyRecapView } from "@/features/WeeklyRecap";
import { weekRange } from "@/lib/weekRange";
import { cleanAttribution } from "@/server/public";

import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/",
}));

const calls: { path: string; method: string; body: unknown }[] = [];
let handler: (path: string, method: string, body: unknown) => unknown = () => ({});
beforeEach(() => {
  calls.length = 0;
  search = new URLSearchParams();
  localStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const path = new URL(r.url).pathname.replace(/^\/api\/(hsf|public)/, "");
    const body = r.method === "GET" ? null : JSON.parse((await r.text()) || "null");
    calls.push({ path, method: r.method, body });
    const out = handler(path, r.method, body);
    return out instanceof Response ? out : jsonResponse(out);
  }));
});
afterEach(() => vi.unstubAllGlobals());

const recap = {
  week: "2026-W44", start: "2026-10-26", end: "2026-10-30", built_at: "2026-11-01T00:00:00Z", scan_days: 5, ranked_names: 31,
  ranked: [{ ticker: "NVDA", name: "NVIDIA", days: 4, best_score: 88, setup: "breakout", first_day: "2026-10-26" },
    { ticker: "GME", name: null, days: 1, best_score: 71, setup: null, first_day: "2026-10-29" }],
  earnings: { start: "2026-11-02", end: "2026-11-06", days: [
    { date: "2026-11-02", total: 40, names: [{ ticker: "PLTR", time: "amc" }] },
    { date: "2026-11-03", total: 0, names: [] }] },
};

describe("weekly setups email list", () => {
  it("sends the email with the visitor's tags and shows the same answer for everyone", async () => {
    localStorage.setItem("hsf-attribution", JSON.stringify({ at: Date.now(), attribution: { utm_source: "x", utm_campaign: "w44" } }));
    handler = (path) => path === "/v1/public/list" ? { enabled: true }
      : { ok: true, message: "Check your inbox: we sent a link to confirm your email." };
    render(<EmailSignup surface="landing" />);
    await userEvent.type(screen.getByPlaceholderText("you@example.com"), "ann@example.com");
    await userEvent.click(screen.getByRole("button", { name: "Send me the weekly setups" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Check your inbox");
    const post = calls.find((c) => c.path === "/v1/public/subscribe")!;
    expect(post.body).toMatchObject({ email: "ann@example.com", surface: "landing", website: null,
      attribution: { utm_source: "x", utm_campaign: "w44" } });
  });

  it("hides itself when the list is switched off", async () => {
    handler = () => ({ enabled: false });
    const { container } = render(<EmailSignup surface="landing" />);
    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });

  it("shows on the landing page", async () => {
    handler = (path) => path === "/v1/public/list" ? { enabled: true } : new Response(JSON.stringify({ detail: "Not found." }), { status: 404 });
    render(<Landing />);
    expect(await screen.findByRole("heading", { name: /Get the weekly setups by email/ })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Weekly recap" })).toHaveAttribute("href", "/weekly");
  });

  it("confirms only when the button is pressed", async () => {
    search = new URLSearchParams({ t: "k".repeat(20) });
    handler = (path, method) => method === "GET" ? { email: "an***@example.com", status: "pending" } : { email: "an***@example.com", status: "confirmed" };
    render(<ListConfirmView />);
    expect(await screen.findByText("an***@example.com")).toBeInTheDocument();
    expect(calls.some((c) => c.method === "POST")).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Confirm my email" }));
    expect(await screen.findByRole("status")).toHaveTextContent("You're on the list");
    expect(calls.at(-1)).toMatchObject({ path: "/v1/public/subscribe/action", body: { token: "k".repeat(20), action: "confirm" } });
  });

  it("offers only unsubscribe from an unsubscribe link, and refuses a bad link", async () => {
    search = new URLSearchParams({ t: "k".repeat(20), action: "unsubscribe" });
    handler = () => ({ email: "an***@example.com", status: "confirmed" });
    const { unmount } = render(<ListConfirmView />);
    expect(await screen.findByRole("button", { name: "Unsubscribe" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Confirm my email" })).toBeNull();
    unmount();
    search = new URLSearchParams({ t: "short" });
    render(<ListConfirmView />);
    expect(screen.getByRole("alert")).toHaveTextContent("isn't valid");
  });

  it("the public proxy only forwards attribution keys", () => {
    expect(cleanAttribution({ utm_source: "x", evil: "1", ref: "abcd" })).toEqual({ utm_source: "x", ref: "abcd" });
  });
});

describe("public weekly recap", () => {
  it("lists rankings, never results, and teases next week's earnings", () => {
    handler = () => ({ enabled: true });
    render(<WeeklyRecapView recap={recap as never} weeks={[{ week: "2026-W44", start: "2026-10-26", end: "2026-10-30" },
      { week: "2026-W43", start: "2026-10-19", end: "2026-10-23" }]} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("The week of Oct 26 – Oct 30, 2026");
    const nvda = screen.getByRole("row", { name: /NVDA/ });
    expect(within(nvda).getByText("88")).toBeInTheDocument();
    expect(within(nvda).getByRole("link")).toHaveAttribute("href", "/share/NVDA");
    expect(screen.getByText("+39 more")).toBeInTheDocument();
    expect(screen.getByText("after close")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Oct 19 – Oct 23, 2026" })).toHaveAttribute("href", "/weekly/2026-W43");
    const text = document.body.textContent!.toLowerCase();
    for (const claim of ["return", "win rate", "beat", "gain", "outperform"]) expect(text).not.toContain(claim);
    expect(text).toContain("not a recommendation");
  });

  it("explains a missing week and links the latest", () => {
    handler = () => ({ enabled: true });
    render(<WeeklyRecapView recap={null} weeks={[{ week: "2026-W44", start: "2026-10-26", end: "2026-10-30" }]} />);
    expect(screen.getByText(/published each weekend/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Read the latest recap" })).toHaveAttribute("href", "/weekly/2026-W44");
    expect(weekRange("2026-12-28", "2027-01-01")).toBe("Dec 28 – Jan 1, 2027");
  });
});

describe("admin marketing", () => {
  const report = {
    days: 30, founder_revenue_cents: 49900, unavailable: [],
    by_source: [{ source: "X", visits: 40, clicks: 9, list_signups: 5, list_confirmed: 3, accounts: 4, paid: 1, founders: 1 }],
    totals: { visits: 40, clicks: 9, list_signups: 5, list_confirmed: 3, accounts: 4, paid: 1, founders: 1 },
    campaigns: [{ utm_source: "x", campaign: "weekly-w44", visits: 12, clicks: 3, list_signups: 2, accounts: 1 }],
    list: { confirmed: 3, pending: 2, unsubscribed: 0, new_in_period: 5, sync_configured: false },
  };

  const social = {
    configured: true, enabled: true, channel_error: null, recent: [],
    channels: [{ name: "hsfinest", service: "threads" }, { name: "hsfinest", service: "instagram" }],
    posts: [
      { kind: "earnings", week: "2026-W46", post_at: "2026-11-09T13:30:00Z", drafted: 2, failed: 0,
        preview: { kind: "earnings", week: "2026-W46", post_at: "2026-11-09T13:30:00Z", image_url: "https://app.hsfinest.com/social/earnings/2026-W46.png",
          texts: { threads: "Big earnings week.\n\nMon: $AAPL", linkedin: "x", instagram: "y" }, problem: null } },
      { kind: "founding", week: "2026-W45", post_at: "2026-11-06T17:00:00Z", drafted: 0, failed: 0, preview: null },
      { kind: "product", week: "2026-W45", post_at: "2026-11-04T17:00:00Z", drafted: 0, failed: 0,
        preview: { kind: "product", week: "2026-W45", post_at: "2026-11-04T17:00:00Z", image_url: "u", texts: { threads: "t", linkedin: "l", instagram: "i" }, problem: null } },
    ],
  };

  it("shows the social drafts and sends one now", async () => {
    handler = (path, method) => method === "POST" ? { sent: 2, already: 1, failed: 0, gave_up: 0 } : social;
    render(<SocialCard />);
    expect(await screen.findByText("2 channels")).toBeInTheDocument();
    const earnings = screen.getByRole("row", { name: /Earnings this week/ });
    expect(within(earnings).getByText("2 drafted")).toBeInTheDocument();
    expect(within(screen.getByRole("row", { name: /Founding 100/ })).getByRole("button", { name: "Send now" })).toBeDisabled();
    await userEvent.click(within(screen.getByRole("row", { name: /What HSF does/ })).getByRole("button", { name: "Send now" }));
    expect(await screen.findByText("Sent 2 drafts to Buffer, 1 already there.")).toBeInTheDocument();
    expect(calls.some((c) => c.method === "POST" && c.path === "/v1/admin/marketing/social/product")).toBe(true);
  });

  it("builds tagged links", () => {
    expect(taggedLink("https://app.hsfinest.com", "/weekly", "x", " Weekly Recap W45! "))
      .toBe("https://app.hsfinest.com/weekly?utm_source=x&utm_medium=social&utm_campaign=weekly-recap-w45");
    expect(taggedLink("https://app.hsfinest.com", "/", "newsletter", "")).toBe("https://app.hsfinest.com/?utm_source=newsletter&utm_medium=email");
  });

  it("shows sources, campaigns, the list and exports the CSV", async () => {
    const saved: string[] = [];
    URL.createObjectURL = vi.fn(() => "blob:x");
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) { saved.push(this.download); });
    handler = (path) => path.endsWith("subscribers.csv")
      ? new Response("email,status\na@x.com,confirmed\n", { status: 200, headers: { "content-type": "text/csv" } })
      : path.endsWith("/social") ? { ...social, configured: false, channels: [] } : report;
    render(<MarketingTab />);
    const row = await screen.findByRole("row", { name: /^X/ });
    expect(within(row).getByText("(3 confirmed)")).toBeInTheDocument();
    expect(screen.getByText("weekly-w44")).toBeInTheDocument();
    expect(screen.getByText("$499")).toBeInTheDocument();
    expect(screen.getByText("Not synced")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Download confirmed (CSV)" }));
    await waitFor(() => expect(saved).toEqual(["hsf-email-list-confirmed.csv"]));
    expect((screen.getByLabelText("Tagged link") as HTMLInputElement).value).toContain("/weekly?utm_source=x&utm_medium=social");
    expect(await screen.findByText("No Buffer key")).toBeInTheDocument();
  });
});
