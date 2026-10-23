import { NextRequest } from "next/server";

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppShell } from "@/components/AppShell";
import { GettingStarted } from "@/features/GettingStarted";
import { SharedStock } from "@/features/SharedStock";
import { shareUrl } from "@/features/ShareStock";
import { sanitizeUsage, usageRoute } from "@/lib/usageAnalytics";
import { proxy } from "@/proxy";
import { REFRESH_COOKIE } from "@/server/cookies";
import { SessionProvider } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

let path = "/today";
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => path,
}));

const calls: { method: string; path: string; body: unknown }[] = [];
let routes: Record<string, (body: unknown) => unknown> = {};
beforeEach(() => {
  path = "/today";
  calls.length = 0;
  routes = {};
  vi.stubGlobal("fetch", async (req: Request) => {
    const url = new URL(req.url);
    const p = url.pathname.replace(/^\/api\/hsf/, "");
    const text = req.method === "GET" ? "" : await req.text();
    const body = text ? JSON.parse(text) : null;
    calls.push({ method: req.method, path: p, body });
    const h = routes[`${req.method} ${p}`];
    return h ? jsonResponse(h(body)) : jsonResponse({ detail: "Not Found" }, 404);
  });
});
afterEach(() => vi.unstubAllGlobals());

const step = (id: string, done: boolean, href = `/${id}`) => ({ id, title: `Step ${id}`, detail: `Do ${id}`, href, done });
const checklist = (done: Record<string, boolean>, extra: Record<string, unknown> = {}) => {
  const steps = [step("watchlist", done.watchlist ?? false, "/watchlists"), step("alert", done.alert ?? false, "/alerts?new=1"),
    step("delivery", done.delivery ?? false, "/alerts#delivery"), step("guide", done.guide ?? false, "/how-hsf-works")];
  const n = steps.filter((s) => s.done).length;
  return { show: n < 4, dismissed: false, done: n, total: 4, steps, ...extra };
};

describe("Getting started", () => {
  it("lists the steps with progress and links to where each is done", async () => {
    routes["GET /v1/me/getting-started"] = () => checklist({ watchlist: true });
    render(<GettingStarted />);
    expect(await screen.findByRole("heading", { name: "Getting started" })).toBeInTheDocument();
    expect(screen.getByText("1 of 4 done")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "1");
    expect(screen.getByText("Step watchlist")).toHaveTextContent("(done)");
    const links = screen.getAllByRole("link").map((a) => a.getAttribute("href"));
    expect(links).toEqual(["/alerts?new=1", "/alerts#delivery", "/how-hsf-works"]); // no button for a done step
  });

  it("opening the guide ticks its step; Hide this closes the card", async () => {
    const u = userEvent.setup();
    routes["GET /v1/me/getting-started"] = () => checklist({});
    routes["POST /v1/me/getting-started/guide"] = () => checklist({ guide: true });
    routes["POST /v1/me/getting-started/dismiss"] = () => checklist({ guide: true }, { show: false, dismissed: true });
    render(<GettingStarted />);
    const read = await screen.findByRole("link", { name: "Read" });
    read.addEventListener("click", (e) => e.preventDefault());
    await u.click(read);
    await screen.findByText("1 of 4 done");
    await u.click(screen.getByRole("button", { name: "Hide this" }));
    await waitFor(() => expect(screen.queryByRole("heading", { name: "Getting started" })).not.toBeInTheDocument());
    expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/v1/me/getting-started/guide", "/v1/me/getting-started/dismiss"]);
  });

  it("shows nothing when the API says it's done or can't answer", async () => {
    const { container } = render(<GettingStarted />);
    await waitFor(() => expect(calls.some((c) => c.path === "/v1/me/getting-started")).toBe(true));
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Alerts badge and feedback", () => {
  it("shows the unread count on Alerts and clears it on the Alerts page", async () => {
    routes["GET /v1/me/alerts/unread"] = () => ({ count: 3, seen_at: null });
    routes["POST /v1/me/alerts/seen"] = () => ({ count: 0, seen_at: "2026-10-23T20:00:00+00:00" });
    const view = render(<SessionProvider initialMe={me("pro")}><AppShell><p>body</p></AppShell></SessionProvider>);
    const badges = await screen.findAllByText("3");
    expect(badges.length).toBe(2); // desktop nav and phone tab bar
    expect(screen.getAllByRole("link", { name: /Alerts, 3 new/ }).length).toBe(2);
    path = "/alerts";
    view.rerender(<SessionProvider initialMe={me("pro")}><AppShell><p>body</p></AppShell></SessionProvider>);
    await waitFor(() => expect(screen.queryByText("3")).not.toBeInTheDocument());
    expect(calls.some((c) => c.method === "POST" && c.path === "/v1/me/alerts/seen")).toBe(true);
  });

  it("sends feedback with the page it came from", async () => {
    const u = userEvent.setup();
    path = "/scanner";
    routes["POST /v1/me/feedback"] = () => ({ ok: true, message: "Thanks" });
    render(<SessionProvider initialMe={me("pro")}><AppShell><p>body</p></AppShell></SessionProvider>);
    await u.click(screen.getByRole("button", { name: "Send feedback" })); // footer
    await u.click(screen.getByLabelText("An idea"));
    await u.type(screen.getByLabelText("Your message"), "Dark charts please");
    await u.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText(/your message was sent/)).toBeInTheDocument();
    const sent = calls.find((c) => c.path === "/v1/me/feedback")!.body as Record<string, unknown>;
    expect(sent).toMatchObject({ kind: "idea", message: "Dark charts please", path: "/scanner" });
  });
});

describe("Shared stock links", () => {
  const ranked = { ticker: "MXL", name: "MaxLinear, Inc.", price: 12.5, change_pct: 3.2, scan_at: "2026-10-23T14:35:00+00:00",
    ranked: true, hsf_score: 77, setup: "breakout", reason: "Broke above the 20-day high", hidden_insights: 4 };

  it("a ranked stock shows its score and one reason, the rest behind a free account", () => {
    render(<SharedStock stock={ranked} asked="MXL" />);
    expect(screen.getByRole("heading", { name: "MXL" })).toBeInTheDocument();
    expect(screen.getByLabelText("HSF Score 77")).toBeInTheDocument();
    expect(screen.getByText(/Broke above the 20-day high/)).toBeInTheDocument();
    expect(screen.getByText("4 more reasons, signals and risks")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See the full MXL analysis, free" })).toHaveAttribute("href", "/signup?next=%2Fstocks%2FMXL");
  });

  it("an unranked stock offers the alert instead of a score", () => {
    render(<SharedStock stock={{ ...ranked, ticker: "NVDA", ranked: false, hsf_score: null, reason: null, hidden_insights: 0 }} asked="NVDA" />);
    expect(screen.queryByLabelText(/HSF Score/)).not.toBeInTheDocument();
    expect(screen.getByText(/Get an alert the moment NVDA becomes a ranked setup/)).toBeInTheDocument();
  });

  it("an unknown stock says so", () => {
    render(<SharedStock stock={null} asked="ZZZZQ" />);
    expect(screen.getByText(/couldn.t find that stock/)).toBeInTheDocument();
  });

  it("share links carry a valid invite code only", () => {
    expect(shareUrl("https://app.hsfinest.com", "BRK.B", "abc23def")).toBe("https://app.hsfinest.com/share/BRK.B?ref=abc23def");
    expect(shareUrl("https://app.hsfinest.com", "MXL", "bad code!")).toBe("https://app.hsfinest.com/share/MXL");
  });

  it("members opening a share link go to the full stock page; visitors and preview images don't", () => {
    const req = (p: string, signedIn: boolean) => {
      const r = new NextRequest(new URL(p, "https://app.hsfinest.com"));
      if (signedIn) r.cookies.set(REFRESH_COOKIE, "rt");
      return r;
    };
    expect(proxy(req("/share/MXL", true)).headers.get("location")).toBe("https://app.hsfinest.com/stocks/MXL");
    expect(proxy(req("/share/MXL", false)).headers.get("location")).toBeNull();
    expect(proxy(req("/share/MXL/opengraph-image", true)).headers.get("location")).toBeNull();
  });

  it("analytics keeps share and onboarding details but no tickers", () => {
    expect(usageRoute("/share/MXL")).toBe("/share/:ticker");
    const out = sanitizeUsage({ uuid: "u", event: "getting_started_step_opened",
      properties: { step: "alert", ticker: "MXL", done: 2, distinct_id: "d" } } as never);
    expect(out?.properties).toMatchObject({ step: "alert", done: 2 });
    expect(out?.properties).not.toHaveProperty("ticker");
    expect(sanitizeUsage({ uuid: "u", event: "made_up", properties: {} } as never)).toBeNull();
  });
});
