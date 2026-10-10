import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ImgHTMLAttributes } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AIScanPanel } from "@/features/AIScanPanel";
import { DemoPage } from "@/features/Demo";
import { Landing, PricingPage } from "@/features/Landing";
import { PaperView } from "@/features/PaperView";
import { StairSteppers } from "@/features/StairSteppers";
import { StockSearchView } from "@/features/StockSearchView";
import { UnsubscribeView } from "@/features/UnsubscribeView";
import { SessionProvider } from "@/session/SessionProvider";

import { me, setup } from "./fixtures";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
const push = vi.fn();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace: vi.fn(), push, back: vi.fn() }),
  usePathname: () => "/",
}));
vi.mock("next/image", () => ({
  default: (props: ImgHTMLAttributes<HTMLImageElement> & { unoptimized?: boolean; priority?: boolean }) => {
    const { src, alt, width, height, className } = props;
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={src} alt={alt} width={width} height={height} className={className} />;
  },
}));

type Call = { method: string; path: string; query: URLSearchParams; body: unknown };
const fetchMock = vi.fn();
let calls: Call[] = [];
/** Route table: "METHOD /path" -> response (or a function of the call). */
function serve(routes: Record<string, Response | ((c: Call) => Response)>) {
  fetchMock.mockImplementation(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const url = new URL(r.url);
    const text = r.method === "GET" ? "" : await r.text();
    const c = { method: r.method, path: url.pathname.replace(/^\/api\/(hsf|public)/, ""), query: url.searchParams, body: text ? JSON.parse(text) : null };
    calls.push(c);
    const hit = routes[`${c.method} ${c.path}`];
    if (!hit) return jsonResponse({ detail: `no route ${c.method} ${c.path}` }, 500);
    const res = typeof hit === "function" ? hit(c) : hit;
    return res.clone();
  });
}

const PLANS = {
  tiers: [
    { id: "basic", name: "Free", price: "Free", yearly_price: null, tagline: "Discover.", alert_limit: 1, highlights: ["HSF Score"] },
    { id: "pro", name: "Pro", price: "$25/mo", yearly_price: "$250/yr", tagline: "Monitor.", alert_limit: 5, highlights: ["Day Trader"] },
    { id: "premium", name: "Premium", price: "$40/mo", yearly_price: "$400/yr", tagline: "Research.", alert_limit: 25, highlights: ["AI scan summaries"] },
  ],
  rows: [{ label: "Alerts", basic: 1, pro: 5, premium: 25 }, { label: "AI scan summaries", basic: false, pro: false, premium: true }],
};

beforeEach(() => {
  search = new URLSearchParams();
  calls = [];
  push.mockReset();
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  localStorage.clear();
  sessionStorage.clear();
});
afterEach(() => vi.unstubAllGlobals());

const events = () => calls.filter((c) => c.path === "/v1/events").map((c) => c.body as { event: string; surface: string; attribution: Record<string, string> });

describe("landing and pricing", () => {
  it("shows plans from the API, records one visit with the utm tags and credits the call to action", async () => {
    window.history.replaceState(null, "", "/?utm_source=reddit&utm_campaign=launch");
    serve({ "GET /v1/plans": jsonResponse(PLANS), "POST /v1/events": new Response(null, { status: 202 }) });
    const u = userEvent.setup();
    const { unmount } = render(<Landing />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Turn the whole market into a short list.");
    expect(await screen.findByText("$25/mo")).toBeInTheDocument();
    expect(screen.getByText("or $250/yr (two months free)")).toBeInTheDocument();
    await u.click(screen.getByRole("tab", { name: "Day Trader" }));
    expect(screen.getByRole("img")).toHaveAttribute("src", "/landing/day-trader-stair-stepper.webp");
    await u.click(screen.getByRole("link", { name: "Create a free account" }));
    await waitFor(() => expect(events().map((e) => e.event)).toEqual(["landing_visit", "primary_cta_click"]));
    expect(events()[0]!.attribution).toEqual({ utm_source: "reddit", utm_campaign: "launch" });
    expect(events()[1]!.surface).toBe("hero");
    unmount();
    window.history.replaceState(null, "", "/pricing");
    render(<PricingPage />);   // same browser session: no second visit; tags kept from the first page
    const table = await screen.findByRole("table");
    const ai = within(table).getByRole("row", { name: /AI scan summaries/ });
    expect(within(ai).getAllByLabelText("Not included")).toHaveLength(2);
    expect(within(ai).getByLabelText("Included")).toBeInTheDocument();
    expect(events().filter((e) => e.event === "landing_visit")).toHaveLength(1);
    window.history.replaceState(null, "", "/");
  });

  it("says when plans can't load and retries", async () => {
    let fail = true;
    serve({ "GET /v1/plans": () => (fail ? jsonResponse({ detail: "The HSF service is starting up. Try again in a moment." }, 503) : jsonResponse(PLANS)),
      "POST /v1/events": new Response(null, { status: 202 }) });
    const u = userEvent.setup();
    render(<PricingPage />);
    expect(await screen.findByText(/starting up/)).toBeInTheDocument();
    fail = false;
    await u.click(screen.getByRole("button", { name: /Try again/ }));
    expect((await screen.findAllByText("$40/mo")).length).toBeGreaterThan(0);
  });

  it("offers a frontend-only demo with sample product state", () => {
    render(<DemoPage />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("See HSF with polished sample data.");
    expect(screen.getByRole("status")).toHaveTextContent("fixed sample data");
    expect(screen.getByRole("table", { name: "Demo ranked setups" })).toHaveTextContent("NVDA");
    expect(screen.getByRole("heading", { name: "Stock detail" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Watchlist intelligence" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Alerts and AI research" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Create a free account" })).toHaveAttribute("href", "/signup");
  });
});

describe("Stock Intelligence", () => {
  it("opens a ticker, refuses non-tickers and lists the top of the latest scan", async () => {
    serve({ "GET /v1/scans/latest": jsonResponse({ scan_at: new Date().toISOString(), total: 2, max_results: 25, limited: false, setups: [setup("AAA", 91), setup("BBB", 80, { chg_pct: -1 })] }) });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("basic")}><StockSearchView /></SessionProvider>);
    expect(await screen.findByRole("link", { name: "AAA" })).toHaveAttribute("href", "/stocks/AAA");
    expect(screen.getByRole("heading", { name: "Top setups to review" })).toBeInTheDocument();
    expect(screen.getByText(/Each review opens the score/)).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "Review" })[0]).toHaveAttribute("href", "/stocks/AAA");
    expect(screen.getAllByText("STRONG · 2 confirming signals").length).toBeGreaterThan(0);
    expect(calls[0]!.query.get("limit")).toBe("12");
    await u.type(screen.getByLabelText("Ticker"), "not a ticker");
    await u.click(screen.getByRole("button", { name: "Open" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Enter a ticker symbol");
    expect(push).not.toHaveBeenCalled();
    await u.clear(screen.getByLabelText("Ticker"));
    await u.type(screen.getByLabelText("Ticker"), "nvda");
    await u.click(screen.getByRole("button", { name: "Open" }));
    expect(push).toHaveBeenCalledWith("/stocks/NVDA");
  });
});

describe("AI research on a scan", () => {
  it("is locked below Premium", () => {
    render(<SessionProvider initialMe={me("pro")}><AIScanPanel /></SessionProvider>);
    expect(screen.getByText("AI scan summaries and chat are part of Premium")).toBeInTheDocument();
  });

  it("summarizes, answers questions with the conversation so far, and keeps the question when a call fails", async () => {
    let chatFails = false;
    serve({
      "POST /v1/ai/summary": jsonResponse({ run_id: 7, text: "## Leaders\n- **AAA** leads on volume\n- BBB follows" }),
      "POST /v1/ai/chat": (c) => (chatFails ? jsonResponse({ detail: "You've reached today's AI limit." }, 429)
        : jsonResponse({ run_id: 7, answer: `You asked ${(c.body as { messages: unknown[] }).messages.length} thing(s).` })),
    });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("premium")}><AIScanPanel runId={7} /></SessionProvider>);
    await u.click(screen.getByRole("button", { name: "Summarize this scan" }));
    expect(await screen.findByText("AAA")).toHaveProperty("tagName", "STRONG");
    expect(screen.getByText("Leaders")).toBeInTheDocument();
    expect(calls[0]!.body).toEqual({ run_id: 7 });
    await u.click(screen.getByRole("button", { name: "What do the top three have in common?" }));
    expect(await screen.findByText("You asked 1 thing(s).")).toBeInTheDocument();
    await u.type(screen.getByLabelText("Your question about this scan"), "And earnings?");
    await u.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByText("You asked 3 thing(s).")).toBeInTheDocument();
    const last = calls.at(-1)!.body as { run_id: number; messages: { role: string }[] };
    expect(last.run_id).toBe(7);
    expect(last.messages.map((m) => m.role)).toEqual(["user", "assistant", "user"]);
    chatFails = true;
    await u.type(screen.getByLabelText("Your question about this scan"), "One more");
    await u.click(screen.getByRole("button", { name: "Ask" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("today's AI limit");
    expect(screen.getByLabelText("Your question about this scan")).toHaveValue("One more");
    expect(screen.queryByText("One more", { selector: "li p" })).not.toBeInTheDocument();
  });
});

describe("Paper trading", () => {
  it("is locked below Premium and says when the server can't trade", async () => {
    const { unmount } = render(<SessionProvider initialMe={me("pro")}><PaperView /></SessionProvider>);
    expect(screen.getByText("Paper trading is part of Premium")).toBeInTheDocument();
    unmount();
    serve({ "GET /v1/paper/account": jsonResponse({ detail: "Paper trading isn't configured." }, 503) });
    render(<SessionProvider initialMe={me("premium")}><PaperView /></SessionProvider>);
    expect(await screen.findByText("Paper trading isn't available right now.")).toBeInTheDocument();
  });

  it("connects keys, then sends an order only after the confirmation", async () => {
    search = new URLSearchParams("ticker=aaa");
    let connected = false;
    const status = () => (connected ? { connected: true, connected_at: new Date().toISOString(), account: { status: "ACTIVE", buying_power: "20000", cash: "10000" } } : { connected: false });
    serve({
      "GET /v1/paper/account": () => jsonResponse(status()),
      "POST /v1/paper/account": (c) => {
        if ((c.body as { api_key: string }).api_key !== "PKGOODKEY1") return jsonResponse({ detail: "Could not validate those keys against the Alpaca paper endpoint." }, 400);
        connected = true;
        return jsonResponse(status());
      },
      "GET /v1/paper/activity": jsonResponse({ connected: true, positions_available: true,
        positions: [{ symbol: "AAA", qty: "3", avg_entry_price: "10", current_price: "11", market_value: "33", unrealized_pl: "3", unrealized_plpc: "0.1" }],
        orders: [{ order_id: "o1", symbol: "AAA", side: "buy", qty: "3", filled_qty: "3", status: "filled", filled_avg_price: "10", submitted_at: new Date().toISOString() }] }),
      "GET /v1/stocks/AAA": jsonResponse({ ticker: "AAA", price: 12.5 }),
      "POST /v1/paper/orders": (c) => jsonResponse({ order_id: "o2", status: "accepted", ticker: "AAA", qty: (c.body as { qty: number }).qty, filled_avg_price: null }, 201),
    });
    const u = userEvent.setup();
    render(<SessionProvider initialMe={me("premium")}><PaperView /></SessionProvider>);
    await u.type(await screen.findByLabelText("API key ID"), "PKBADKEY99");
    await u.type(screen.getByLabelText("API secret"), "secret-123456");
    await u.click(screen.getByRole("button", { name: "Connect paper account" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not validate");
    await u.clear(screen.getByLabelText("API key ID"));
    await u.type(screen.getByLabelText("API key ID"), "PKGOODKEY1");
    await u.click(screen.getByRole("button", { name: "Connect paper account" }));
    expect(await screen.findByText("$20,000.00")).toBeInTheDocument();
    expect(await screen.findByText("$3.00 (+10.00%)")).toBeInTheDocument();
    expect(screen.getByLabelText("Ticker")).toHaveValue("AAA");
    await u.clear(screen.getByLabelText("Shares"));
    await u.type(screen.getByLabelText("Shares"), "4");
    await u.click(screen.getByRole("button", { name: "Review order" }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText("$50.00")).toBeInTheDocument();
    expect(calls.some((c) => c.path === "/v1/paper/orders")).toBe(false);
    await u.click(within(dialog).getByRole("button", { name: "Buy 4 AAA" }));
    expect(await screen.findByText(/Sent: buy 4 AAA \(accepted\)/)).toBeInTheDocument();
    expect(calls.find((c) => c.path === "/v1/paper/orders")!.body).toEqual({ ticker: "AAA", qty: 4, confirm: true });
  });
});

describe("Unsubscribe link", () => {
  it("changes nothing on open, then turns off the email named in the link, or all of them", async () => {
    search = new URLSearchParams("t=linktoken1234&k=alerts");
    let prefs = { digest: true, evening: true, alerts: true };
    serve({
      "GET /v1/email-preferences/unsubscribe": jsonResponse({ email: "pr***@example.com", prefs }),
      "POST /v1/email-preferences/unsubscribe": (c) => {
        const { kind } = c.body as { kind: string };
        prefs = kind === "all" ? { digest: false, evening: false, alerts: false } : { ...prefs, [kind]: false };
        return jsonResponse({ email: "pr***@example.com", prefs });
      },
    });
    const u = userEvent.setup();
    render(<UnsubscribeView />);
    expect(await screen.findByText("pr***@example.com")).toBeInTheDocument();
    expect(calls.map((c) => c.method)).toEqual(["GET"]);
    await u.click(screen.getByRole("button", { name: "Unsubscribe from the alert emails" }));
    expect(await screen.findByText("You're unsubscribed from the alert emails.")).toBeInTheDocument();
    expect(calls[1]!.body).toEqual({ token: "linktoken1234", kind: "alerts" });
    await u.click(screen.getByRole("button", { name: "Unsubscribe from all HSF emails" }));
    expect(await screen.findByText(/unsubscribed from all HSF emails/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Unsubscribe/ })).not.toBeInTheDocument();
  });

  it("says when the link is invalid", async () => {
    search = new URLSearchParams("t=wrongtoken99");
    serve({ "GET /v1/email-preferences/unsubscribe": jsonResponse({ detail: "This unsubscribe link isn't valid." }, 400) });
    const { unmount } = render(<UnsubscribeView />);
    expect(await screen.findByRole("alert")).toHaveTextContent("isn't valid");
    unmount();
    search = new URLSearchParams();
    render(<UnsubscribeView />);
    expect(screen.getByRole("alert")).toHaveTextContent("isn't valid");
    expect(calls).toHaveLength(1);
  });
});

describe("Stair-steppers", () => {
  it("checks the top 40 on request and lists matches, best fit first, and thin data", async () => {
    serve({ "GET /v1/day-trader/stair-steppers": jsonResponse({ checked: [], matches: [
      { ticker: "BBB", status: "ok", r2: 0.86, trend_pct_per_hour: 1.2, max_pullback_pct: 0.4, bars: 45, as_of: new Date().toISOString() },
      { ticker: "AAA", status: "ok", r2: 0.93, trend_pct_per_hour: 0.9, max_pullback_pct: 0.3, bars: 45, as_of: new Date().toISOString() },
    ], all: [{ ticker: "AAA", status: "ok" }, { ticker: "BBB", status: "ok" }, { ticker: "CCC", status: "insufficient" }] }) });
    const symbols = Array.from({ length: 45 }, (_, i) => `S${i}`);
    const u = userEvent.setup();
    render(<StairSteppers symbols={symbols} />);
    expect(screen.getByText("Checks the top 40 of 45 by DT score.")).toBeInTheDocument();
    await u.selectOptions(screen.getByLabelText("Direction"), "either");
    await u.click(screen.getByRole("button", { name: "Check 40 symbols" }));
    const rows = await screen.findAllByRole("row");
    expect(within(rows[1]!).getByRole("link")).toHaveTextContent("AAA");
    expect(within(rows[1]!).getByText("0.930")).toBeInTheDocument();
    expect(screen.getByText(/Not enough 1-minute data to judge \(1\): CCC/)).toBeInTheDocument();
    const q = calls[0]!.query;
    expect(q.get("symbols")!.split(",")).toHaveLength(40);
    expect([q.get("direction"), q.get("window"), q.get("r2_min"), q.get("max_pullback"), q.get("min_trend")]).toEqual(["either", "45", "0.8", "1", "0.5"]);
  });
});
