// Audit and polish pass (2026-11-01): error recovery beside the affected content, no
// duplicate requests, missing data never shown as a measured zero, and 320px reflow.
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AlertsView } from "@/features/AlertsView";
import { JournalInsightsCard } from "@/features/JournalInsights";
import { JournalView } from "@/features/JournalView";
import { SymbolEvidence } from "@/features/OutcomeEvidence";
import { SetupTable } from "@/features/SetupTable";
import { clearApiCache } from "@/hooks/useApi";
import { etDate, num, pct, price, probPct } from "@/lib/format";
import { SessionProvider } from "@/session/SessionProvider";
import { fakeApi } from "./fakeApi";
import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/scanner",
}));

beforeEach(() => clearApiCache());
afterEach(() => vi.unstubAllGlobals());

const outage = (requestId = "srv-503") => jsonResponse({ detail: "The HSF service is unavailable right now." }, 503, { "x-request-id": requestId });

describe("missing numbers", () => {
  it("never formats NaN, Infinity or a bad date as a value", () => {
    expect(pct(Infinity)).toBe("—");
    expect(pct(-Infinity)).toBe("—");
    expect(num(Number.NaN)).toBe("—");
    expect(price(Infinity)).toBe("—");
    expect(probPct(Infinity)).toBe("—");
    expect(etDate("not a date")).toBe("");
    expect(pct(0)).toBe("0.00%"); // a measured zero still shows as zero
  });
});

describe("Alerts page", () => {
  let api: ReturnType<typeof fakeApi>;
  beforeEach(() => {
    api = fakeApi({ rules: true, events: [] });
    vi.stubGlobal("fetch", vi.fn((input: Request | string, init?: RequestInit) => api.fetch(input, init)));
  });

  it("shows unknown counts, not 0 active / 0 paused, when HSF rules fail to load", async () => {
    api.fail("GET", /^\/v1\/alerts\/rules$/, 503);
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    const strip = await screen.findByLabelText("Alert summary");
    await waitFor(() => expect(within(strip).getAllByText("—")).toHaveLength(2));
    expect(within(strip).queryByText("0")).not.toBeInTheDocument();
  });

  it("puts a retry beside Recently fired when it fails, and the retry reloads it", async () => {
    api.fail("GET", /^\/v1\/alerts\/events/, 503);
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    const card = (await screen.findByRole("heading", { name: "Recently fired" })).closest("section")!;
    fireEvent.click(await within(card).findByRole("button", { name: "Try again" }));
    expect(await within(card).findByText("No recent triggers.")).toBeInTheDocument();
    expect(api.count("GET", /^\/v1\/alerts\/events/)).toBe(2);
  });

  it("asks for the alert rules once, though the rules card appears after the page loads", async () => {
    render(<SessionProvider initialMe={me("pro")}><AlertsView /></SessionProvider>);
    await screen.findByLabelText("Alert summary");
    await waitFor(() => expect(api.count("GET", /^\/v1\/alerts\/rules\/types$/)).toBeGreaterThan(0));
    expect(api.count("GET", /^\/v1\/alerts\/rules$/)).toBe(1);
  });
});

describe("Journal", () => {
  it("reads insights once on arrival and again only after a refresh", async () => {
    const paths: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (request: Request) => {
      const url = new URL(request.url);
      paths.push(url.pathname);
      if (url.pathname.endsWith("/insights")) return jsonResponse({ closed: 0, summary: null });
      return jsonResponse({ trades: [{ id: 1, ticker: "OLD", open: true }], has_more: false, stats: null });
    }));
    render(<SessionProvider initialMe={me("pro")}><JournalView /></SessionProvider>);
    await screen.findByRole("link", { name: "OLD" });
    await screen.findByText(/Close a trade to see your win rate/);
    const insights = () => paths.filter((p) => p.endsWith("/v1/journal/insights")).length;
    expect(insights()).toBe(1);
    fireEvent.click(screen.getByRole("button", { name: "Refresh journal" }));
    await waitFor(() => expect(insights()).toBe(2));
  });

  it("explains failed insights with a retry instead of hiding the card", async () => {
    let calls = 0;
    vi.stubGlobal("fetch", vi.fn(async () => (++calls === 1 ? outage() : jsonResponse({ closed: 0, summary: null }))));
    render(<SessionProvider initialMe={me("pro")}><JournalInsightsCard pro /></SessionProvider>);
    expect(await screen.findByText("Couldn't load your journal insights right now")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/Close a trade to see your win rate/)).toBeInTheDocument();
  });
});

describe("Stock page signal history", () => {
  it("says the ticker's history failed, with a retry, instead of leaving the section blank", async () => {
    let calls = 0;
    vi.stubGlobal("fetch", vi.fn(async () => (++calls === 1 ? outage()
      : jsonResponse({ ticker: "ZS", date_range: null, horizons: [{ horizon: 5, sample_size: 0, matured_count: 0 }] }))));
    render(<SymbolEvidence ticker="ZS" />);
    expect(await screen.findByText("Couldn't load ZS's signal history right now")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("No earlier HSF signals for ZS.")).toBeInTheDocument();
  });

  it("stays quiet on an API without the outcomes route", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "Not Found" }, 404)));
    const { container } = render(<SymbolEvidence ticker="OLD" />);
    await waitFor(() => expect(screen.queryByText(/Loading/)).not.toBeInTheDocument());
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Scanner table", () => {
  it("uses the page's research preferences instead of asking again", async () => {
    const paths: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (request: Request) => { paths.push(new URL(request.url).pathname); return jsonResponse({}); }));
    const prefs = { data: { table: { columns: ["last", "change"], compact: false } }, error: null, loading: false, refreshing: false, updatedAt: 1, reload: () => {} };
    render(<SessionProvider initialMe={me("pro")}>
      <SetupTable rows={[{ ticker: "ZS", score: 80, last: 10 } as never]} premium={false} prefs={prefs as never} />
    </SessionProvider>);
    expect(screen.getAllByText("ZS").length).toBeGreaterThan(0);
    await new Promise((r) => setTimeout(r, 20));
    expect(paths.filter((p) => p.endsWith("/research-preferences"))).toHaveLength(0);
  });
});

describe("320px reflow and tap targets", () => {
  const css = readFileSync(join(__dirname, "..", "src", "app", "globals.css"), "utf8");
  it("wraps Setup mix rows (360px and below) and the Scanner header buttons (400px and below)", () => {
    expect(css).toMatch(/@media \(max-width: 360px\) \{\n  \.mix-row \{ flex-wrap: wrap;/);
    expect(css).toMatch(/@media \(max-width: 400px\) \{\n[^}]*\.page-head \.scanner-actions \{ flex-wrap: wrap; \}/);
  });
  it("gives ticker links and phone compare boxes 24px targets", () => {
    expect(css).toContain("a.tk { min-height: 24px; line-height: 24px; }");
    expect(css).toContain('.sc-top input[type="checkbox"] { width: 24px; height: 24px;');
  });
});
