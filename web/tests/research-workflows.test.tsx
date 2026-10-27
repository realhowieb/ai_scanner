import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { StockSearch } from "@/components/StockSearch";
import { CustomScanView } from "@/features/CustomScanView";
import { CompareStocksView } from "@/features/CompareStocksView";
import { NotificationDelivery } from "@/features/NotificationDelivery";
import { OperationsView } from "@/features/OperationsView";
import { SavedResearchViews } from "@/features/SavedResearchViews";
import { SetupTable } from "@/features/SetupTable";
import type { Schemas } from "@/api/client";
import { SessionProvider } from "@/session/SessionProvider";
import { clearApiCache } from "@/hooks/useApi";
import { me, setup, stockDetail } from "./fixtures";
import { jsonResponse } from "./helpers";

let params = new URLSearchParams();
const push = vi.fn();
vi.mock("next/navigation", () => ({ useSearchParams: () => params, useRouter: () => ({ push, replace: push }) }));
let preferences: Schemas["WorkspacePreferences"];
let deliveries: Schemas["Delivery"][];
let patches: Schemas["WorkspacePatch"][];
let paths: string[];
const fetchMock = vi.fn(async (request: Request) => {
  const url = new URL(request.url); const path = url.pathname.replace("/api/hsf", "");
  paths.push(path);
  if (path === "/v1/me/research-preferences") {
    if (request.method === "PATCH") { const body = await request.json(); patches.push(body); preferences = { ...preferences, ...body }; }
    return jsonResponse(preferences);
  }
  if (path === "/v1/symbols/search") return jsonResponse({ available: true, results: [{ ticker: "AAPL", name: "Apple Inc." }] });
  if (path === "/v1/alerts/deliveries") return jsonResponse(deliveries);
  if (path.endsWith("/retry")) { deliveries = deliveries.map((d) => ({ ...d, status: "accepted", retryable: false })); return jsonResponse(deliveries[0]); }
  if (path === "/v1/stocks/AAPL") return jsonResponse(stockDetail({ ticker: "AAPL", rvol: 1.5 }));
  if (path === "/v1/stocks/MSFT") return jsonResponse({ detail: "Stock data unavailable" }, 503);
  if (path === "/v1/admin/requests") return jsonResponse({ scope: "This API process", total: 1, errors: 1, slow: 0, scan_queue: [{ status: "queued", count: 2 }],
    providers: [{ name: "Alpaca", ok: false, ms: 150, at: "2026-10-16T20:00:00Z" }],
    requests: [{ request_id: "web-code-123", method: "GET", route: "/v1/scans/latest", status: 503, ms: 200, at: "2026-10-16T20:00:00Z" }] });
  if (path === "/v1/sparklines") return jsonResponse({ series: {} });
  return jsonResponse([]);
});
beforeEach(() => {
  params = new URLSearchParams(); paths = []; patches = []; push.mockClear();
  preferences = { scanner_views: [], custom_views: [], table: { columns: ["last", "change", "gap", "rvol"], compact: false },
    quiet_hours: { enabled: false, start: "22:00", end: "07:00", timezone: "America/New_York" } };
  deliveries = [{ id: "failed-receipt", status: "failed", attempts: 1, at: "2026-10-16T20:00:00Z", title: "AAPL alert", retryable: true }];
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => vi.unstubAllGlobals());

it("saves scanner views to the account and reloads them after remount", async () => {
  const apply = vi.fn();
  const first = render(<SavedResearchViews kind="scanner_views" current={{ name: "", min: 60, signal: "breakout", size: 25, sort: "score", when: "today" }} onApply={apply} />);
  await waitFor(() => expect(screen.getByLabelText("Saved views")).toBeEnabled());
  fireEvent.change(screen.getByLabelText("View name"), { target: { value: "Morning" } });
  fireEvent.click(screen.getByText("Save view"));
  await waitFor(() => expect(patches).toHaveLength(1));
  first.unmount(); clearApiCache();
  render(<SavedResearchViews kind="scanner_views" current={{ name: "", min: 0, signal: "", size: 25, sort: "score", when: "today" }} onApply={apply} />);
  await screen.findByRole("option", { name: "Morning" });
  fireEvent.change(screen.getByLabelText("Saved views"), { target: { value: "Morning" } });
  expect(apply).toHaveBeenCalledWith(expect.objectContaining({ min: 60, signal: "breakout" }));
  fireEvent.click(screen.getByText("Delete view"));
  await waitFor(() => expect(patches[1]?.scanner_views).toEqual([]));
});

it("searches company names with keyboard selection and remembers recent tickers", async () => {
  render(<StockSearch />);
  const search = screen.getByRole("combobox");
  fireEvent.focus(search);
  fireEvent.change(search, { target: { value: "Apple" } });
  await screen.findByRole("option", { name: /AAPL Apple Inc/ });
  fireEvent.keyDown(search, { key: "ArrowDown" });
  fireEvent.submit(search.closest("form")!);
  expect(push).toHaveBeenCalledWith("/stocks/AAPL");
  fireEvent.focus(search);
  expect(await screen.findByRole("option", { name: /AAPL Recent search/ })).toBeInTheDocument();
});

it("limits compare selection and saves table preferences without hiding ticker or score", async () => {
  render(<SessionProvider initialMe={me("pro")}><SetupTable rows={["AAA", "BBB", "CCC", "DDD", "EEE"].map((t) => setup(t, 70))} sparks premium={false} /></SessionProvider>);
  await screen.findByText("Table display");
  for (const t of ["AAA", "BBB", "CCC", "DDD"]) fireEvent.click(screen.getAllByLabelText(`Compare ${t}`)[0]!);
  expect(screen.getAllByLabelText("Compare EEE")[0]).toBeDisabled();
  expect(screen.getByText("Compare selected")).toHaveAttribute("href", "/stocks/compare?symbols=AAA%2CBBB%2CCCC%2CDDD");
  fireEvent.click(screen.getByText("Table display"));
  fireEvent.click(screen.getByLabelText("Last price"));
  fireEvent.click(screen.getByLabelText("Compact spacing"));
  fireEvent.click(screen.getByText("Save display"));
  await waitFor(() => expect(patches[0]?.table?.compact).toBe(true));
  await waitFor(() => expect(screen.queryByRole("columnheader", { name: "Last" })).not.toBeInTheDocument());
  expect(screen.getByRole("columnheader", { name: "Ticker" })).toBeInTheDocument();
  expect(screen.getByRole("columnheader", { name: "HSF Score" })).toBeInTheDocument();
});

it("compares independent stock responses and retains a successful column during an outage", async () => {
  params = new URLSearchParams("symbols=AAPL,MSFT");
  render(<CompareStocksView />);
  await screen.findByRole("link", { name: "AAPL" });
  await screen.findByText("Couldn't load MSFT right now");
  expect(screen.getByText("Relative volume")).toBeInTheDocument();
  expect(paths).toContain("/v1/stocks/AAPL"); expect(paths).toContain("/v1/stocks/MSFT");
});

it("rejects more than four compare names before fetching stock data", () => {
  params = new URLSearchParams("symbols=AAPL,MSFT,NVDA,AMD,TSLA");
  render(<CompareStocksView />);
  expect(screen.getByRole("alert")).toHaveTextContent("2–4");
  expect(paths).toEqual([]);
});

it("saves quiet hours and removes retry controls after a notification is accepted", async () => {
  render(<NotificationDelivery />);
  await screen.findByText("Retry notification");
  await screen.findByText("Enable quiet hours");
  fireEvent.click(screen.getByLabelText("Enable quiet hours"));
  fireEvent.change(screen.getByLabelText("Time zone"), { target: { value: "America/Los_Angeles" } });
  fireEvent.click(screen.getByText("Save quiet hours"));
  await waitFor(() => expect(patches[0]?.quiet_hours).toMatchObject({ enabled: true, timezone: "America/Los_Angeles" }));
  fireEvent.click(screen.getByText("Retry notification"));
  await waitFor(() => expect(screen.queryByText("Retry notification")).not.toBeInTheDocument());
  expect(paths).toContain("/v1/alerts/deliveries/failed-receipt/retry");
});

it("shows support codes, provider failures and shared scan queue counts", async () => {
  render(<OperationsView />);
  await screen.findByText(/web-code-123/);
  expect(screen.getByText(/2 queued/)).toBeInTheDocument();
  expect(screen.getByText(/Alpaca: Needs attention/)).toBeInTheDocument();
});


it("refuses saved custom filters that are outside the account's current plan", async () => {
  preferences.custom_views = [{ name: "Pro volume", universe: "sp500", ticker: "", watchlist_id: null,
    session: "regular", top_n: 25, min_price: null, max_price: null, min_dollar_vol: null, unusual: true,
    gap: false, min_gap: null, score_all: false, earnings_window: "today" }];
  const create = vi.fn();
  render(<SessionProvider initialMe={me("basic")}><CustomScanView client={{ list: async () => [], create, get: vi.fn(), cancel: vi.fn() }} /></SessionProvider>);
  await screen.findByRole("option", { name: "Pro volume" });
  fireEvent.change(screen.getByLabelText("Saved views"), { target: { value: "Pro volume" } });
  expect(screen.getByRole("alert")).toHaveTextContent("outside your current plan");
  expect(screen.getByRole("checkbox", { name: /Unusual volume/ })).not.toBeChecked();
  expect(create).not.toHaveBeenCalled();
});
