// Features built on API endpoints the app wasn't using yet: the evidence explorer
// (/v1/outcomes/query), results through time (/v1/outcomes/timeseries), past signals on a
// stock page (records from /v1/outcomes/symbols), releasing a founder seat
// (/v1/founding/checkout/cancel) and the Admin ML rollback (/v1/admin/ml/rollback).
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { OutcomeSymbol } from "@/api/account";
import { ModelsTab } from "@/features/AdminModels";
import { FounderView } from "@/features/FounderView";
import { OutcomeEvidence, PastSignals, explorerQuery } from "@/features/OutcomeEvidence";

import oc from "./fixtures/outcomes.json";
import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/track-record",
}));

type Call = { method: string; path: string; query: URLSearchParams; body: unknown };
let calls: Call[] = [];
let routes: Record<string, (c: Call) => Response> = {};

beforeEach(() => {
  search = new URLSearchParams();
  calls = [];
  routes = {};
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const url = new URL(r.url);
    const text = r.method === "GET" ? "" : await r.text();
    const c = { method: r.method, path: url.pathname.replace(/^\/api\/(hsf|public)/, ""), query: url.searchParams, body: text ? JSON.parse(text) : null };
    calls.push(c);
    const hit = routes[`${c.method} ${c.path}`];
    return hit ? hit(c) : jsonResponse({ detail: `no route ${c.method} ${c.path}` }, 404);
  }));
});
afterEach(() => vi.unstubAllGlobals());

const envelope = { ...oc.summary, filters: oc.symbol.filters };
const point = (period_start: string, median_excess_return: number, matured_count: number, evidence_quality = "LIMITED") => ({
  period_start, observation_count: matured_count + 1, matured_count, pending_count: 1, evidence_quality,
  average_return: 0.01, median_return: 0.01, average_excess_return: median_excess_return, median_excess_return,
  win_rate: 0.5, benchmark_beat_rate: 0.6, benchmark_count: matured_count,
});

function evidenceRoutes() {
  routes["GET /v1/outcomes/summary"] = () => jsonResponse(oc.summary);
  routes["GET /v1/outcomes/scores"] = () => jsonResponse(oc.scores);
  routes["GET /v1/outcomes/horizons"] = () => jsonResponse(oc.horizons);
  routes["GET /v1/outcomes/setups"] = () => jsonResponse(oc.setups);
  routes["GET /v1/outcomes/timeseries"] = (c) => jsonResponse({ ...envelope, period: c.query.get("period"), points: [
    point("2026-10-05", 0.012, 14), point("2026-10-12", -0.004, 22), point("2026-10-19", 0.02, 3, "INSUFFICIENT")] });
  routes["GET /v1/outcomes/query"] = (c) => jsonResponse({ ...envelope, horizon: Number(c.query.get("horizon")),
    metrics: oc.symbol.horizons[2], coverage: oc.summary.coverage,
    observations: { page: Number(c.query.get("page")), page_size: 25, total: 60, items: oc.symbol.observations.items } });
}

describe("Track record: through time", () => {
  it("charts weekly results and switches to months", async () => {
    evidenceRoutes();
    render(<OutcomeEvidence />);
    const card = (await screen.findByRole("heading", { name: "Through time" })).closest("section")!;
    expect(await within(card).findByText(/3 periods from/)).toBeInTheDocument();
    expect(card).toHaveTextContent("2 of 3 periods were above SPY");
    expect(calls.find((c) => c.path === "/v1/outcomes/timeseries")!.query.get("period")).toBe("week");
    await userEvent.click(within(card).getByLabelText("By month"));
    await waitFor(() => expect(calls.some((c) => c.path === "/v1/outcomes/timeseries" && c.query.get("period") === "month")).toBe(true));
  });
});

describe("Track record: explore past signals", () => {
  it("lists matching signals and sends only the filters chosen", async () => {
    evidenceRoutes();
    const u = userEvent.setup();
    render(<OutcomeEvidence />);
    const card = (await screen.findByRole("heading", { name: "Explore past signals" })).closest("section")!;
    expect(await within(card).findByText("60 signals")).toBeInTheDocument();
    expect(within(card).getByRole("link", { name: "AAA" })).toHaveAttribute("href", "/stocks/AAA");
    expect(card).toHaveTextContent("Page 1 of 3");
    const first = calls.find((c) => c.path === "/v1/outcomes/query")!;
    expect([...first.query.keys()].sort()).toEqual(["horizon", "page", "page_size"]);

    await u.selectOptions(within(card).getByLabelText("Setup"), "gapper");
    await u.type(within(card).getByLabelText("HSF score from"), "80");
    await u.type(within(card).getByLabelText("Ticker"), "aaa");
    await u.selectOptions(within(card).getByLabelText("Held for"), "3");
    await u.click(within(card).getByRole("button", { name: "Show signals" }));
    await waitFor(() => expect(calls.some((c) => c.path === "/v1/outcomes/query" && c.query.get("setup") === "gapper")).toBe(true));
    const q = calls.filter((c) => c.path === "/v1/outcomes/query").at(-1)!.query;
    expect(Object.fromEntries(q)).toEqual({ setup: "gapper", min_score: "80", ticker: "AAA", horizon: "3", page: "1", page_size: "25" });

    await u.click(within(card).getByRole("button", { name: "Older" }));
    await waitFor(() => expect(calls.some((c) => c.path === "/v1/outcomes/query" && c.query.get("page") === "2")).toBe(true));
  });

  it("swaps a reversed score range and leaves blanks out", () => {
    const q = explorerQuery({ setup: "", min: "90", max: "70", from: "", to: "", ticker: " ", horizon: 5 }, 1);
    expect(q).toEqual({ setup: undefined, min_score: 70, max_score: 90, start_date: undefined, end_date: undefined, ticker: undefined, horizon: 5, page: 1, page_size: 25 });
  });
});

describe("Stock page: past signals", () => {
  it("lists each earlier signal with its 1, 3 and 5 day result and a pending one as pending", async () => {
    const base = oc.symbol as unknown as OutcomeSymbol;
    const pending = { ...base.observations.items[0]!, observation_id: "2", entry_day: "2026-11-10", hsf_score: 77,
      outcomes: [{ horizon: 1, status: "matured" as const, raw_return: -0.01, excess_return: -0.012 }, { horizon: 3, status: "pending" as const }, { horizon: 5, status: "pending" as const }] };
    const d = { ...base, observations: { ...base.observations, total: 14, items: [pending, ...base.observations.items] } };
    render(<PastSignals d={d} />);
    await userEvent.click(screen.getByText("Past signals on AAA (14)"));
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("77");
    expect(rows[0]).toHaveTextContent("-1.00%");
    expect(within(rows[0]!).getAllByText("pending").length).toBeGreaterThanOrEqual(2);
    expect(rows[1]).toHaveTextContent("82");
    expect(rows[1]).toHaveTextContent("+4.00%");
    expect(screen.getByText(/Newest 2 of 14 signals/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Track record" })).toHaveAttribute("href", "/track-record?ticker=AAA#explore");
  });

  it("shows nothing when the ticker has no signals", () => {
    const base = oc.symbol as unknown as OutcomeSymbol;
    const { container } = render(<PastSignals d={{ ...base, observations: { ...base.observations, total: 0, items: [] } }} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Founding 100: back from Stripe without paying", () => {
  const offer = {
    open: true, checkout_open: true, preview: false, missing_details: [], acknowledgment: "I understand.", policies: [], membership: null, tiers: [],
    upgrade: { price_cents: 50000, eligible: false, on_sale: false, disclosure: { name: "x", payment: "a: b", duration: "a: b", recurring: "a: b", refund: "a: b" } },
  };

  it("releases the held seat on request", async () => {
    search = new URLSearchParams("checkout=cancelled");
    routes["GET /v1/founding/offer"] = () => jsonResponse(offer);
    routes["POST /v1/founding/checkout/cancel"] = () => jsonResponse({ released: true, paid: false });
    render(<FounderView />);
    expect(await screen.findByText(/Your seat is held for a few more minutes/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /release my seat/ }));
    expect(await screen.findByText(/Seat released/)).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "POST").map((c) => c.path)).toEqual(["/v1/founding/checkout/cancel"]);
  });

  it("says so when nothing was held", async () => {
    search = new URLSearchParams("checkout=cancelled");
    routes["GET /v1/founding/offer"] = () => jsonResponse(offer);
    routes["POST /v1/founding/checkout/cancel"] = () => jsonResponse({ released: false, paid: false });
    render(<FounderView />);
    await userEvent.click(await screen.findByRole("button", { name: /release my seat/ }));
    expect(await screen.findByText(/no open checkout to close/)).toBeInTheDocument();
  });

  it("offers no release button on a normal visit", async () => {
    routes["GET /v1/founding/offer"] = () => jsonResponse(offer);
    render(<FounderView />);
    expect(await screen.findByRole("heading", { name: "Founding 100" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /release my seat/ })).toBeNull();
  });
});

describe("Admin > Models: roll back a champion", () => {
  const champion = {
    id: 3, model_version: "ml-v4-catboost-production-f5d-000000000003", provider: "catboost", kind: "model", feature_set: "production",
    target_label: "F_clean_path_5d", horizon_days: 5, dataset_version: "hsf-ml-2026-11-13-v1", status: "champion",
    status_changed_at: "2026-11-20T15:00:00Z", trained_at: null, created_at: null, has_artifact: true, artifact_sha256: "c".repeat(64),
    calibration: null, metrics: {},
  };

  it("needs a reason and sends the champion's target and horizon with approval", async () => {
    routes["GET /v1/admin/ml/models"] = () => jsonResponse({ models: [champion], count: 1 });
    routes["GET /v1/admin/ml/models/3"] = () => jsonResponse({ model: champion, events: [], shadow: { scored: 0, labeled: 0, entry_days: 0, last_scored_at: null }, promotion_gates: null });
    routes["POST /v1/admin/ml/rollback"] = () => jsonResponse({ restored: 2, retired: 3, note: "Registry only: AI Confidence and alerts are unchanged." });
    const u = userEvent.setup();
    render(<ModelsTab />);
    await u.click(await screen.findByRole("button", { name: `Open ${champion.model_version}` }));
    const dialog = await screen.findByRole("dialog");
    await u.click(await within(dialog).findByRole("button", { name: "Roll back to previous champion" }));
    const confirm = within(dialog).getByRole("button", { name: "Confirm roll back (I approve)" });
    expect(confirm).toBeDisabled();
    await u.type(within(dialog).getAllByLabelText(/Reason/).at(-1)!, "drift after promotion");
    await u.click(confirm);
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.path).toBe("/v1/admin/ml/rollback");
    expect(post.body).toEqual({ target_label: "F_clean_path_5d", horizon_days: 5, reason: "drift after promotion", approve: true });
    expect(await within(dialog).findByRole("status")).toHaveTextContent("model #2 is champion again and #3 is retired");
  });
});
