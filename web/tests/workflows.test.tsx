import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Schemas } from "@/api/client";
import { setSessionExpiredHandler } from "@/api/client";
import { AlertsView } from "@/features/AlertsView";
import { SetupTable } from "@/features/SetupTable";
import { StockView } from "@/features/StockView";
import { WatchlistsView } from "@/features/WatchlistsView";
import { SessionProvider } from "@/session/SessionProvider";

import { fakeApi } from "./fakeApi";
import { me, setup } from "./fixtures";

let search = new URLSearchParams();
const replace = vi.fn((url: string) => { search = new URLSearchParams(url.split("?")[1] ?? ""); });
vi.mock("next/navigation", () => ({
  useSearchParams: () => search,
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/watchlists",
}));

let api: ReturnType<typeof fakeApi>;
beforeEach(() => {
  search = new URLSearchParams();
  replace.mockClear();
  api = fakeApi({ scan: [{ ticker: "AAPL", score: 81, last: 201.5 }] });
  vi.stubGlobal("fetch", vi.fn((input: Request | string, init?: RequestInit) => api.fetch(input, init)));
});
afterEach(() => vi.unstubAllGlobals());

const user = () => userEvent.setup();
const wrap = (ui: React.ReactNode, plan: "basic" | "pro" | "premium" = "pro") => render(<SessionProvider initialMe={me(plan)}>{ui}</SessionProvider>);

describe("Watchlists", () => {
  it("empty state → create the first list; a duplicate name is refused inside the dialog", async () => {
    const u = user();
    wrap(<WatchlistsView />);
    await u.click(await screen.findByRole("button", { name: "Create your first watchlist" }));
    const dlg = screen.getByRole("dialog", { name: "New watchlist" });
    expect(within(dlg).getByLabelText("Name")).toHaveFocus();
    await u.type(within(dlg).getByLabelText("Name"), "Breakouts");
    await u.click(within(dlg).getByRole("button", { name: "Create" }));
    expect(await screen.findByRole("heading", { name: /Breakouts/ })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(api.lists).toHaveLength(1);

    await u.click(screen.getByRole("button", { name: "New watchlist" }));
    const again = screen.getByRole("dialog", { name: "New watchlist" });
    await u.type(within(again).getByLabelText("Name"), "breakouts");
    await u.click(within(again).getByRole("button", { name: "Create" }));
    expect(await within(again).findByRole("alert")).toHaveTextContent('already have a watchlist named "breakouts"');
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(api.lists).toHaveLength(1);
  });

  it("rename, make default and delete (confirmed; Keep it changes nothing)", async () => {
    api.seedList("Main", ["MSFT"], true);
    const second = api.seedList("Swing", ["NVDA"]);
    search = new URLSearchParams(`id=${second.id}`);
    const u = user();
    wrap(<WatchlistsView />);
    await screen.findByRole("heading", { name: /Swing/ });

    await u.click(screen.getByRole("button", { name: "Rename" }));
    const dlg = screen.getByRole("dialog", { name: "Rename watchlist" });
    await u.clear(within(dlg).getByLabelText("Name"));
    await u.type(within(dlg).getByLabelText("Name"), "Swing trades");
    await u.click(within(dlg).getByRole("button", { name: "Rename" }));
    expect(await screen.findByRole("heading", { name: /Swing trades/ })).toBeInTheDocument();

    await u.click(screen.getByRole("button", { name: "Make default" }));
    await waitFor(() => expect(api.lists.find((w) => w.id === second.id)?.is_default).toBe(true));
    expect(await screen.findByRole("button", { name: "Default list" })).toBeDisabled();

    await u.click(screen.getByRole("button", { name: "Delete" }));
    await u.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Keep it" }));
    expect(api.lists).toHaveLength(2);
    await u.click(screen.getByRole("button", { name: "Delete" }));
    await u.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete watchlist" }));
    await waitFor(() => expect(api.lists.map((w) => w.name)).toEqual(["Main"]));
    expect(await screen.findByRole("heading", { name: /Main/ })).toBeInTheDocument();
  });

  it("a failed delete keeps the list and shows the error with its support code", async () => {
    api.seedList("Keep me", ["AAPL"]);
    api.fail("DELETE", /^\/v1\/watchlists\/\d+$/, 503);
    const u = user();
    wrap(<WatchlistsView />);
    await screen.findByRole("heading", { name: /Keep me/ });
    await u.click(screen.getByRole("button", { name: "Delete" }));
    await u.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete watchlist" }));
    const alert = await within(screen.getByRole("dialog")).findByRole("alert");
    expect(alert).toHaveTextContent("unavailable");
    expect(alert).toHaveTextContent("Support code: srv-injected");
    expect(api.lists).toHaveLength(1);
    expect(screen.getAllByText("Keep me").length).toBeGreaterThan(0);
  });

  it("adds tickers (reports already present and invalid), edits a note, removes a ticker after confirming", async () => {
    api.seedList("Main", ["AAPL"]);
    const u = user();
    wrap(<WatchlistsView />);
    await screen.findByRole("heading", { name: /Main/ });
    await u.type(screen.getByLabelText("Add tickers"), "msft, aapl BAD$$");
    await u.click(screen.getByRole("button", { name: "Add" }));
    expect(await screen.findByText(/Added MSFT\. Already in the list: AAPL\. Not valid ticker symbols \(not saved\): BAD\$\$/)).toBeInTheDocument();
    expect(screen.getByLabelText("Add tickers")).toHaveValue("BAD$$");
    expect(await screen.findByRole("link", { name: "MSFT" })).toHaveAttribute("href", "/stocks/MSFT");

    await u.click(screen.getByRole("button", { name: "Add note for AAPL" }));
    await u.type(screen.getByLabelText("Note for AAPL"), "Earnings next week");
    await u.click(screen.getByRole("button", { name: "Save note" }));
    expect(await screen.findByText("Earnings next week")).toBeInTheDocument();
    expect(api.lists[0]!.items.find((i) => i.ticker === "AAPL")!.note).toBe("Earnings next week");

    await u.click(screen.getByRole("button", { name: "Remove MSFT from Main" }));
    await u.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(screen.queryByRole("link", { name: "MSFT" })).not.toBeInTheDocument());
    expect(api.lists[0]!.items.map((i) => i.ticker)).toEqual(["AAPL"]);
  });

  it("an older API: scores from one scan request, only for tickers among the plan's rows", async () => {
    api.seedList("Main", ["AAPL", "ZZZ", "QQQ"]);
    wrap(<WatchlistsView />);
    expect(await screen.findByLabelText("HSF Score 81")).toBeInTheDocument();
    expect(screen.getAllByText("Not among your plan's ranked rows in the latest scan")).toHaveLength(2);
    expect(api.count("GET", /^\/v1\/scans\/latest/)).toBe(1);
    expect(api.count("GET", /^\/v1\/stocks\//)).toBe(0);
  });

  it("shows each ticker's scan state from the list itself, sortable by HSF Score", async () => {
    api = fakeApi({ scanState: true, scan: [{ ticker: "NVDA", score: 90, last: 120 }, { ticker: "AAPL", score: 81, last: 201.5 }] });
    api.seedList("Main", ["AAPL", "NVDA", "ZZZ"]);
    const u = user();
    wrap(<WatchlistsView />);
    expect(await screen.findByLabelText("HSF Score 81")).toBeInTheDocument();
    expect(screen.getByText("#2 of 2")).toBeInTheDocument();          // AAPL's place in the scan
    expect(screen.getByText("Not ranked in the latest scan")).toBeInTheDocument();
    expect(screen.getAllByText("Breakout")).toHaveLength(2);
    expect(api.count("GET", /^\/v1\/scans\/latest/)).toBe(0);      // no second request
    const order = () => screen.getAllByRole("link").map((a) => a.textContent).filter((t) => ["AAPL", "NVDA", "ZZZ"].includes(t ?? ""));
    expect(order()).toEqual(["AAPL", "NVDA", "ZZZ"]);
    await u.click(screen.getByLabelText("HSF Score"));
    expect(order()).toEqual(["NVDA", "AAPL", "ZZZ"]);
  });
});

describe("Watchlist Intelligence", () => {
  it("shows the server's changes, PreBreakout, RVOL and alert counts; adds and deletes a list alert", async () => {
    api = fakeApi({ scanState: true, rules: true, scan: [{ ticker: "NVDA", score: 90, last: 120 }, { ticker: "AAPL", score: 81, last: 201.5 }],
      intel: { NVDA: { score_change: 6, rank_change: 3, prebreakout: true, rvol: 3.14 }, AAPL: { score_change: -2, rank_change: -1, rvol: null } } });
    const w = api.seedList("Main", ["AAPL", "NVDA"]);
    const u = user();
    wrap(<WatchlistsView />, "premium");
    expect(await screen.findByText("▲6")).toBeInTheDocument();
    expect(screen.getByText("Up 3 places")).toBeInTheDocument();
    expect(screen.getByText("▼2")).toBeInTheDocument();
    expect(screen.getByText("Down 1 place")).toBeInTheDocument();
    expect(screen.getByText("PreBreakout")).toBeInTheDocument();
    expect(screen.getByText("RVOL 3.1x")).toBeInTheDocument();

    const card = await screen.findByRole("heading", { name: "Alerts on this list" });
    expect(card).toBeInTheDocument();
    const select = screen.getByLabelText("Alert me when any ticker");
    expect(within(select).queryByRole("option", { name: "Becomes PreBreakout" })).not.toBeInTheDocument(); // not on this plan
    await u.click(screen.getByRole("button", { name: "Add alert" }));
    await waitFor(() => expect(api.rules).toHaveLength(1));
    const posted = api.calls.find((c) => c.method === "POST" && c.path === "/v1/alerts/rules")!.body;
    expect(posted).toEqual({ rule_type: "HSF_SCORE_CROSS_ABOVE", watchlist_id: w.id, enabled: true, threshold: 80 });
    expect(await screen.findByText("HSF Score crosses above 80")).toBeInTheDocument();
    expect(await screen.findAllByText("1 alert")).toHaveLength(2);      // counts come back from the server
    await u.click(screen.getByRole("button", { name: "Delete alert HSF Score crosses above" }));
    await waitFor(() => expect(api.rules).toHaveLength(0));
  });

  it("an API without intelligence or rules leaves the page as it was", async () => {
    api = fakeApi({ scanState: true, scan: [{ ticker: "AAPL", score: 81, last: 201.5 }] });
    api.seedList("Main", ["AAPL"]);
    wrap(<WatchlistsView />);
    expect(await screen.findByLabelText("HSF Score 81")).toBeInTheDocument();
    await waitFor(() => expect(api.count("GET", /\/intelligence$/)).toBe(1));
    expect(screen.queryByRole("heading", { name: "Alerts on this list" })).not.toBeInTheDocument();
    expect(screen.queryByText(/RVOL/)).not.toBeInTheDocument();
  });
});

describe("Save to watchlist", () => {
  it("from a Scanner row: add, then 'already in it', and create-and-save", async () => {
    api.seedList("Main", ["MSFT"]);
    const u = user();
    wrap(<SetupTable rows={[setup("AAPL", 80)]} premium={false} />);
    await u.click(screen.getAllByRole("button", { name: "Save AAPL to a watchlist" })[0]!);
    const dlg = screen.getByRole("dialog", { name: "Save AAPL to a watchlist" });
    await u.click(await within(dlg).findByRole("button", { name: /Main/ }));
    expect(await within(dlg).findByText("Saved AAPL to Main.")).toBeInTheDocument();
    expect(within(dlg).getByRole("button", { name: /Main/ })).toBeDisabled();
    await u.type(within(dlg).getByLabelText("New watchlist"), "Fresh");
    await u.click(within(dlg).getByRole("button", { name: "Create and save" }));
    expect(await within(dlg).findByText("Saved AAPL to Fresh.")).toBeInTheDocument();
    expect(api.lists.find((w) => w.name === "Fresh")!.items.map((i) => i.ticker)).toEqual(["AAPL"]);
    await u.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("a ticker already on the list is reported by the server, not duplicated", async () => {
    api.seedList("Main", ["AAPL"]);
    const u = user();
    wrap(<SetupTable rows={[setup("AAPL", 80)]} premium={false} />);
    await u.click(screen.getAllByRole("button", { name: "Save AAPL to a watchlist" })[0]!);
    await u.click(await screen.findByRole("button", { name: /Main/ }));
    expect(await screen.findByText("AAPL is already in Main.")).toBeInTheDocument();
    expect(api.lists[0]!.items).toHaveLength(1);
  });
});

const stock = (over: Partial<Schemas["StockDetail"]> = {}): Schemas["StockDetail"] => ({
  ticker: "AAPL", scan_at: new Date().toISOString(), in_latest_scan: true, has_setup: true, from_history: false, price: 201.5, change_pct: 1,
  hsf_score: 81, status: "STRONG", primary_setup: "breakout", signals: [], score_components: null, movement: null, score_change: null,
  reasons: [], risks: [], watch_next: [], breakout_score: null, prob: null, earnings_days: null, history_summary: null,
  historical_context: null, outcome_cohort: null, historical_locked: false, lifecycle: [], bars: [], bars_as_of: null,
  watchlists: [], alerts: [], ...over,
});

describe("Stock Intelligence actions", () => {
  it("knows which lists already hold the ticker, and reports a save", async () => {
    const main = api.seedList("Main", ["AAPL"]);
    api.seedList("Other");
    const onChanged = vi.fn();
    const u = user();
    wrap(<StockView s={stock({ watchlists: [{ id: main.id, name: "Main" }] })} premium={false} onChanged={onChanged} />);
    expect(screen.getByRole("link", { name: "Main" })).toHaveAttribute("href", `/watchlists?id=${main.id}`);
    await u.click(screen.getByRole("button", { name: "Watch AAPL elsewhere" }));
    const dlg = screen.getByRole("dialog");
    expect(await within(dlg).findByRole("button", { name: /Main.*Already in it/ })).toBeDisabled();
    await u.click(within(dlg).getByRole("button", { name: /Other/ }));
    await within(dlg).findByText("Saved AAPL to Other.");
    expect(onChanged).toHaveBeenCalled();
  });

  it("price alert: prefilled with the ticker and scan price, labelled not live; non-firing value created", async () => {
    const onChanged = vi.fn();
    const u = user();
    wrap(<StockView s={stock()} premium={false} onChanged={onChanged} />);
    await u.click(screen.getByRole("button", { name: "Alert near $201.50" }));
    const dlg = screen.getByRole("dialog", { name: "Price alert for AAPL" });
    expect(await within(dlg).findByText(/Last scan price \$201\.50 .*not a live quote/)).toBeInTheDocument();
    expect(within(dlg).queryByLabelText("Alert type")).not.toBeInTheDocument();
    expect(within(dlg).getByLabelText("Ticker")).toHaveValue("AAPL");
    expect(within(dlg).getByLabelText("Price ($)")).toHaveValue("201.5");
    await u.clear(within(dlg).getByLabelText("Price ($)"));
    await u.type(within(dlg).getByLabelText("Price ($)"), "999999");
    await u.click(within(dlg).getByRole("button", { name: "Create alert" }));
    expect(await within(dlg).findByText("Created: AAPL price rises above $999,999.")).toBeInTheDocument();
    expect(api.alertRows).toMatchObject([{ type: "price", ticker: "AAPL", threshold: 999999, direction: "above" }]);
    expect(onChanged).toHaveBeenCalled();
  });

  it("at the plan's alert limit the dialog says so instead of offering the form", async () => {
    api = fakeApi({ alertLimit: 1 });
    api.alertRows.push({ id: 99, type: "watchlist", ticker: null, threshold: null, direction: null, watchlist_only: false, enabled: true, last_fired_at: null, created_at: "" });
    const u = user();
    wrap(<StockView s={stock()} premium={false} />, "basic");
    await u.click(screen.getByRole("button", { name: "Alert near $201.50" }));
    expect(await screen.findByText("You're using all 1 alert on your plan.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create alert" })).not.toBeInTheDocument();
  });
});

describe("Alerts", () => {
  it("validates with the server's rules before sending; shows duplicates and server 422s", async () => {
    const u = user();
    wrap(<AlertsView />);
    const form = await screen.findByRole("button", { name: "Create alert" });
    await u.selectOptions(screen.getByLabelText("Alert type"), "price");
    await u.type(screen.getByLabelText("Ticker"), "NVDA");
    await u.type(screen.getByLabelText("Price ($)"), "0");
    await u.click(form);
    expect(screen.getByText("Price ($) must be greater than 0.")).toBeInTheDocument();
    expect(api.count("POST", /^\/v1\/alerts$/)).toBe(0);

    await u.clear(screen.getByLabelText("Price ($)"));
    await u.type(screen.getByLabelText("Price ($)"), "999999");
    await u.click(form);
    expect(await screen.findByText("NVDA price rises above $999,999")).toBeInTheDocument();
    expect(screen.getByText("Using 1 of 5 on your Pro plan")).toBeInTheDocument();

    await u.selectOptions(screen.getByLabelText("Alert type"), "price");
    await u.type(screen.getByLabelText("Ticker"), "NVDA");
    await u.type(screen.getByLabelText("Price ($)"), "999999");
    await u.click(screen.getByRole("button", { name: "Create alert" }));
    expect(await screen.findByText("You already have this alert.", { exact: false })).toBeInTheDocument();
    expect(api.alertRows).toHaveLength(1);
  });

  it("starts each type with the web's default threshold (breakout 8, move 5)", async () => {
    const u = user();
    wrap(<AlertsView />);
    await screen.findByRole("button", { name: "Create alert" });
    expect(screen.getByLabelText("Breakout Score at or above")).toHaveValue("8");
    await u.selectOptions(screen.getByLabelText("Alert type"), "move");
    expect(screen.getByLabelText("Move at least (%)")).toHaveValue("5");
    await u.selectOptions(screen.getByLabelText("Alert type"), "watchlist");
    expect(screen.queryByLabelText(/Threshold|at least|Price/)).not.toBeInTheDocument();
  });

  it("turns an alert off and on, and deletes it after confirming", async () => {
    api.alertRows.push({ id: 7, type: "rvol", ticker: "TSLA", threshold: 2, direction: null, watchlist_only: false, enabled: true, last_fired_at: null, created_at: new Date().toISOString() });
    const u = user();
    wrap(<AlertsView />);
    await u.click(await screen.findByRole("button", { name: "Turn off: TSLA trades at 2× its average volume" }));
    expect(await screen.findByRole("button", { name: "Turn on: TSLA trades at 2× its average volume" })).toBeInTheDocument();
    expect(api.alertRows[0]!.enabled).toBe(false);
    await u.click(screen.getByRole("button", { name: "Delete: TSLA trades at 2× its average volume" }));
    await u.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete alert" }));
    expect(await screen.findByText("No alerts yet.")).toBeInTheDocument();
  });

  it("a failed toggle leaves the alert as it was and explains why", async () => {
    api.alertRows.push({ id: 8, type: "move", ticker: "AMD", threshold: 5, direction: null, watchlist_only: false, enabled: true, last_fired_at: null, created_at: "" });
    api.fail("PATCH", /^\/v1\/alerts\/8$/, 503);
    const u = user();
    wrap(<AlertsView />);
    await u.click(await screen.findByRole("button", { name: /Turn off: AMD/ }));
    expect(await screen.findByText(/unavailable/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Turn off: AMD/ })).toBeInTheDocument();
    expect(api.alertRows[0]!.enabled).toBe(true);
  });

  it("at the limit: no form, a clear next step (upgrade from Free)", async () => {
    api = fakeApi({ alertLimit: 1, emailEnabled: false });
    api.alertRows.push({ id: 1, type: "watchlist", ticker: null, threshold: null, direction: null, watchlist_only: false, enabled: true, last_fired_at: null, created_at: "" });
    wrap(<AlertsView />, "basic");
    expect(await screen.findByText("You're using all 1 alert on your plan.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Create alert" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Upgrade to Pro" }).length).toBeGreaterThan(0);
    expect(screen.getByText("Email alerts are part of Pro.")).toBeInTheDocument();
  });

  it("email delivery toggle saves the preference; push is not promised; empty history", async () => {
    const u = user();
    wrap(<AlertsView />);
    const box = await screen.findByRole("checkbox", { name: /Email me when an alert fires/ });
    expect(box).toBeChecked();
    await u.click(box);
    await waitFor(() => expect(api.prefs.alerts).toBe(false));
    await waitFor(() => expect(screen.getByRole("checkbox", { name: /Email me when an alert fires/ })).not.toBeChecked());
    expect(screen.getByText("Phone push notifications aren't available yet.")).toBeInTheDocument();
    expect(screen.getByText("Nothing has fired yet.")).toBeInTheDocument();
  });

  it("a mutation that finds the session expired hands over to sign-in once", async () => {
    const expired = vi.fn();
    setSessionExpiredHandler(expired);
    api.alertRows.push({ id: 9, type: "move", ticker: "AMD", threshold: 5, direction: null, watchlist_only: false, enabled: true, last_fired_at: null, created_at: "" });
    api.fail("PATCH", /^\/v1\/alerts\/9$/, 401, { detail: "Your session has ended. Sign in again.", code: "session_expired" });
    const u = user();
    wrap(<AlertsView />);
    await u.click(await screen.findByRole("button", { name: /Turn off: AMD/ }));
    await waitFor(() => expect(expired).toHaveBeenCalledTimes(1));
  });
});

describe("before the matching API is deployed", () => {
  it("explains instead of breaking when GET /v1/alerts/types doesn't exist yet", async () => {
    api.fail("GET", /^\/v1\/alerts\/types$/, 404, { detail: "Not Found" });
    wrap(<AlertsView />);
    expect(await screen.findByRole("note")).toHaveTextContent("needs the latest HSF API");
    expect(screen.getByText("No alerts yet.")).toBeInTheDocument();
  });
});

describe("dialog keyboard behaviour", () => {
  it("Escape closes and focus returns to the opener", async () => {
    api.seedList("Main");
    const u = user();
    wrap(<WatchlistsView />);
    const opener = await screen.findByRole("button", { name: "Rename" });
    await u.click(opener);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    await u.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });
});

