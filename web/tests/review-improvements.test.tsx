import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { AlertComposer } from "@/features/AlertComposer";
import { AlertsView } from "@/features/AlertsView";
import { StockView } from "@/features/StockView";
import { SymbolLine } from "@/features/OutcomeEvidence";
import { DayTraderView } from "@/features/DayTraderView";
import { scheduledSession, stateHorizon } from "@/lib/researchTime";
import { SessionProvider } from "@/session/SessionProvider";
import { fakeApi } from "./fakeApi";
import { me, stockDetail } from "./fixtures";
import { jsonResponse } from "./helpers";
import types from "./fixtures/alert-types.json";
import outcomes from "./fixtures/outcomes.json";
import type { AlertType } from "@/api/userData";
import type { Schemas } from "@/api/client";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(), usePathname: () => "/alerts", useRouter: () => ({ replace: vi.fn(), push: vi.fn() }) }));
afterEach(() => vi.unstubAllGlobals());
const session = (child: React.ReactNode, plan: "basic" | "premium" = "premium") => <SessionProvider initialMe={me(plan)}>{child}</SessionProvider>;

it("creates a ticker HSF rule with the catalog threshold, preserves server errors, and hides unavailable plan choices", async () => {
  const api = fakeApi({ rules: true });
  api.fail("POST", /^\/v1\/alerts\/rules$/, 403, { detail: "Plan allowance reached." });
  vi.stubGlobal("fetch", api.fetch);
  const done = vi.fn(), user = userEvent.setup();
  render(session(<AlertComposer types={types as AlertType[]} onCreated={done} />, "basic"));
  await screen.findByRole("option", { name: "HSF Score crosses above" });
  expect(screen.getByRole("option", { name: /Becomes PreBreakout/ })).toBeDisabled();
  await user.selectOptions(screen.getByLabelText("Alert type"), "rule:HSF_SCORE_CROSS_ABOVE");
  await user.click(screen.getByRole("button", { name: "Create alert" }));
  expect(api.count("POST", /^\/v1\/alerts\/rules$/)).toBe(0);
  await user.type(screen.getByLabelText("Ticker"), "nvda");
  expect(screen.queryByLabelText("Also email me")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Create alert" }));
  expect(await screen.findByText("Plan allowance reached.")).toBeInTheDocument();
  expect(done).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Create alert" }));
  await waitFor(() => expect(done).toHaveBeenCalledOnce());
  expect(api.calls.filter((c) => c.method === "POST").at(-1)?.body).toEqual({ rule_type: "HSF_SCORE_CROSS_ABOVE", enabled: true, ticker: "NVDA", threshold: 80, delivery_channels: ["in_app"] });
});

it("requires an owned watchlist selection for setup rules and never offers market-wide HSF rules", async () => {
  const api = fakeApi({ rules: true }); const list = api.seedList("Main", ["NVDA"]);
  vi.stubGlobal("fetch", api.fetch);
  const user = userEvent.setup();
  render(session(<AlertComposer types={types as AlertType[]} onCreated={vi.fn()} />));
  await screen.findByRole("option", { name: "New HSF setup" });
  await user.selectOptions(screen.getByLabelText("Monitor"), "watchlist");
  await user.selectOptions(screen.getByLabelText("Alert type"), "rule:SETUP_APPEARED");
  await screen.findByRole("option", { name: "Main" });
  await user.click(screen.getByRole("button", { name: "Create alert" }));
  expect(api.count("POST", /^\/v1\/alerts\/rules$/)).toBe(0);
  await user.selectOptions(screen.getByLabelText("Watchlist"), String(list.id));
  await user.click(screen.getByLabelText("Also email me"));
  await user.click(screen.getByRole("button", { name: "Create alert" }));
  await waitFor(() => expect(api.count("POST", /^\/v1\/alerts\/rules$/)).toBe(1));
  expect(api.calls.find((c) => c.method === "POST")?.body).toEqual({ rule_type: "SETUP_APPEARED", enabled: true, watchlist_id: list.id, delivery_channels: ["in_app", "email"] });
  await user.selectOptions(screen.getByLabelText("Monitor"), "market");
  expect(screen.queryByRole("option", { name: "HSF Score crosses above" })).not.toBeInTheDocument();
  expect(screen.getByLabelText("Breakout Score at or above")).toHaveValue("8");
});

it("shows active HSF rules in the empty activity state and in Rules", async () => {
  const api = fakeApi({ rules: true });
  await api.fetch(new Request("http://localhost/api/hsf/v1/alerts/rules", { method: "POST", body: JSON.stringify({ rule_type: "HSF_SCORE_CROSS_ABOVE", ticker: "NVDA", threshold: 80 }) }));
  vi.stubGlobal("fetch", api.fetch);
  render(session(<AlertsView />));
  expect(await screen.findByText("Your alert is active; no trigger has been recorded yet.")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "View rules" }));
  expect(await screen.findByText("NVDA: HSF Score crosses above 80")).toBeVisible();
  expect(screen.getByRole("tab", { name: "Rules" })).toHaveFocus();
  expect(screen.getByText(/No evaluation recorded yet/)).toBeVisible();
});

it("keeps market-only scan conditions out of stock and watchlist forms", async () => {
  const api = fakeApi({ rules: true });
  vi.stubGlobal("fetch", async (req: Request) => {
    const response = await api.fetch(req);
    if (new URL(req.url).pathname.endsWith("/alerts/rules/types")) {
      return jsonResponse([...await response.json(), {
        type: "SCAN_MATCH", label: "New scanner matches", scope: "market", available: true,
      }]);
    }
    return response;
  });
  const user = userEvent.setup();
  render(session(<AlertComposer types={types as AlertType[]} onCreated={vi.fn()} />));
  await screen.findByRole("option", { name: "New HSF setup" });
  expect(screen.queryByRole("option", { name: "New scanner matches" })).not.toBeInTheDocument();
  await user.selectOptions(screen.getByLabelText("Monitor"), "watchlist");
  expect(screen.queryByRole("option", { name: "New scanner matches" })).not.toBeInTheDocument();
  await user.selectOptions(screen.getByLabelText("Monitor"), "market");
  expect(screen.getByRole("link", { name: "save a scan alert from the Scanner" })).toHaveAttribute("href", "/scanner");
});

it("keeps grouped signals, raw observations, price outcomes and elapsed-time persistence distinct", async () => {
  vi.stubGlobal("fetch", vi.fn(async (req: Request) => new URL(req.url).pathname.includes("/outcomes/")
    ? jsonResponse(outcomes.symbol) : jsonResponse({ detail: "Not found" }, 404)));
  render(<StockView s={stockDetail({ historical_context: { bucket: "70-79", positive_rate: .53, n: 73, sufficient: true }, outcome_cohort: { status: "STRONG", score_band: "75+", horizon: "H24", comparable: 146, follow_through_rate: .01, evidence_strength: "STRONG" } })} premium />);
  const history = screen.getByRole("region", { name: "Historical context" });
  expect(await within(history).findByText(/1 matured signal for AAA/)).toBeInTheDocument();
  expect(within(history).getByRole("region", { name: "Saved ticker observations" })).toHaveTextContent("4 saved scan observations");
  expect(within(history).getByRole("region", { name: "Price outcomes by score range" })).toHaveTextContent("53% reached +4% within 5 trading days · 73 matured observations");
  expect(within(history).getByRole("region", { name: "Setup persistence" })).toHaveTextContent("24 elapsed hours later");
  expect(within(history).getByRole("region", { name: "Setup persistence" })).toHaveTextContent("1% persisted or strengthened · 146 comparable observations");
});

it("uses the returned trading-day horizon for a fallback signal outcome", () => {
  const h = outcomes.symbol.horizons[0]!;
  render(<SymbolLine d={{ ...outcomes.symbol, horizons: [{ ...h, horizon: 3, matured_count: 1, benchmark_count: 1, benchmark_beat_count: 1 }] } as Schemas["OutcomeSymbol"]} />);
  expect(screen.getByText(/Historical HSF evidence/).closest("p")).toHaveTextContent("beat SPY over 3 trading days");
  expect(screen.getByText(/Historical HSF evidence/).closest("p")).toHaveTextContent("Median 3-trading-day return");
});

it("explains weekends only with confirmed schedule coverage and never converts elapsed hours to trading days", () => {
  const scan = "2026-10-16T19:45:00Z", sunday = "2026-10-18T22:00:00Z";
  expect(scheduledSession(scan, sunday, true)).toContain("Friday, Oct 16");
  expect(scheduledSession(scan, sunday, false)).toBeNull();
  expect(scheduledSession(scan, "2026-10-19T15:00:00Z", true)).toBeNull();
  expect(stateHorizon("H24")).toContain("24 elapsed hours");
  expect(stateHorizon("OTHER")).toContain("definition unavailable");
});

it("labels closed-session rankings and retains weak candidates without promoting them", async () => {
  vi.stubGlobal("fetch", vi.fn(async (req: Request) => new URL(req.url).pathname.endsWith("/sparklines") ? jsonResponse({ series: {} }) : jsonResponse({
    state: "closed", as_of: "2026-10-18T22:00:00Z", source: "movers", symbols: ["AAA"], missing: 0, summary: { best_long: "AAA" },
    rows: [{ ticker: "AAA", day_trade_score: 53, dt_quality: "weak", dt_direction: "bullish", dt_reasons: [], dt_conflicts: [], quote_flags: [] }],
  })));
  render(session(<DayTraderView />));
  expect(await screen.findByText("Last-session intraday rankings · market closed.")).toBeInTheDocument();
  expect(screen.getByLabelText("Day Trader summary")).toHaveTextContent("No strong long setup · AAA highest-ranked long · DT 53 · weak");
  expect(screen.getByText(/The API does not supply the session date/)).toBeInTheDocument();
});


it("does not claim strong setups are absent just because the highest-ranked candidate is weak", async () => {
  vi.stubGlobal("fetch", vi.fn(async (req: Request) => new URL(req.url).pathname.endsWith("/sparklines") ? jsonResponse({ series: {} }) : jsonResponse({
    state: "closed", as_of: "2026-10-18T22:00:00Z", source: "movers", symbols: ["AAA", "BBB"], missing: 0, summary: { best_long: "AAA" },
    rows: [{ ticker: "AAA", day_trade_score: 80, dt_quality: "weak", dt_direction: "bullish", dt_reasons: [], dt_conflicts: [], quote_flags: [] },
      { ticker: "BBB", day_trade_score: 75, dt_quality: "strong", dt_direction: "bullish", dt_reasons: [], dt_conflicts: [], quote_flags: [] }],
  })));
  render(session(<DayTraderView />));
  await screen.findByText("Last-session intraday rankings · market closed.");
  expect(screen.getByLabelText("Day Trader summary")).not.toHaveTextContent("No strong long setup");
  expect(screen.getByLabelText("Day Trader summary")).toHaveTextContent("AAA highest-ranked long · DT 80 · weak");
});
