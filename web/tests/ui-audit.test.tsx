import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { WatchlistsView } from "@/features/WatchlistsView";
import { JournalView } from "@/features/JournalView";
import { SnapshotTime } from "@/features/SnapshotTime";
import { CompareStocksView } from "@/features/CompareStocksView";
import { ScanHistoryView } from "@/features/ScanHistoryView";
import { AlertsView } from "@/features/AlertsView";
import { DayTraderView } from "@/features/DayTraderView";
import { PasswordField } from "@/components/PasswordField";
import { annualValue } from "@/features/Landing";
import { frac } from "@/features/AdminAnalytics";
import { SessionProvider } from "@/session/SessionProvider";
import { me, stockDetail } from "./fixtures";
import { fakeApi } from "./fakeApi";
import { jsonResponse } from "./helpers";
let params = new URLSearchParams();
vi.mock("next/navigation", () => ({ useSearchParams: () => params, useRouter: () => ({ replace: vi.fn(), push: vi.fn() }), usePathname: () => "/scanner" }));
const at = "2026-10-16T20:00:00Z";
beforeEach(() => { params = new URLSearchParams(); });
afterEach(() => vi.unstubAllGlobals());
const session = (children: React.ReactNode) => <SessionProvider initialMe={me("premium")}>{children}</SessionProvider>;
it("uses a missed-slot flag for matching snapshots and never treats an archive as current", async () => {
  const fetch = vi.fn(async () => jsonResponse({ scan_at: at, stale: false, freshness: { state: "fresh" } }));
  vi.stubGlobal("fetch", fetch);
  const view = render(<SnapshotTime at={at} />);
  expect(await screen.findByText(/On schedule/)).toBeInTheDocument();
  view.unmount(); fetch.mockClear();
  render(<SnapshotTime at={at} archived />);
  expect(screen.getByText(/Archived snapshot/)).toHaveTextContent("Saved snapshot");
  expect(fetch).not.toHaveBeenCalled();
});
it("keeps unknown schedule coverage neutral instead of applying an overnight age rule", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ scan_at: "2026-10-15T20:00:00Z", stale: true })));
  render(<SnapshotTime at={at} />);
  expect(await screen.findByText(/Saved snapshot/)).toBeInTheDocument();
  expect(screen.queryByText(/Scan delayed/)).not.toBeInTheDocument();
});
it("highlights differing and missing comparison fields after both columns load", async () => {
  params = new URLSearchParams("symbols=AAPL,MSFT");
  vi.stubGlobal("fetch", vi.fn(async (req: Request) => {
    const path = new URL(req.url).pathname;
    if (path.endsWith("/latest")) return jsonResponse({ scan_at: at, stale: false });
    return jsonResponse(stockDetail({ ticker: path.endsWith("MSFT") ? "MSFT" : "AAPL", scan_at: at, rvol: path.endsWith("MSFT") ? null : 2 }));
  }));
  render(<CompareStocksView />);
  expect(await screen.findByText("Highlighted differences: Relative volume.")).toBeInTheDocument();
  expect(screen.getAllByText(/means this scan did not return a value/)).toHaveLength(2);
  expect(screen.getAllByText(/not an all-clear/)).toHaveLength(2);
});
it("retrieves saved scans by source and ET date without changing the archive", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse([
    { id: 1, label: "SP500", created_at: "2026-10-16T23:00:00Z", row_count: 20, duration_s: 125 },
    { id: 2, label: "NASDAQ", created_at: "2026-10-17T02:00:00Z", row_count: 5, duration_s: 40 },
  ])));
  render(session(<ScanHistoryView />));
  expect(await screen.findByRole("link", { name: "S&P 500" })).toBeInTheDocument();
  const u = userEvent.setup();
  await u.selectOptions(screen.getByLabelText("Scan source"), "NASDAQ");
  fireEvent.change(screen.getByLabelText("Through (ET)"), { target: { value: "2026-10-16" } });
  expect(screen.getByRole("link", { name: "NASDAQ" })).toHaveAttribute("href", "/scanner/history/2");
  expect(screen.queryByRole("link", { name: "S&P 500" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("From (ET)"), { target: { value: "2026-10-17" } });
  expect(screen.getByText("No scans match these filters.")).toBeInTheDocument();
  await u.click(screen.getByRole("button", { name: "Clear filters" }));
  expect(screen.getByText("2m 5s")).toBeInTheDocument();
});
it("lets keyboard users switch alert panels and returns focus after closing creation", async () => {
  const api = fakeApi(); vi.stubGlobal("fetch", api.fetch);
  render(session(<AlertsView />)); const u = userEvent.setup();
  const activity = await screen.findByRole("tab", { name: "Activity" });
  activity.focus(); await u.keyboard("{ArrowRight}");
  expect(screen.getByRole("tab", { name: "Rules" })).toHaveFocus();
  expect(screen.getByRole("tabpanel", { name: "Rules" })).toBeVisible();
  await u.click(screen.getByRole("button", { name: "New alert" }));
  const dialog = screen.getByRole("dialog", { name: "New alert" });
  expect(await within(dialog).findByRole("button", { name: "Create alert" })).toBeEnabled();
  await u.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "New alert" })).toHaveFocus();
});
it("shows fifteen candidates first and exposes the complete Day Trader list on request", async () => {
  const rows = Array.from({ length: 20 }, (_, i) => ({ ticker: `TEST${i}`, day_trade_score: 90 - i, dt_quality: "developing", dt_direction: "bullish", dt_reasons: [], dt_conflicts: [], quote_flags: [] }));
  vi.stubGlobal("fetch", vi.fn(async (req: Request) => jsonResponse(new URL(req.url).pathname.includes("/sparklines") ? { series: {} } : { state: "closed", source: "watchlist", symbols: rows.map((r) => r.ticker), missing: 0, as_of: at, rows })));
  render(session(<DayTraderView />));
  const u = userEvent.setup();
  const all = await screen.findByRole("button", { name: "Show all 20 candidates" });
  expect(screen.queryByRole("link", { name: "TEST19" })).not.toBeInTheDocument();
  await u.click(all);
  expect(screen.getByRole("link", { name: "TEST19" })).toBeInTheDocument();
  await u.click(screen.getByRole("button", { name: "Show first 15" }));
  expect(screen.queryByRole("link", { name: "TEST19" })).not.toBeInTheDocument();
});
it("reveals passwords without submitting or losing the entered value", async () => {
  const submit = vi.fn((e: React.FormEvent) => e.preventDefault());
  render(<form onSubmit={submit}><PasswordField label="New password" defaultValue="local-test-value" /></form>);
  const u = userEvent.setup();
  await u.click(screen.getByRole("button", { name: "Show new password" }));
  expect(screen.getByLabelText("New password")).toHaveAttribute("type", "text");
  expect(screen.getByLabelText("New password")).toHaveValue("local-test-value");
  await u.click(screen.getByRole("button", { name: "Hide new password" }));
  expect(submit).not.toHaveBeenCalled();
});
it("calculates annual value from actual prices and preserves small research returns", () => {
  expect(annualValue("$25/mo", "$250/yr")).toBe("$20.83/mo equivalent · save 17% vs monthly");
  expect(annualValue("Free", "$0/yr")).toBeNull();
  expect(annualValue("$25/mo", "$350/yr")).toBeNull();
  expect(frac(-0.0002, true)).toBe("-0.020%");
  expect(frac(-0, true)).toBe("0.0%");
});

it("labels a matching missed scheduled scan as delayed", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ scan_at: at, stale: true, freshness: { state: "stale" } })));
  render(<SnapshotTime at={at} />);
  expect(await screen.findByText(/Scan delayed/)).toBeInTheDocument();
});
it("shows a dated bar for an unranked watched stock without inventing a scan score", async () => {
  const api = fakeApi({ scanState: true }); api.seedList("Main", ["AAPL"]);
  vi.stubGlobal("fetch", (req: Request) => new URL(req.url).pathname.endsWith("/sparklines")
    ? Promise.resolve(jsonResponse({ series: {}, last_bar: { AAPL: { close: 172, session: "2026-10-16" } } })) : api.fetch(req));
  render(session(<WatchlistsView />));
  expect(await screen.findByText("$172.00", { exact: false })).toHaveTextContent("daily bar Oct 16");
  expect(screen.getByText("Not ranked in the latest scan")).toBeInTheDocument();
  expect(screen.queryByLabelText(/HSF Score \d/)).not.toBeInTheDocument();
});
it("keeps return statistics unavailable while all journal trades remain open", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ trades: [{ id: 1, ticker: "AAPL", open: true, shares: 1, entry_price: 100, mark: 101, pnl: 1, pnl_pct: 1, entered_at: at }],
    stats: { closed: 0, wins: 0, avg_return_pct: 0 }, summary: { open: 1, closed: 0, avg_return_pct: 0, win_rate: 0 } })));
  render(session(<JournalView />));
  expect(await screen.findByText("No closed trades yet; return and win-rate statistics are unavailable.")).toBeInTheDocument();
  const stats = screen.getByLabelText("Journal summary");
  expect(stats).toHaveTextContent("— avg return");
  expect(stats).toHaveTextContent("— win rate");
  expect(stats).not.toHaveTextContent("0.00%");
});
