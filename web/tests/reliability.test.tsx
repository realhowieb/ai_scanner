import { act, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import { JournalView } from "@/features/JournalView";
import { PaperView } from "@/features/PaperView";
import { StockSearchView } from "@/features/StockSearchView";
import { useAction } from "@/hooks/useAction";
import { SessionProvider } from "@/session/SessionProvider";
import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

const push = vi.fn();
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push, replace: vi.fn() }),
  usePathname: () => "/stocks",
}));
afterEach(() => { vi.unstubAllGlobals(); push.mockClear(); });

it("locks a mutation before React can rerender, then permits another after completion", async () => {
  let finish: (value: string) => void = () => {};
  const task = vi.fn(() => new Promise<string>((resolve) => { finish = resolve; }));
  const { result } = renderHook(useAction);
  let first: Promise<unknown> = Promise.resolve();
  await act(async () => {
    first = result.current.run(task);
    expect(await result.current.run(task)).toBeUndefined();
  });
  expect(task).toHaveBeenCalledOnce();
  expect(result.current.busy).toBe(true);
  await act(async () => { finish("done"); await first; });
  await act(async () => { expect(await result.current.run(async () => "next")).toBe("next"); });
  expect(result.current.busy).toBe(false);
});

it("releases the action lock after a synchronous failure", async () => {
  const { result } = renderHook(useAction);
  await act(async () => { await result.current.run(() => { throw new Error("failed"); }); });
  expect(result.current.error).not.toBeNull();
  await act(async () => { expect(await result.current.run(async () => true)).toBe(true); });
  expect(result.current.error).toBeNull();
});

it.each(["loading", "empty", "failed"])("keeps ticker search usable when the scan is %s", async (state) => {
  vi.stubGlobal("fetch", vi.fn(() => state === "loading" ? new Promise<Response>(() => {}) : Promise.resolve(
    state === "failed" ? jsonResponse({ detail: "Temporarily unavailable" }, 503)
      : jsonResponse({ setups: [], scan_at: null, total: 0, max_results: 25 }),
  )));
  render(<StockSearchView />);
  if (state === "failed") await screen.findByText("Couldn't load the latest scan right now");
  if (state === "empty") await screen.findByText("No ranked setups in the latest scan.");
  fireEvent.change(screen.getByLabelText("Open any ticker"), { target: { value: "nvda" } });
  fireEvent.click(screen.getByRole("button", { name: "Open" }));
  expect(push).toHaveBeenCalledWith("/stocks/NVDA");
});

it("pages the journal and lets users return from an empty last page", async () => {
  const offsets: number[] = [];
  vi.stubGlobal("fetch", vi.fn(async (request: Request) => {
    const url = new URL(request.url);
    const offset = Number(url.searchParams.get("offset"));
    offsets.push(offset);
    return jsonResponse({ trades: offset ? [] : [{ id: 1001, ticker: "OLD", open: true }], has_more: !offset, stats: null });
  }));
  render(<SessionProvider initialMe={me("pro")}><JournalView /></SessionProvider>);
  await screen.findByRole("link", { name: "OLD" });
  fireEvent.click(screen.getByRole("button", { name: "Next page" }));
  await screen.findByText("No trades on this page.");
  expect(screen.getByText("Page 2")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
  await screen.findByRole("link", { name: "OLD" });
  expect(offsets).toContain(50);
  await waitFor(() => expect(offsets.filter((offset) => offset === 0).length).toBe(2));
});

it("labels unavailable paper totals without claiming the account is empty", async () => {
  vi.stubGlobal("fetch", vi.fn(async (request: Request) => new URL(request.url).pathname.endsWith("/account")
    ? jsonResponse({ connected: true, account: { status: "ACTIVE" } })
    : jsonResponse({ connected: true, positions_available: false, positions: [], orders: [],
      summary: { positions: null, market_value: null, unrealized_pl: null, orders: 0 } })));
  render(<SessionProvider initialMe={me("premium")}><PaperView /></SessionProvider>);
  const summary = await screen.findByLabelText("Paper activity summary");
  expect(within(summary).getAllByText("Unavailable")).toHaveLength(3);
  expect(within(summary).queryByText("$0.00")).not.toBeInTheDocument();
  expect(screen.queryByText("No open paper positions.")).not.toBeInTheDocument();
});
