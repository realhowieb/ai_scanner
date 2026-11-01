// Research polish leftovers (2026-11-01): watchlist quotes cover every ticker, not the first 50.
import { afterEach, describe, expect, it, vi } from "vitest";

import { QUOTE_CHUNK, watchlistQuotes } from "@/features/WatchlistsView";

import { jsonResponse } from "./helpers";

afterEach(() => vi.unstubAllGlobals());

const tickers = (n: number) => Array.from({ length: n }, (_, i) => `T${String(i).padStart(3, "0")}`);
const symbolsOf = (r: Request) => (new URL(r.url).searchParams.get("symbols") ?? "").split(",");

describe("Watchlist quotes", () => {
  it("asks for every ticker in pieces of 50 and merges the answers", async () => {
    const fetch = vi.fn(async (input: Request) => jsonResponse({ feed: "iex",
      quotes: Object.fromEntries(symbolsOf(input).map((t) => [t, { last: 1, chg_pct: 0, at: "2026-10-30T19:59:00Z", source: "trade" }])) }));
    vi.stubGlobal("fetch", fetch);
    const all = tickers(120);
    const out = await watchlistQuotes(all, new AbortController().signal);
    expect(fetch).toHaveBeenCalledTimes(3);
    expect(fetch.mock.calls.map(([r]) => symbolsOf(r as Request).length)).toEqual([QUOTE_CHUNK, QUOTE_CHUNK, 20]);
    expect(Object.keys(out.quotes ?? {}).sort()).toEqual(all);
    expect(out.feed).toBe("iex");
  });

  it("keeps the pieces that answered when one fails, and fails only when all do", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: Request) => symbolsOf(input).includes("T000")
      ? jsonResponse({ detail: "busy" }, 503)
      : jsonResponse({ feed: "sip", quotes: Object.fromEntries(symbolsOf(input).map((t) => [t, { last: 2, chg_pct: 1, at: null }])) })));
    const out = await watchlistQuotes(tickers(60), new AbortController().signal);
    expect(Object.keys(out.quotes ?? {})).toHaveLength(10);
    expect(out.feed).toBe("sip");

    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "busy" }, 503)));
    await expect(watchlistQuotes(tickers(60), new AbortController().signal)).rejects.toBeTruthy();
  });
});

describe("Hard page load", () => {
  it("starts each request once when this tab already knows the account (no cancel-and-resend after hydration)", async () => {
    vi.resetModules();
    sessionStorage.setItem("hsf_me", JSON.stringify({ email: "pro@example.com", plan: "pro" }));
    const React = await import("react");
    const { renderToString } = await import("react-dom/server");
    const { hydrateRoot } = await import("react-dom/client");
    const { act } = await import("react");
    const { useApi } = await import("@/hooks/useApi");
    const starts: string[] = [];
    const aborts: string[] = [];
    function Probe() {
      const s = useApi<string>("tape", (signal) => {
        starts.push("tape");
        signal.addEventListener("abort", () => aborts.push("tape"));
        return new Promise(() => {});
      });
      return React.createElement("p", null, s.loading ? "loading" : "done");
    }
    const html = renderToString(React.createElement(Probe));
    const box = document.createElement("div");
    box.innerHTML = html;
    document.body.appendChild(box);
    let root: ReturnType<typeof hydrateRoot> | undefined;
    await act(async () => { root = hydrateRoot(box, React.createElement(Probe)); });
    await act(async () => { await new Promise((r) => setTimeout(r, 20)); });
    expect(starts).toEqual(["tape"]);
    expect(aborts).toEqual([]);
    act(() => root?.unmount());
    box.remove();
    sessionStorage.clear();
  });
});
