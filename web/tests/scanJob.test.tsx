import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/api/client";
import { clampInterval, useScanJob } from "@/hooks/useScanJob";
import type { ScanApi, ScanJob } from "@/hooks/useScanJob";

function job(status: ScanJob["status"], extra: Partial<ScanJob> = {}): ScanJob {
  return {
    scan_id: "a".repeat(32), status, universe: "sp500",
    params: { universe: "sp500", ticker: null, watchlist_id: null, score_all: false, profile: "regular", session_requested: "regular",
      session: "regular", min_price: 1, max_price: 1000, min_dollar_vol: 0, min_gap: 0, apply_gap_filter: false, unusual_volume: false,
      top_n: 25, max_results: 25, full_lists: false, max_nasdaq: null, max_combo: null },
    progress: { phase: status === "running" ? "scanning" : status, symbols: 500, elapsed_s: 3 },
    result: status === "complete" ? { label: "SP500", session: "regular", symbols_scanned: 500, duration_s: 9, total: 0, setups: [] } : null,
    error: status === "failed" ? "The scan failed." : null, created_at: null, started_at: null, finished_at: null, ...extra,
  };
}

function client(gets: Array<ScanJob | ApiError>, opts: { list?: ScanJob[]; create?: () => Promise<ScanJob> } = {}): ScanApi & { calls: { get: number } } {
  const calls = { get: 0 };
  return {
    calls,
    list: vi.fn(async () => opts.list ?? []),
    create: vi.fn(opts.create ?? (async () => job("queued"))),
    get: vi.fn(async () => {
      const next = gets[Math.min(calls.get, gets.length - 1)]!;
      calls.get += 1;
      if (next instanceof ApiError) throw next;
      return next;
    }),
  };
}

async function tick(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("custom scan lifecycle", () => {
  it("polls queued → running → complete every 3 s and then stops", async () => {
    const c = client([job("running"), job("running"), job("complete")]);
    const { result } = renderHook(() => useScanJob({ client: c, intervalMs: 3000 }));
    await act(async () => { await result.current.start({ universe: "sp500" }); });
    expect(result.current.job?.status).toBe("queued");
    await tick(2999);
    expect(c.calls.get).toBe(0);
    await tick(1);
    expect(result.current.job?.status).toBe("running");
    await tick(6000);
    expect(result.current.job?.status).toBe("complete");
    expect(c.calls.get).toBe(3);
    await tick(30_000);
    expect(c.calls.get).toBe(3);
  });

  it("stops polling on failure and shows the job's safe error", async () => {
    const c = client([job("failed")]);
    const { result } = renderHook(() => useScanJob({ client: c }));
    await act(async () => { await result.current.start({ universe: "sp500" }); });
    await tick(3000);
    expect(result.current.job?.error).toBe("The scan failed.");
    await tick(20_000);
    expect(c.calls.get).toBe(1);
  });

  it("resumes the account's active scan after a reload", async () => {
    const c = client([job("complete")], { list: [job("complete", { scan_id: "b".repeat(32) }), job("running")] });
    const { result } = renderHook(() => useScanJob({ client: c }));
    await tick(0);
    expect(c.get).toHaveBeenCalledWith("a".repeat(32), expect.anything());
    expect(result.current.job?.status).toBe("complete");
  });

  it("waits for Retry-After when polling is rate limited, keeping the scan", async () => {
    const c = client([new ApiError(429, "Too many", null, 20, null), job("complete")]);
    const { result } = renderHook(() => useScanJob({ client: c }));
    await act(async () => { await result.current.start({ universe: "sp500" }); });
    await tick(3000);
    expect(c.calls.get).toBe(1);
    expect(result.current.job?.status).toBe("queued");
    await tick(19_000);
    expect(c.calls.get).toBe(1);
    await tick(1000);
    expect(result.current.job?.status).toBe("complete");
  });

  it("stops polling when unmounted (navigation) or signed out (401)", async () => {
    const c = client([job("running")]);
    const { result, unmount } = renderHook(() => useScanJob({ client: c }));
    await act(async () => { await result.current.start({ universe: "sp500" }); });
    await tick(3000);
    unmount();
    await tick(30_000);
    expect(c.calls.get).toBe(1);

    const c2 = client([new ApiError(401, "ended", null, null, { code: "session_expired" })]);
    const h2 = renderHook(() => useScanJob({ client: c2 }));
    await act(async () => { await h2.result.current.start({ universe: "sp500" }); });
    await tick(3000);
    await tick(30_000);
    expect(c2.calls.get).toBe(1);
  });

  it("follows the already-running scan on 409", async () => {
    const c = client([job("complete")], {
      create: async () => { throw new ApiError(409, "You already have a scan running.", null, null, { scan_id: "c".repeat(32) }); },
    });
    const { result } = renderHook(() => useScanJob({ client: c }));
    await act(async () => { await result.current.start({ universe: "nasdaq" }); });
    await tick(0);
    expect(c.get).toHaveBeenCalledWith("c".repeat(32), expect.anything());
    expect(result.current.job?.status).toBe("complete");
  });

  it("counts down Retry-After when the service is busy (503) or the hourly limit is hit (429)", async () => {
    for (const status of [503, 429]) {
      const c = client([], { create: async () => { throw new ApiError(status, "busy", "web-12345678", 5, null); } });
      const { result, unmount } = renderHook(() => useScanJob({ client: c }));
      await act(async () => { await result.current.start({ universe: "sp500" }); });
      expect(result.current.retryInS).toBe(5);
      await tick(1000);
      expect(result.current.retryInS).toBe(4);
      for (let i = 0; i < 4; i++) await tick(1000);
      expect(result.current.retryInS).toBeNull();
      unmount();
    }
  });

  it("keeps the poll interval within 2-5 s", () => {
    expect(clampInterval(500)).toBe(2000);
    expect(clampInterval(3000)).toBe(3000);
    expect(clampInterval(60_000)).toBe(5000);
  });
});
