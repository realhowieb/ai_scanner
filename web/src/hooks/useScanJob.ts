"use client";

// Custom-scan lifecycle: start (POST /v1/scans → 202), poll GET /v1/scans/{id} every
// 2-5 s until complete or failed, resume an active scan after navigation or reload
// (the API remembers it, so nothing is kept in the browser), back off on 429/503
// for Retry-After, and stop polling on completion, failure, unmount or sign-out.
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";

export type ScanJob = Schemas["ScanJob"];
// The generated types mark fields that have server defaults as required (they are
// always present in responses); in a request they are optional, so the request
// type says so and the server fills the defaults.
export type ScanCreate = Omit<Schemas["ScanCreate"], "filters" | "score_all"> & {
  score_all?: boolean;
  filters?: Partial<Schemas["ScanFilters"]>;
};

export type ScanApi = {
  create: (body: ScanCreate) => Promise<ScanJob>;
  get: (id: string, signal?: AbortSignal) => Promise<ScanJob>;
  list: (signal?: AbortSignal) => Promise<ScanJob[]>;
  cancel: (id: string) => Promise<ScanJob>;
};

export const defaultScanApi: ScanApi = {
  create: (body) => unwrap(api.POST("/v1/scans", { body: body as Schemas["ScanCreate"] })) as Promise<ScanJob>,
  get: (id, signal) => unwrap(api.GET("/v1/scans/{scan_id}", { params: { path: { scan_id: id } }, signal })),
  list: (signal) => unwrap(api.GET("/v1/scans", { params: { query: { limit: 5 } }, signal })),
  cancel: (id) => unwrap(api.DELETE("/v1/scans/{scan_id}", { params: { path: { scan_id: id } } })),
};

/** The API's error text for a scan stopped with DELETE /v1/scans/{id}. */
export const CANCELLED = "Cancelled.";

export const ACTIVE = new Set(["queued", "running"]);
const DEFAULT_BACKOFF_S = 10;
const TRANSIENT = new Set([0, 429, 502, 503, 504]);

export function clampInterval(ms: number): number {
  return Math.min(5000, Math.max(2000, ms));
}

export type ScanJobState = {
  job: ScanJob | null;
  error: ApiError | null;
  starting: boolean;
  /** Seconds until a refused start may be retried (429/503 Retry-After). */
  retryInS: number | null;
  start: (body: ScanCreate) => Promise<void>;
  cancel: () => Promise<void>;
  cancelling: boolean;
  dismiss: () => void;
};

export function useScanJob({ enabled = true, intervalMs = 3000, client = defaultScanApi }: {
  enabled?: boolean; intervalMs?: number; client?: ScanApi;
} = {}): ScanJobState {
  const [job, setJob] = useState<ScanJob | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [starting, setStarting] = useState(false);
  const [retryInS, setRetryInS] = useState<number | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ctl = useRef<AbortController | null>(null);
  const alive = useRef(true);
  const every = clampInterval(intervalMs);

  const stop = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    ctl.current?.abort();
    ctl.current = null;
  }, []);

  // The poll loop lives in a ref so it can schedule itself.
  const loop = useRef<(id: string, delayMs: number) => void>(() => {});
  useEffect(() => {
    loop.current = (id: string, delayMs: number) => {
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(async () => {
        if (!alive.current) return;
        ctl.current = new AbortController();
        try {
          const j = await client.get(id, ctl.current.signal);
          if (!alive.current) return;
          setJob(j);
          if (ACTIVE.has(j.status)) loop.current(id, every);
        } catch (e) {
          if (!alive.current || (e instanceof DOMException && e.name === "AbortError")) return;
          if (e instanceof ApiError && TRANSIENT.has(e.status)) {
            loop.current(id, (e.retryAfterS ?? DEFAULT_BACKOFF_S) * 1000); // keep the scan, wait, try again
            return;
          }
          // 401 (signed out: the session handler takes over), 404 (gone) or anything else: stop.
          setError(e instanceof ApiError ? e : null);
          if (e instanceof ApiError && e.status === 404) setJob(null);
        }
      }, delayMs);
    };
  }, [client, every]);
  const poll = useCallback((id: string, delayMs: number) => loop.current(id, delayMs), []);

  // Resume the account's active scan, if any.
  useEffect(() => {
    alive.current = true;
    if (!enabled) return;
    const c = new AbortController();
    client.list(c.signal)
      .then((jobs) => {
        const active = jobs.find((j) => ACTIVE.has(j.status));
        if (active && alive.current) {
          setJob(active);
          poll(active.scan_id, 0);
        }
      })
      .catch(() => {});
    return () => {
      alive.current = false;
      c.abort();
      stop();
    };
  }, [enabled, client, poll, stop]);

  useEffect(() => {
    if (!enabled) stop();
  }, [enabled, stop]);

  useEffect(() => {
    if (retryInS === null || retryInS <= 0) return;
    const t = setTimeout(() => setRetryInS((s) => (s && s > 1 ? s - 1 : null)), 1000);
    return () => clearTimeout(t);
  }, [retryInS]);

  const start = useCallback(async (body: ScanCreate) => {
    setStarting(true);
    setError(null);
    setRetryInS(null);
    try {
      const j = await client.create(body);
      setJob(j);
      poll(j.scan_id, every);
    } catch (e) {
      const err = e instanceof ApiError ? e : new ApiError(0, "Couldn't start the scan.", null, null, null);
      const running = (err.body as { scan_id?: unknown } | null)?.scan_id;
      if (err.status === 409 && typeof running === "string") {
        poll(running, 0); // already running: follow that one
      }
      if ((err.status === 429 || err.status === 503) && err.retryAfterS) setRetryInS(err.retryAfterS);
      setError(err);
    } finally {
      if (alive.current) setStarting(false);
    }
  }, [client, poll, every]);

  const cancel = useCallback(async () => {
    const id = job?.scan_id;
    if (!id) return;
    setCancelling(true);
    stop();
    try {
      const j = await client.cancel(id);
      if (alive.current) setJob(j);
      if (alive.current && ACTIVE.has(j.status)) poll(id, every); // shouldn't happen; keep following it
    } catch (e) {
      if (alive.current) {
        setError(e instanceof ApiError ? e : null);
        poll(id, every); // the scan is still there: keep showing it
      }
    } finally {
      if (alive.current) setCancelling(false);
    }
  }, [job, client, stop, poll, every]);

  const dismiss = useCallback(() => {
    stop();
    setJob(null);
    setError(null);
  }, [stop]);

  return { job, error, starting, retryInS, start, cancel, cancelling, dismiss };
}
