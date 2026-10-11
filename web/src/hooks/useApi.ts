"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/api/client";
import { readCachedMe, subscribeCachedMe } from "@/session/meCache";

export type ApiState<T> = {
  data: T | null;
  error: ApiError | null;
  /** True only while nothing is on screen for this key yet. */
  loading: boolean;
  /** True while a remembered answer is shown and a fresh one is on its way. */
  refreshing: boolean;
  reload: () => void;
};

type Result<T> = { token: string; data: T | null; error: ApiError | null };

// The last answer per key for this tab, so going back to a page shows it at once and
// refreshes it in the background instead of a spinner. In memory only (a full page load
// starts empty); dropped when the signed-in account changes or signs out.
const CACHE_MAX_ENTRIES = 60;
const CACHE_MAX_AGE_MS = 10 * 60_000;
const cache = new Map<string, { data: unknown; at: number }>();
let cacheOwner: string | null = null;

function syncOwner(): void {
  const email = readCachedMe()?.email ?? null;
  if (email !== cacheOwner) {
    cache.clear();
    cacheOwner = email;
  }
}

if (typeof window !== "undefined") subscribeCachedMe(syncOwner);

export function clearApiCache(): void {
  cache.clear();
}

function cachedData(key: string): { data: unknown } | undefined {
  const hit = cache.get(key);
  if (!hit) return undefined;
  if (Date.now() - hit.at > CACHE_MAX_AGE_MS) {
    cache.delete(key);
    return undefined;
  }
  return hit;
}

function remember(key: string, data: unknown): void {
  syncOwner();
  cache.delete(key); // re-insert so the Map's order is least recently written first
  cache.set(key, { data, at: Date.now() });
  while (cache.size > CACHE_MAX_ENTRIES) cache.delete(cache.keys().next().value as string);
}

/** Loads once per `key` (null = don't load), cancels on unmount or key change and
 * keeps the last data while a new key loads. A key answered earlier in this tab shows
 * its remembered data at once while it reloads. */
export function useApi<T>(key: string | null, load: (signal: AbortSignal) => Promise<T>): ApiState<T> {
  const [tick, setTick] = useState(0);
  const [res, setRes] = useState<Result<T> | null>(null);
  const loadRef = useRef(load);
  useEffect(() => {
    loadRef.current = load;
  });
  const token = key === null ? null : `${key}#${tick}`;

  useEffect(() => {
    if (token === null || key === null) return;
    const ctl = new AbortController();
    loadRef.current(ctl.signal).then(
      (data) => {
        remember(key, data);
        setRes({ token, data, error: null });
      },
      (e: unknown) => {
        if (ctl.signal.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
        const error = e instanceof ApiError ? e : new ApiError(0, "Something went wrong.", null, null, null);
        setRes((prev) => ({ token, data: prev?.data ?? null, error }));
      },
    );
    return () => ctl.abort();
  }, [token, key]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  const current = res !== null && res.token === token;
  const hit = key !== null && !current ? cachedData(key) : undefined;
  const data = current ? res.data : hit !== undefined ? (hit.data as T) : (res?.data ?? null);
  return {
    data,
    error: current ? res.error : null,
    loading: token !== null && !current && hit === undefined,
    refreshing: token !== null && !current && hit !== undefined,
    reload,
  };
}
