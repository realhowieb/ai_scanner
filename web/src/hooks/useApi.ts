"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/api/client";

export type ApiState<T> = { data: T | null; error: ApiError | null; loading: boolean; reload: () => void };

type Result<T> = { token: string; data: T | null; error: ApiError | null };

/** Loads once per `key` (null = don't load), cancels on unmount or key change and
 * keeps the last data while a new key loads. */
export function useApi<T>(key: string | null, load: (signal: AbortSignal) => Promise<T>): ApiState<T> {
  const [tick, setTick] = useState(0);
  const [res, setRes] = useState<Result<T> | null>(null);
  const loadRef = useRef(load);
  useEffect(() => {
    loadRef.current = load;
  });
  const token = key === null ? null : `${key}#${tick}`;

  useEffect(() => {
    if (token === null) return;
    const ctl = new AbortController();
    loadRef.current(ctl.signal).then(
      (data) => setRes({ token, data, error: null }),
      (e: unknown) => {
        if (ctl.signal.aborted || (e instanceof DOMException && e.name === "AbortError")) return;
        const error = e instanceof ApiError ? e : new ApiError(0, "Something went wrong.", null, null, null);
        setRes((prev) => ({ token, data: prev?.data ?? null, error }));
      },
    );
    return () => ctl.abort();
  }, [token]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  const current = res !== null && res.token === token;
  return { data: res?.data ?? null, error: current ? res.error : null, loading: token !== null && !current, reload };
}
