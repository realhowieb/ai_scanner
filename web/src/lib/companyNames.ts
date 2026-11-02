"use client";

// Short company names under tickers. Every TickerLink on a page asks for its name; the
// asks are gathered for a moment and sent as one /v1/symbols/names request (up to 200
// tickers), then kept for the session. Signed-out pages (401) and failures just show
// no names.
import { useSyncExternalStore } from "react";

import { api, unwrap } from "@/api/client";

const names = new Map<string, string | null>();   // null: asked, no name known
const pending = new Set<string>();
const listeners = new Set<() => void>();
let timer: ReturnType<typeof setTimeout> | null = null;
let version = 0;

const notify = () => { version += 1; listeners.forEach((l) => l()); };

async function flush() {
  timer = null;
  const batch = [...pending].slice(0, 200);
  batch.forEach((t) => pending.delete(t));
  if (pending.size) timer = setTimeout(flush, 0);
  if (!batch.length) return;
  try {
    const res = await unwrap(api.GET("/v1/symbols/names", { params: { query: { tickers: batch.join(",") } } }));
    batch.forEach((t) => names.set(t, res.names[t] ?? null));
  } catch {
    batch.forEach((t) => names.set(t, null));
  }
  notify();
}

function want(ticker: string) {
  if (names.has(ticker) || pending.has(ticker)) return;
  pending.add(ticker);
  timer ??= setTimeout(flush, 40);
}

function subscribe(l: () => void) { listeners.add(l); return () => { listeners.delete(l); }; }

/** The ticker's short company name ("Apple Inc."), or null while loading / unknown. */
export function useCompanyName(ticker: string, enabled = true): string | null {
  const t = ticker.trim().toUpperCase();
  useSyncExternalStore(subscribe, () => version, () => 0);
  if (!enabled || !t) return null;
  if (!names.has(t)) want(t);
  return names.get(t) ?? null;
}

/** Test hook: forget every name. */
export function resetCompanyNames() { names.clear(); pending.clear(); version = 0; }
