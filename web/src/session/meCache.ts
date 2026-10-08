// The last /v1/me answer for this tab, so a full page load can render the page and
// start its own request at once instead of waiting ~0.6 s for /v1/me first. It is
// still revalidated on every load; it holds no token, and every sign-in, sign-out
// and expired session clears it.
import type { Schemas } from "@/api/client";

type Me = Schemas["Me"];

const KEY = "hsf_me";
let lastRaw: string | null = null;
let lastMe: Me | null = null;
const listeners = new Set<() => void>();

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null; // blocked storage: behave as before (wait for /v1/me)
  }
}

export function readCachedMe(): Me | null {
  let raw: string | null = null;
  try {
    raw = storage()?.getItem(KEY) ?? null;
  } catch {
    raw = null;
  }
  if (raw !== lastRaw) {
    lastRaw = raw;
    try {
      lastMe = raw ? (JSON.parse(raw) as Me) : null;
    } catch {
      lastMe = null;
    }
  }
  return lastMe;
}

export function writeCachedMe(me: Me | null): void {
  try {
    const s = storage();
    if (me) s?.setItem(KEY, JSON.stringify(me));
    else s?.removeItem(KEY);
  } catch {
    /* full or blocked: the cache is only a speed-up */
  }
  listeners.forEach((fn) => fn());
}

export function clearCachedMe(): void {
  writeCachedMe(null);
}

export function subscribeCachedMe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
