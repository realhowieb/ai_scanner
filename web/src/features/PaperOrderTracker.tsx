"use client";

import Link from "next/link";
import { useEffect, useState, useSyncExternalStore } from "react";

import { api, unwrap } from "@/api/client";
import { ErrorLine, RefreshStatus } from "@/components/ui";
import { useApi } from "@/hooks/useApi";

const event = "hsf-paper-tracking";
const empty = () => null;
function subscribe(listener: () => void) {
  window.addEventListener(event, listener);
  window.addEventListener("storage", listener);
  return () => {
    window.removeEventListener(event, listener);
    window.removeEventListener("storage", listener);
  };
}

/** Only the confirmation ID is stored; checks can recover it after a page reload. */
export function usePaperTracking(owner: string | undefined) {
  const slot = `hsf-paper-tracking:${owner}`;
  const stored = useSyncExternalStore(subscribe, () => {
    try {
      const key = sessionStorage.getItem(slot);
      return key && /^[A-Za-z0-9_-]{8,128}$/.test(key) ? key : null;
    } catch { return null; }
  }, empty);
  const [local, setLocal] = useState<{ slot: string; key: string | null } | null>(null);
  const save = (key: string | null) => {
    setLocal({ slot, key });
    try { if (key) sessionStorage.setItem(slot, key); else sessionStorage.removeItem(slot); } catch { /* mounted recovery still works */ }
    window.dispatchEvent(new Event(event));
  };
  return { key: local?.slot === slot ? local.key : stored, save };
}

export function PaperOrderTracker({ confirmationId }: { confirmationId: string }) {
  const state = useApi(`paper-confirmation:${confirmationId}`, (signal) =>
    unwrap(api.GET("/v1/paper/orders/{confirmation_id}", {
      params: { path: { confirmation_id: confirmationId } }, signal,
    })));
  const data = state.data;
  const { loading, refreshing, error, reload, updatedAt } = state;
  const done = data?.state === "synced" || data?.state === "rejected";
  useEffect(() => {
    if (done || loading || refreshing || error?.status === 404) return;
    const poll = () => { if (document.visibilityState === "visible") reload(); };
    const timer = setTimeout(poll, (error?.retryAfterS ?? (error ? 30 : 10)) * 1000);
    document.addEventListener("visibilitychange", poll);
    return () => { clearTimeout(timer); document.removeEventListener("visibilitychange", poll); };
  }, [done, loading, refreshing, error, reload, updatedAt]);
  const label = !data ? error?.status === 404 ? "Confirmation not found" : error ? "Unable to check saved confirmation" : "Checking saved confirmation…"
    : data.state === "rejected" ? "Order rejected"
    : data.state === "synced" && data.filled_qty === 0 ? `Order ${data.status}; no shares filled`
    : data.state === "synced" && data.status !== "filled" ? `Order ${data.status} · ${data.filled_qty} of ${data.qty} shares filled · Journal synced`
    : data.journal_synced && data.state === "synced" ? "Filled · Journal synced"
    : data.status === "filled" ? "Filled · Journal sync pending"
    : data.filled_qty > 0 ? `Partially filled · ${data.filled_qty} of ${data.qty} shares`
    : data.order_id ? `Accepted · ${data.status} · Awaiting fill`
    : "Submission pending · Confirmation saved";
  return <section className="notice" aria-label="Latest paper order">
    <p className="strong" role="status">{label}</p>
    {data && <p>{data.ticker} · {data.qty} shares · <Link href="/journal">Open Journal</Link></p>}
    {data?.broker_available === false && <p>The broker could not confirm an update. Showing the saved status; do not place a replacement order.</p>}
    {state.error?.status === 404
      ? <p>No saved confirmation was found. If submission was interrupted, review the same ticker and quantity to retry with the original confirmation.</p>
      : !data && <ErrorLine error={state.error} />}
    {data && <RefreshStatus updatedAt={state.updatedAt} refreshing={state.refreshing} error={state.error} onRetry={state.reload} sourceNote="The saved order status may be older." />}
    <button type="button" className="btn btn-sm" onClick={state.reload} disabled={state.loading || state.refreshing}>Check order status</button>
  </section>;
}
