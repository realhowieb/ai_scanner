"use client";

// "Set a price alert" for one ticker, prefilled at a given price: shows the plan's alert
// quota, then the alert form locked to the price type. Used by the stock page and Day Trader.
import Link from "next/link";
import { useState } from "react";

import { alerts } from "@/api/userData";
import { Dialog } from "@/components/Dialog";
import { ErrorLine } from "@/components/ui";
import { useApi } from "@/hooks/useApi";

import { AlertForm, TYPES_UNAVAILABLE, describeAlert, typesUnavailable } from "./AlertForm";

export function PriceAlertButton({ ticker, at, note, label, compact = false, onCreated }: {
  ticker: string; at: number | null | undefined; note?: string; label: string; compact?: boolean; onCreated?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const types = useApi(open ? "alert-types" : null, (signal) => alerts.types(signal));
  const quota = useApi(open ? `alert-quota:${ticker}` : null, (signal) => alerts.list(signal));
  const full = !!quota.data && quota.data.used >= quota.data.limit;
  const close = () => { setOpen(false); setDone(null); };
  return (
    <>
      {compact ? (
        <button type="button" className="icon-btn" aria-label={label} title={label} onClick={() => setOpen(true)}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /></svg>
        </button>
      ) : (
        <button type="button" className="btn" onClick={() => setOpen(true)}>{label}</button>
      )}
      <Dialog open={open} title={`Price alert for ${ticker}`} onClose={close}>
        {done ? (
          <div className="stack-sm">
            <p className="notice" role="status">Created: {done}.</p>
            <div className="row-actions"><button type="button" className="btn" onClick={close} data-autofocus>Done</button><Link href="/alerts">See all alerts →</Link></div>
          </div>
        ) : types.error && typesUnavailable(types.error.status) ? <p className="notice" role="note">{TYPES_UNAVAILABLE}</p>
          : types.error || quota.error ? <ErrorLine error={types.error ?? quota.error} /> : !types.data || !quota.data ? <p className="cap">Loading…</p> : full ? (
          <div className="stack-sm">
            <p className="strong">You&apos;re using all {quota.data.limit} alert{quota.data.limit === 1 ? "" : "s"} on your plan.</p>
            <p className="cap">Delete one on the <Link href="/alerts">Alerts</Link> page, or upgrade for more.</p>
          </div>
        ) : (
          <>
            <p className="cap">Using {quota.data.used} of {quota.data.limit} alerts.{note ? ` ${note}` : ""}</p>
            <AlertForm types={types.data} lockType prefill={{ type: "price", ticker, threshold: at != null ? Math.round(at * 100) / 100 : undefined, direction: "above" }}
              onCreated={(a) => { setDone(describeAlert(a)); onCreated?.(); }} />
          </>
        )}
      </Dialog>
    </>
  );
}
