"use client";

import { useId, useSyncExternalStore } from "react";

export type BillingInterval = "month" | "year";
const KEY = "hsf_billing_interval";
const CHANGED = "hsf-billing-interval-changed";
let memory: BillingInterval = "month";

function read(): BillingInterval {
  try { return sessionStorage.getItem(KEY) === "year" ? "year" : "month"; }
  catch { return memory; }
}

function subscribe(listener: () => void) {
  window.addEventListener(CHANGED, listener);
  return () => window.removeEventListener(CHANGED, listener);
}

export function useBillingInterval(): [BillingInterval, (value: BillingInterval) => void] {
  const interval = useSyncExternalStore(subscribe, read, () => "month" as const);
  const set = (value: BillingInterval) => {
    memory = value;
    try { sessionStorage.setItem(KEY, value); } catch { /* storage is optional */ }
    window.dispatchEvent(new Event(CHANGED));
  };
  return [interval, set];
}

export function BillingIntervalToggle() {
  const [interval, setInterval] = useBillingInterval();
  const name = useId();
  return <div className="row-actions">
    <div className="seg-group" role="radiogroup" aria-label="Billing interval">
      {(["month", "year"] as const).map((value) => <label key={value} className={`seg${interval === value ? " on" : ""}`}>
        <input type="radio" className="sr-only" name={name} value={value} checked={interval === value}
          onChange={() => setInterval(value)} />
        {value === "month" ? "Monthly" : "Yearly"}
      </label>)}
    </div>
    <span className="cap">{interval === "year" ? "Billed once a year" : "Billed monthly"}</span>
  </div>;
}
