"use client";

import { useEffect, useRef } from "react";

export const AUTO_REFRESH_MS = 5 * 60_000;

/** True on a weekday between 9:25 and 16:05 New York time (scans land during the session). */
export function inMarketHours(now: Date = new Date()): boolean {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(now);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  if (get("weekday") === "Sat" || get("weekday") === "Sun") return false;
  const minutes = Number(get("hour")) * 60 + Number(get("minute"));
  return minutes >= 9 * 60 + 25 && minutes <= 16 * 60 + 5;
}

/** Calls `reload` every `everyMs` while the page is open and visible during market hours,
 * and once on coming back to the tab if a refresh was due meanwhile. */
export function useAutoRefresh(reload: () => void, { everyMs = AUTO_REFRESH_MS, enabled = true } = {}): void {
  const reloadRef = useRef(reload);
  useEffect(() => {
    reloadRef.current = reload;
  });
  useEffect(() => {
    if (!enabled) return;
    let last = Date.now();
    const due = () => Date.now() - last >= everyMs && inMarketHours() && document.visibilityState === "visible";
    const run = () => {
      if (!due()) return;
      last = Date.now();
      reloadRef.current();
    };
    const t = setInterval(run, Math.min(everyMs, 60_000));
    document.addEventListener("visibilitychange", run);
    return () => {
      clearInterval(t);
      document.removeEventListener("visibilitychange", run);
    };
  }, [everyMs, enabled]);
}
