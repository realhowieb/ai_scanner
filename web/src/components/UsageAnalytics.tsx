"use client";

import { useEffect, useRef, useSyncExternalStore } from "react";
import { usePathname } from "next/navigation";
import { Card } from "@/components/ui";
import { analyticsConfigured, captureUsage, setUsageEnabled, usageAllowed, usageRoute, USAGE_PREFERENCE_EVENT } from "@/lib/usageAnalytics";

function subscribe(callback: () => void) {
  window.addEventListener(USAGE_PREFERENCE_EVENT, callback);
  window.addEventListener("storage", callback);
  return () => { window.removeEventListener(USAGE_PREFERENCE_EVENT, callback); window.removeEventListener("storage", callback); };
}

export function UsageAnalytics() {
  const path = usePathname();
  const previous = useRef<string | null>(null);
  const allowed = useSyncExternalStore(subscribe, usageAllowed, () => false);
  useEffect(() => {
    if (!allowed) { previous.current = null; return; }
    if (previous.current === path) return;
    previous.current = path;
    const route = usageRoute(path);
    if (!route) return;
    captureUsage("page_viewed", { route });
    if (route === "/scanner") captureUsage("scanner_viewed", { route });
    if (route === "/stocks/:ticker") captureUsage("stock_viewed", { route });
  }, [path, allowed]);
  return null;
}

export function UsageAnalyticsPreference() {
  const allowed = useSyncExternalStore(subscribe, usageAllowed, () => false);
  if (!analyticsConfigured()) return null;
  return <Card title="Usage privacy" id="usage-privacy">
    <label className="check"><input type="checkbox" checked={allowed} onChange={(e) => setUsageEnabled(e.target.checked)} /><span>Share anonymous feature usage</span></label>
    <p className="cap">Helps us see which HSF features people use and return to. We send page categories and successful watchlist and alert actions to PostHog, without ticker symbols, watchlist names, notes, account details or form contents. Session recording is off.</p>
    <p className="cap">This setting applies to this browser. Do Not Track and Global Privacy Control take priority.</p>
  </Card>;
}
