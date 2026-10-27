"use client";

import { useEffect, useState } from "react";

import { api } from "@/api/client";

const RECHECK_MS = 5 * 60 * 1000;
const MIN_GAP_MS = 60 * 1000;
let lastAsked = 0;

/** How many alerts fired since the Alerts page was last opened. Opening Alerts marks them
 * seen; elsewhere the count is asked on navigation (at most once a minute) and every 5 minutes. */
export function useUnreadAlerts(path: string, signedIn: boolean): number {
  const [count, setCount] = useState(0);
  const onAlerts = path === "/alerts" || path.startsWith("/alerts/");

  useEffect(() => {
    if (!signedIn) return;
    let live = true;
    if (onAlerts) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- opening Alerts clears the badge at once
      setCount(0);
      lastAsked = 0;
      void api.POST("/v1/me/alerts/seen").catch(() => undefined);
      return () => { live = false; };
    }
    const ask = (force: boolean) => {
      if (!force && Date.now() - lastAsked < MIN_GAP_MS) return;
      lastAsked = Date.now();
      void api.GET("/v1/me/alerts/unread").then((r) => {
        if (live && r.data && typeof r.data.count === "number") setCount(r.data.count);
      }).catch(() => undefined);
    };
    ask(false);
    const timer = window.setInterval(() => { if (document.visibilityState === "visible") ask(true); }, RECHECK_MS);
    return () => { live = false; window.clearInterval(timer); };
  }, [path, signedIn, onAlerts]);

  return onAlerts ? 0 : count;
}

export function AlertsBadge({ count }: { count: number }) {
  if (count <= 0) return null;
  return <><span className="nav-badge" aria-hidden="true">{count > 9 ? "9+" : count}</span><span className="sr-only">, {count} new</span></>;
}
