"use client";

import { useEffect } from "react";

import { isStandalone } from "@/lib/homeScreen";

/** In the Home Screen app, registers /sw.js so a page opened without signal shows the
 * offline screen instead of a blank one. Browser tabs register it only for alerts. */
export function OfflineReady() {
  useEffect(() => {
    if (!isStandalone() || !("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(() => undefined);
  }, []);
  return null;
}
