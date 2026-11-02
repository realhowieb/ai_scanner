"use client";
// On www.hsfinest.com, links into the app (Sign in, Start free, Open HSF) go straight to
// app.hsfinest.com with the visitor's utm tags and referrer, instead of bouncing through a
// redirect that would drop them. Does nothing on any other host.
import { useEffect } from "react";

import { handoffParams } from "@/lib/funnel";
import { APP_ORIGIN, isMarketingHost, marketingRoute } from "@/lib/siteHosts";

/** The app address for a same-site link on the marketing host, or null to leave the link alone. */
export function appHandoffUrl(href: string, here: string, params: URLSearchParams): string | null {
  let url: URL;
  try {
    url = new URL(href, here);
  } catch {
    return null;
  }
  if (url.origin !== new URL(here).origin || marketingRoute(url.pathname) !== "app") return null;
  const to = new URL(url.pathname + url.search + url.hash, APP_ORIGIN);
  params.forEach((v, k) => {
    if (!to.searchParams.has(k)) to.searchParams.set(k, v);
  });
  return to.toString();
}

export function MarketingHandoff() {
  useEffect(() => {
    if (!isMarketingHost(window.location.host)) return;
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = (e.target as Element | null)?.closest?.("a[href]") as HTMLAnchorElement | null;
      if (!a || (a.target && a.target !== "_self") || a.hasAttribute("download")) return;
      const to = appHandoffUrl(a.getAttribute("href") || "", window.location.href, handoffParams());
      if (!to) return;
      // Stops the in-page router; the link's own click handler (funnel tracking) still runs.
      e.preventDefault();
      window.location.assign(to);
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, []);
  return null;
}
