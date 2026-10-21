"use client";

import type { CaptureResult, PostHog } from "posthog-js";
import { posthogHost } from "./posthogConfig";

export const USAGE_PREFERENCE_EVENT = "hsf-usage-preference";
const PREFERENCE = "hsf-usage-disabled";
const EVENTS = new Set(["page_viewed", "scanner_viewed", "stock_viewed", "watchlist_created", "watchlist_stock_added", "alert_created", "signup_completed"]);
export type UsageEvent = "page_viewed" | "scanner_viewed" | "stock_viewed" | "watchlist_created" | "watchlist_stock_added" | "alert_created" | "signup_completed";
type UsageProperties = { route?: string; kind?: "price" | "technical" | "hsf" | "scan" };
const ROUTES = new Set(["/", "/demo", "/methodology", "/pricing", "/signup", "/today", "/brief", "/scanner", "/scanner/custom", "/scanner/history", "/compare", "/watchlists", "/alerts", "/track-record", "/day-trader", "/journal", "/paper-trading", "/account"]);
let sdk: PostHog | null = null;
let loading: Promise<PostHog | null> | null = null;
let identityGeneration = 0;

export function analyticsConfigured(): boolean {
  return !!process.env.NEXT_PUBLIC_POSTHOG_KEY?.trim();
}

/** Route templates only: no tickers, account identifiers, query strings or tokens. */
export function usageRoute(pathname: string): string | null {
  const path = pathname.split(/[?#]/, 1)[0]!;
  if (/^\/stocks\/[^/]+$/.test(path)) return "/stocks/:ticker";
  return ROUTES.has(path) ? path : null;
}

export function usageAllowed(): boolean {
  if (typeof window === "undefined" || !analyticsConfigured()) return false;
  if (navigator.doNotTrack === "1" || (navigator as Navigator & { globalPrivacyControl?: boolean }).globalPrivacyControl) return false;
  try { return localStorage.getItem(PREFERENCE) !== "1"; } catch { return false; }
}

/** Final outbound allowlist also strips SDK-added URLs, referrers and person properties. */
export function sanitizeUsage(event: CaptureResult | null): CaptureResult | null {
  if (!event || !EVENTS.has(event.event)) return null;
  const safe: Record<string, unknown> = {};
  for (const key of ["distinct_id", "$device_id", "$session_id", "$window_id", "$insert_id", "$lib", "$lib_version", "token"]) {
    const value = event.properties[key];
    if (typeof value === "string") safe[key] = value;
  }
  const route = event.properties.route;
  if (typeof route === "string") {
    const normalized = usageRoute(route);
    if (normalized) safe.route = normalized;
  }
  const kind = event.properties.kind;
  if (typeof kind === "string" && ["price", "technical", "hsf", "scan"].includes(kind)) safe.kind = kind;
  safe.$process_person_profile = false;
  return { uuid: event.uuid, event: event.event, properties: safe, ...(event.timestamp ? { timestamp: event.timestamp } : {}) };
}

async function client(): Promise<PostHog | null> {
  if (!usageAllowed()) return null;
  if (sdk) return sdk;
  const initGeneration = identityGeneration;
  if (!loading) loading = import("posthog-js").then(({ default: posthog }) => {
    if (!usageAllowed()) return null;
    posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY!, {
      api_host: posthogHost(process.env.NEXT_PUBLIC_POSTHOG_HOST),
      persistence: "localStorage", cross_subdomain_cookie: false,
      autocapture: false, capture_pageview: false, capture_pageleave: false,
      capture_dead_clicks: false, capture_heatmaps: false, capture_performance: false,
      capture_exceptions: false, disable_session_recording: true,
      disable_surveys: true, disable_product_tours: true, disable_web_experiments: true,
      disable_external_dependency_loading: true, advanced_disable_flags: true,
      person_profiles: "never", respect_dnt: true,
      save_referrer: false, save_campaign_params: false,
      before_send: (event) => usageAllowed() ? sanitizeUsage(event) : null,
    });
    if (initGeneration !== identityGeneration) posthog.reset(true);
    sdk = posthog;
    return sdk;
  }).catch(() => null).finally(() => { loading = null; });
  return loading;
}

/** Best effort: analytics must never delay or reject a successful product action. */
export function captureUsage(event: UsageEvent, properties: UsageProperties = {}): void {
  if (!usageAllowed()) return;
  const generation = identityGeneration;
  void client().then((ph) => {
    if (ph && usageAllowed() && generation === identityGeneration) ph.capture(event, properties, event === "signup_completed" ? { send_instantly: true } : undefined);
  }).catch(() => undefined);
}

export function setUsageEnabled(enabled: boolean): void {
  try {
    localStorage.setItem(PREFERENCE, enabled ? "0" : "1");
    if (!enabled) { identityGeneration++; sdk?.opt_out_capturing(); sdk?.reset(true); }
    else sdk?.opt_in_capturing({ captureEventName: false });
    window.dispatchEvent(new Event(USAGE_PREFERENCE_EVENT));
  } catch { /* Blocked storage or analytics never blocks the app. */ }
}

/** Called before sign-out, so an anonymous browser identity isn't shared across accounts. */
export function resetUsageIdentity(): void {
  identityGeneration++;
  try { sdk?.reset(true); } catch { /* Best effort. */ }
}
