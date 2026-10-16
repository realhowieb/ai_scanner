"use client";

// Session restoration: /v1/me through the BFF (which refreshes the cookie session if
// the access token expired). Plan labels and feature availability come only from
// the server's answer; the client never decides what a plan includes.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import { account } from "@/api/account";
import type { ApiError, Schemas } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { disablePush } from "@/lib/webPush";
import { BILLING_STARTED, beginBilling, finishBilling, pendingBilling } from "@/lib/billingReturn";

import { clearCachedMe, readCachedMe, subscribeCachedMe, writeCachedMe } from "./meCache";

export type Me = Schemas["Me"];
export type Feature = keyof typeof FEATURE_PLAN;

/** Which plan to suggest when a feature is locked (for the upgrade button only). */
export const FEATURE_PLAN = {
  can_premarket: "pro",
  can_afterhours: "pro",
  can_scan_nasdaq: "pro",
  can_unusual_volume: "pro",
  can_scan_history: "pro",
  can_day_trader: "pro",
  can_early_breakout: "premium",
  can_full_universe: "premium",
  can_ai_notes: "premium",
} as const;

type Ctx = {
  me: Me | null;
  loading: boolean;
  error: ApiError | null;
  can: (feature: string) => boolean;
  signOut: () => Promise<void>;
  reload: () => void;
  syncBilling: () => Promise<void>;
  billingBusy: boolean;
  billingStatus: string | null;
  billingError: ApiError | null;
};

const SessionContext = createContext<Ctx | null>(null);

export async function signOut(): Promise<void> {
  finishBilling();
  clearCachedMe();
  // A signed-out browser must stop getting this account's alerts; never hold sign-out up for it.
  await Promise.race([disablePush().catch(() => undefined), new Promise((r) => setTimeout(r, 1500))]);
  try {
    await fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" });
  } finally {
    // A full page load on purpose: it drops every piece of client state from the old session.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination
    window.location.assign("/login");
  }
}

export function SessionProvider({ children, initialMe = null }: { children: ReactNode; initialMe?: Me | null }) {
  const [revalidate, setRevalidate] = useState(false);
  const state = useApi<Me>(initialMe && !revalidate ? null : "me", (signal) => unwrap(api.GET("/v1/me", { signal })));
  const reloadMe = state.reload;
  const reload = useCallback(() => { setRevalidate(true); reloadMe(); }, [reloadMe]);
  const [billingBusy, setBillingBusy] = useState(false);
  const [billingStatus, setBillingStatus] = useState<string | null>(null);
  const [billingError, setBillingError] = useState<ApiError | null>(null);
  const syncing = useRef<Promise<void> | null>(null);
  const syncBilling = useCallback((): Promise<void> => {
    if (syncing.current) return syncing.current;
    setBillingBusy(true);
    setBillingError(null);
    const request = account.syncBilling().then((result) => {
      setBillingStatus(result.message);
      reload();
    }).catch((error: unknown) => {
      setBillingStatus(null);
      setBillingError(error as ApiError);
      // Still refresh the server's current plan if Stripe is unavailable.
      reload();
    }).finally(() => { syncing.current = null; setBillingBusy(false); });
    syncing.current = request;
    return request;
  }, [reload]);
  // This tab's last answer lets the page render (and fetch its data) while /v1/me revalidates.
  const cached = useSyncExternalStore(subscribeCachedMe, readCachedMe, () => null);
  useEffect(() => {
    if (state.data) writeCachedMe(state.data);
    else if (state.error?.status === 401) clearCachedMe();
  }, [state.data, state.error]);
  const me = state.data ?? initialMe ?? cached;
  useEffect(() => {
    const pending = pendingBilling();
    if (state.data && pending && (pending.expected ? state.data.plan === pending.expected
      : pending.previous !== null && state.data.plan !== pending.previous)) {
      finishBilling();
    }
  }, [state.data]);
  useEffect(() => {
    // Safari may open Stripe outside an installed Mac web app. Refresh the
    // original app on focus as well as on the return page; never depend on opener.
    let timer: ReturnType<typeof setTimeout> | undefined;
    let disposed = false;
    let cycle = 0;
    const check = async (attempt: number, currentCycle: number) => {
      if (disposed || currentCycle !== cycle || !pendingBilling() || document.visibilityState === "hidden") return;
      await syncBilling();
      if (!disposed && currentCycle === cycle && attempt < 2) {
        timer = setTimeout(() => void check(attempt + 1, currentCycle), 5000);
      }
    };
    const resume = () => { clearTimeout(timer); void check(0, ++cycle); };
    window.addEventListener("focus", resume);
    window.addEventListener(BILLING_STARTED, resume);
    document.addEventListener("visibilitychange", resume);
    if (new URLSearchParams(window.location.search).get("billing") === "return" && !pendingBilling()) {
      beginBilling(null);
    } else resume();
    return () => {
      disposed = true;
      clearTimeout(timer);
      window.removeEventListener("focus", resume);
      window.removeEventListener(BILLING_STARTED, resume);
      document.removeEventListener("visibilitychange", resume);
    };
  }, [syncBilling]);
  const can = useCallback((feature: string) => !!me?.entitlements?.[feature], [me]);
  const value = useMemo<Ctx>(() => ({ me, loading: !me && state.loading, error: state.error, can, signOut, reload,
    syncBilling, billingBusy, billingStatus, billingError }),
    [me, state.loading, state.error, can, reload, syncBilling, billingBusy, billingStatus, billingError]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): Ctx {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession outside SessionProvider");
  return ctx;
}
