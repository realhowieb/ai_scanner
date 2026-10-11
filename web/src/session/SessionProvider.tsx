"use client";

// Session restoration: /v1/me through the BFF (which refreshes the cookie session if
// the access token expired). Plan labels and feature availability come only from
// the server's answer; the client never decides what a plan includes.
import { createContext, useCallback, useContext, useEffect, useMemo, useSyncExternalStore } from "react";
import type { ReactNode } from "react";

import { api, unwrap } from "@/api/client";
import type { ApiError, Schemas } from "@/api/client";
import { useApi } from "@/hooks/useApi";
import { disablePush } from "@/lib/webPush";

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
};

const SessionContext = createContext<Ctx | null>(null);

export async function signOut(): Promise<void> {
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
  const state = useApi<Me>(initialMe ? null : "me", (signal) => unwrap(api.GET("/v1/me", { signal })));
  // This tab's last answer lets the page render (and fetch its data) while /v1/me revalidates.
  const cached = useSyncExternalStore(subscribeCachedMe, readCachedMe, () => null);
  useEffect(() => {
    if (state.data) writeCachedMe(state.data);
    else if (state.error?.status === 401) clearCachedMe();
  }, [state.data, state.error]);
  const me = initialMe ?? state.data ?? cached;
  const can = useCallback((feature: string) => !!me?.entitlements?.[feature], [me]);
  const value = useMemo<Ctx>(() => ({ me, loading: !me && state.loading, error: state.error, can, signOut, reload: state.reload }),
    [me, state.loading, state.error, can, state.reload]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): Ctx {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession outside SessionProvider");
  return ctx;
}
