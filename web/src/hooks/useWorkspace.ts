"use client";

import { useSyncExternalStore } from "react";

import { api, unwrap } from "@/api/client";
import type { Schemas } from "@/api/client";
import { useApi } from "@/hooks/useApi";

let revision = 0;
const listeners = new Set<() => void>();
const subscribe = (f: () => void) => { listeners.add(f); return () => { listeners.delete(f); }; };
const snapshot = () => revision;

/** `enabled: false` skips the request (a parent that already reads the preferences passes them down). */
export function useWorkspace(enabled = true) {
  const version = useSyncExternalStore(subscribe, snapshot, () => 0);
  return useApi(enabled ? "research-preferences" : null, (signal) => unwrap(api.GET("/v1/me/research-preferences", { signal })), version);
}

export async function saveWorkspace(body: Schemas["WorkspacePatch"]) {
  const data = await unwrap(api.PATCH("/v1/me/research-preferences", { body }));
  revision += 1;
  listeners.forEach((f) => f());
  return data;
}
