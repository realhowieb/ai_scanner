// The policy texts (GET /v1/founding/policies/{type}), fetched on the web server for /terms,
// /privacy, /founding-100/terms and /founding-100/refunds. Until the pages are switched on
// (HSF_LEGAL_PAGES_ENABLED for /terms and /privacy, FOUNDING_LEGAL_PAGES_ENABLED for the
// founding ones, which also opens /privacy) the public call is 404; a signed-in admin
// then gets the same text from the admin preview, using their own session cookie.
import "server-only";

import { cookies } from "next/headers";

import type { Schemas } from "@/api/client";

import { ACCESS_COOKIE, accessTokenUsable } from "./cookies";
import { upstream } from "./upstream";

export type Policy = Schemas["Policy"];
export type PolicyType = Policy["policy_type"];
export type LoadedPolicy = { policy: Policy; preview: boolean };

async function getJson<T>(path: string, headers: Record<string, string>): Promise<T | null> {
  try {
    const res = await upstream(path, { method: "GET", headers: { accept: "application/json", ...headers } });
    return res.ok ? ((await res.json()) as T) : null;
  } catch {
    return null;
  }
}

/** null when the page isn't published (and the visitor isn't an admin) or the API is down. */
export async function loadPolicy(type: PolicyType, version?: number): Promise<LoadedPolicy | null> {
  const q = version ? `?version=${version}` : "";
  const open = await getJson<Policy>(`/v1/founding/policies/${type}${q}`, {});
  if (open) return { policy: open, preview: false };
  let token: string | undefined;
  try {
    token = (await cookies()).get(ACCESS_COOKIE)?.value;
  } catch {
    return null; // outside a request (build)
  }
  if (!token || !accessTokenUsable(token)) return null;
  const admin = await getJson<Policy>(`/v1/admin/founding/policies/${type}${q}`, { authorization: `Bearer ${token}` });
  return admin ? { policy: admin, preview: true } : null;
}

/** ?version=N from the page URL, when it is a plain positive number. */
export function versionParam(raw: string | string[] | undefined): number | undefined {
  const v = Array.isArray(raw) ? raw[0] : raw;
  return v && /^[1-9]\d{0,3}$/.test(v) ? Number(v) : undefined;
}
