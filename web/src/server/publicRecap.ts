// The public weekly recap, fetched on the web server for /weekly and /weekly/<week>.
import "server-only";

import { headers } from "next/headers";

import type { Schemas } from "@/api/client";

import { withClientAddress } from "./clientAddress";
import { upstream } from "./upstream";

export type RecapIndex = Schemas["PublicRecapIndex"];
export type Recap = Schemas["PublicRecap"];
export const WEEK_RE = /^\d{4}-W\d{2}$/;

async function caller() {
  try {
    return withClientAddress(new Request("http://weekly.local", { headers: await headers() }), upstream);
  } catch {
    return upstream; // outside a request (build, sitemap)
  }
}

async function getJson<T>(path: string): Promise<T | null> {
  try {
    const res = await (await caller())(path, { method: "GET", headers: { accept: "application/json" } });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function loadRecapIndex(): Promise<RecapIndex> {
  return (await getJson<RecapIndex>("/v1/public/weekly")) ?? { latest: null, weeks: [] };
}

/** null for a malformed or unpublished week, or when the API can't answer. */
export async function loadRecap(raw: string): Promise<Recap | null> {
  const week = decodeURIComponent(raw || "").trim().toUpperCase();
  if (!WEEK_RE.test(week)) return null;
  return getJson<Recap>(`/v1/public/weekly/${week}`);
}
