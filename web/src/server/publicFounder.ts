// The Founders Wall and a shared founder card, fetched on the web server for the public
// /founders pages and the card's preview image.
import "server-only";

import { headers } from "next/headers";

import type { Schemas } from "@/api/client";

import { withClientAddress } from "./clientAddress";
import { upstream } from "./upstream";

export type FounderCard = Schemas["PublicFounderCard"];
export type FoundersWall = Schemas["FoundersWall"];
export const CARD_CODE_RE = /^[a-z0-9]{4,32}$/;

async function caller() {
  try {
    return withClientAddress(new Request("http://founders.local", { headers: await headers() }), upstream);
  } catch {
    return upstream; // outside a request (build)
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

/** null for an unknown code, a member who hides the badge, or when the API can't answer. */
export async function loadFounderCard(raw: string): Promise<FounderCard | null> {
  const code = decodeURIComponent(raw || "").trim().toLowerCase();
  if (!CARD_CODE_RE.test(code)) return null;
  return getJson<FounderCard>(`/v1/public/founder-card/${code}`);
}

export async function loadFoundersWall(): Promise<FoundersWall | null> {
  const wall = await getJson<FoundersWall>("/v1/public/founding/wall");
  return wall?.open ? wall : null;
}
