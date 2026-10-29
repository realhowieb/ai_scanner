// The signed-out stock snapshot behind a shared link (GET /v1/public/stocks/{ticker}),
// fetched on the web server for the /share page and its preview image.
import "server-only";

import { headers } from "next/headers";

import type { Schemas } from "@/api/client";

import { withClientAddress } from "./clientAddress";
import { upstream } from "./upstream";

export type PublicStock = Schemas["PublicStock"];
export const SHARE_TICKER_RE = /^[A-Za-z0-9][A-Za-z0-9.-]{0,9}$/;

/** null for an unknown ticker or when the API can't answer (the page then says so). */
export async function loadPublicStock(raw: string): Promise<PublicStock | null> {
  const ticker = decodeURIComponent(raw || "").trim().toUpperCase();
  if (!SHARE_TICKER_RE.test(ticker)) return null;
  let call = upstream;
  try {
    call = withClientAddress(new Request("http://share.local", { headers: await headers() }), upstream);
  } catch {
    /* outside a request (build): the server's own address */
  }
  try {
    const res = await call(`/v1/public/stocks/${encodeURIComponent(ticker)}`, { method: "GET", headers: { accept: "application/json" } });
    if (!res.ok) return null;
    return (await res.json()) as PublicStock;
  } catch {
    return null;
  }
}

export function siteUrl(): string {
  return (process.env.NEXT_PUBLIC_SITE_URL || "https://app.hsfinest.com").replace(/\/+$/, "");
}

/** "HSF AI Founder #017" when a shared link's invite code belongs to a founder who shows
 * the badge; null otherwise (or when the API can't answer). */
export async function loadFounderBadge(ref: string | null | undefined): Promise<string | null> {
  const code = String(ref || "").trim().toLowerCase();
  if (!/^[a-z0-9]{4,32}$/.test(code)) return null;
  let call = upstream;
  try {
    call = withClientAddress(new Request("http://share.local", { headers: await headers() }), upstream);
  } catch {
    /* outside a request */
  }
  try {
    const res = await call(`/v1/public/founder-badge/${code}`, { method: "GET", headers: { accept: "application/json" } });
    if (!res.ok) return null;
    const body = (await res.json()) as { badge?: string };
    return typeof body.badge === "string" ? body.badge : null;
  } catch {
    return null;
  }
}
