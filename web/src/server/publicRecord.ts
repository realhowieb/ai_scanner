// The public track record, fetched on the web server for /results.
import "server-only";

import { headers } from "next/headers";

import type { Schemas } from "@/api/client";

import { withClientAddress } from "./clientAddress";
import { upstream } from "./upstream";

export type PublicRecord = Schemas["PublicRecord"];

/** null when the API can't answer; the page then says so instead of showing numbers. */
export async function loadPublicRecord(): Promise<PublicRecord | null> {
  try {
    let call = upstream;
    try {
      call = withClientAddress(new Request("http://results.local", { headers: await headers() }), upstream);
    } catch {
      /* outside a request (build) */
    }
    const res = await call("/v1/public/track-record", { method: "GET", headers: { accept: "application/json" } });
    return res.ok ? ((await res.json()) as PublicRecord) : null;
  } catch {
    return null;
  }
}
