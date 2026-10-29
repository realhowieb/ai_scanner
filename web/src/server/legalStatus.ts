// Whether the Terms of Service and Privacy Policy are published (GET /v1/legal/status),
// read on the web server so the sign-up form renders the right agreement text at once.
import "server-only";

import type { Schemas } from "@/api/client";

import { upstream } from "./upstream";

export type LegalStatus = Schemas["LegalStatus"];

/** False when the pages are off or the API can't be reached (the plain agreement shows). */
export async function termsPublished(): Promise<boolean> {
  try {
    const res = await upstream("/v1/legal/status", { method: "GET", headers: { accept: "application/json" } });
    return res.ok && ((await res.json()) as LegalStatus).open === true;
  } catch {
    return false;
  }
}
