// Tells public pages (pricing, how it works, legal) whether this browser is signed in,
// since the session cookies are HttpOnly. Only checks the cookie exists, like the page guard.
import type { NextRequest } from "next/server";

import { REFRESH_COOKIE } from "@/server/cookies";

export const dynamic = "force-dynamic";

export function GET(req: NextRequest): Response {
  const signedIn = !!req.cookies.get(REFRESH_COOKIE)?.value;
  return new Response(JSON.stringify({ signed_in: signedIn }), { headers: { "content-type": "application/json", "cache-control": "no-store" } });
}
