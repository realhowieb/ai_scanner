// Page guard (Next.js 16 "proxy", formerly middleware). It only checks that a
// session cookie exists; the BFF does the real work (refresh, expiry) on API calls.
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { REFRESH_COOKIE } from "@/server/cookies";

export function proxy(req: NextRequest) {
  const signedIn = !!req.cookies.get(REFRESH_COOKIE)?.value;
  const { pathname, search } = req.nextUrl;
  if (pathname === "/login") {
    return signedIn ? NextResponse.redirect(new URL("/today", req.url)) : NextResponse.next();
  }
  if (!signedIn) {
    const to = new URL("/login", req.url);
    if (pathname !== "/") to.searchParams.set("next", pathname + search);
    return NextResponse.redirect(to);
  }
  if (pathname === "/") return NextResponse.redirect(new URL("/today", req.url));
  return NextResponse.next();
}

export const config = {
  matcher: ["/", "/login", "/today/:path*", "/scanner/:path*", "/stocks/:path*", "/watchlists/:path*", "/alerts/:path*", "/account/:path*", "/track-record/:path*", "/brief/:path*"],
};
