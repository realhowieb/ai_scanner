// Page guard (Next.js 16 "proxy", formerly middleware). It only checks that a
// session cookie exists; the BFF does the real work (refresh, expiry) on API calls.
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { REFRESH_COOKIE } from "@/server/cookies";
import { APP_ORIGIN, MUSIC_ORIGIN, isMarketingHost, marketingRoute } from "@/lib/siteHosts";

/** www.hsfinest.com: public pages are served here; the app and old music pages live elsewhere. */
function marketing(req: NextRequest): NextResponse {
  const { pathname, search } = req.nextUrl;
  const route = marketingRoute(pathname);
  if (route === "serve") return NextResponse.next();
  if (route === "music") return NextResponse.redirect(new URL(pathname + search, MUSIC_ORIGIN), 301);
  return NextResponse.redirect(new URL(pathname + search, APP_ORIGIN), 307);
}

// The app's own pages this guard looks at (the plain matcher entries below). The marketing
// matcher can also reach other hosts, so anything else passes straight through.
const APP_SECTIONS = ["/share", "/today", "/scanner", "/stocks", "/watchlists", "/alerts", "/account", "/track-record",
  "/brief", "/journal", "/ask", "/day-trader", "/paper", "/admin"];
function guarded(pathname: string): boolean {
  return ["/", "/login", "/signup", "/signed-out"].includes(pathname)
    || APP_SECTIONS.some((p) => pathname === p || pathname.startsWith(p + "/"));
}

export function proxy(req: NextRequest) {
  if (isMarketingHost(req.headers.get("host"))) return marketing(req);
  if (!guarded(req.nextUrl.pathname)) return NextResponse.next();
  const signedIn = !!req.cookies.get(REFRESH_COOKIE)?.value;
  const { pathname, search } = req.nextUrl;
  if (pathname === "/login" || pathname === "/signup" || pathname === "/signed-out") {
    return signedIn ? NextResponse.redirect(new URL("/today", req.url)) : NextResponse.next();
  }
  if (pathname === "/") {
    // Signed out: the landing page. Its utm tags are read by the page, so keep the query.
    return signedIn ? NextResponse.redirect(new URL("/today", req.url)) : NextResponse.next();
  }
  if (pathname.startsWith("/share/")) {
    // A shared stock link: members go straight to the full stock page.
    const [ticker = "", ...rest] = pathname.slice("/share/".length).split("/");
    return signedIn && ticker && rest.length === 0 ? NextResponse.redirect(new URL(`/stocks/${ticker}`, req.url)) : NextResponse.next();
  }
  if (!signedIn) {
    const to = new URL("/login", req.url);
    to.searchParams.set("next", pathname + search);
    return NextResponse.redirect(to);
  }
  return NextResponse.next();
}

export const config = {
  matcher: [
    // Every page on the marketing host (www.hsfinest.com), apart from build files.
    // Kept literal: Next reads this config at build time.
    { source: "/((?!_next/static|_next/image).*)", has: [{ type: "host", value: "(www\\.)?hsfinest\\.com" }] },
    "/", "/login", "/signup", "/signed-out", "/share/:path*", "/today/:path*", "/scanner/:path*", "/stocks/:path*", "/watchlists/:path*", "/alerts/:path*", "/account/:path*", "/track-record/:path*", "/brief/:path*", "/journal/:path*", "/ask/:path*", "/day-trader/:path*", "/paper/:path*", "/admin/:path*"],
};
