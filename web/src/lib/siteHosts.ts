// www.hsfinest.com (and the bare domain) is the marketing front door: it serves the public
// pages from this same app, and sends everything else to the app at app.hsfinest.com so
// sign-in lives on one host. The old music site moved to music.hsfinest.com.

export const APP_ORIGIN = "https://app.hsfinest.com";
export const MUSIC_ORIGIN = "https://music.hsfinest.com";
export const MARKETING_HOSTS = ["www.hsfinest.com", "hsfinest.com"] as const;

export function isMarketingHost(host: string | null | undefined): boolean {
  const h = (host || "").toLowerCase().split(":")[0] ?? "";
  return (MARKETING_HOSTS as readonly string[]).includes(h);
}

// Public pages and the files and public API calls they need.
const SERVE_EXACT = new Set(["/", "/pricing", "/how-hsf-works", "/demo", "/terms", "/privacy", "/robots.txt",
  "/sitemap.xml", "/manifest.webmanifest", "/sw.js", "/offline.html", "/icon.svg", "/apple-icon.png", "/api/healthz",
  // Public pages ask this to pick their signed-in view; sent to the app host, the browser blocks it (CORS).
  "/api/auth/status"]);
const SERVE_PREFIX = ["/weekly", "/founders", "/founding-100", "/share/", "/social/", "/subscribe/", "/unsubscribe",
  "/api/public/", "/api/logo/", "/landing/", "/licenses/"];
// Generated files whose names Next may suffix (opengraph-image-abc123, icon-192.png).
const SERVE_START = ["/opengraph-image", "/icon-", "/_next/"];

// Pages of the old Hostinger music site, so old links and search results still land.
const MUSIC_PREFIX = ["/shop", "/blog-hsf", "/music", "/events", "/services"];

function under(pathname: string, prefix: string): boolean {
  if (prefix.endsWith("/")) return pathname.startsWith(prefix);
  return pathname === prefix || pathname.startsWith(prefix + "/");
}

export type MarketingRoute = "serve" | "music" | "app";

/** What the marketing host does with a path: serve it here, or send it to the music site or the app. */
export function marketingRoute(pathname: string): MarketingRoute {
  if (SERVE_EXACT.has(pathname) || SERVE_PREFIX.some((p) => under(pathname, p))
    || SERVE_START.some((p) => pathname.startsWith(p))) return "serve";
  if (MUSIC_PREFIX.some((p) => under(pathname, p))) return "music";
  return "app";
}
