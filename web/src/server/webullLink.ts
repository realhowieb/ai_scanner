// "Open in Webull": the web server asks the API for the stock's Webull page (Webull's
// quote URLs need the listing exchange, which the API looks up) and redirects there.
// Signed out, no cookies or tokens sent either way.
import type { Upstream } from "./bff";

export const WEBULL_FALLBACK = "https://www.webull.com/quote/us";
const TICKER = /^[A-Za-z][A-Za-z0-9]{0,5}([.-][A-Za-z]{1,2})?$/;

/** Only ever redirect to webull.com, whatever the API answers. */
function safeWebull(url: unknown): string | null {
  if (typeof url !== "string") return null;
  try {
    const u = new URL(url);
    return u.protocol === "https:" && u.hostname === "www.webull.com" ? u.toString() : null;
  } catch {
    return null;
  }
}

export async function webullTarget(raw: string, upstream: Upstream): Promise<string> {
  const ticker = decodeURIComponent(raw || "").trim().toUpperCase();
  if (!TICKER.test(ticker)) return WEBULL_FALLBACK;
  try {
    const res = await upstream(`/v1/public/webull-link/${encodeURIComponent(ticker)}`, { method: "GET", headers: { accept: "application/json" } });
    if (!res.ok) return WEBULL_FALLBACK;
    const body = (await res.json()) as { url?: unknown };
    return safeWebull(body.url) ?? WEBULL_FALLBACK;
  } catch {
    return WEBULL_FALLBACK;
  }
}
