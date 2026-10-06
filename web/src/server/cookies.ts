// Session cookies. Both tokens live only in HttpOnly cookies set by this server;
// browser JavaScript can never read them (no localStorage, no readable cookie).
// In production they are also Secure with the __Host- prefix (the browser refuses
// them unless Secure, Path=/ and no Domain). `npm run dev` on plain http drops
// Secure and the prefix, because Safari (and every browser on an IP address) won't
// store Secure cookies over http, which left sign-in stuck on the login page.
export function secureCookies(env: Record<string, string | undefined> = process.env): boolean {
  return env.NODE_ENV === "production" || env.HSF_COOKIE_SECURE === "1";
}

const SECURE = secureCookies();
export const ACCESS_COOKIE = SECURE ? "__Host-hsf_at" : "hsf_at";
export const REFRESH_COOKIE = SECURE ? "__Host-hsf_rt" : "hsf_rt";
export const REFRESH_MAX_AGE_S = 30 * 24 * 3600; // the API's refresh-token lifetime
const EXPIRY_MARGIN_S = 30;

export function parseCookies(header: string | null): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of (header || "").split(";")) {
    const i = part.indexOf("=");
    if (i < 1) continue;
    const name = part.slice(0, i).trim();
    const value = part.slice(i + 1).trim();
    try {
      out[name] = decodeURIComponent(value);
    } catch {
      /* ignore malformed values */
    }
  }
  return out;
}

function serialize(name: string, value: string, maxAge: number): string {
  return `${name}=${encodeURIComponent(value)}; Path=/; Max-Age=${Math.max(0, Math.floor(maxAge))}; HttpOnly;${SECURE ? " Secure;" : ""} SameSite=Lax`;
}

export type TokenPair = { access_token: string; refresh_token: string; expires_in: number };

export function sessionCookies(pair: TokenPair): string[] {
  return [
    serialize(ACCESS_COOKIE, pair.access_token, Math.max(1, pair.expires_in - EXPIRY_MARGIN_S)),
    serialize(REFRESH_COOKIE, pair.refresh_token, REFRESH_MAX_AGE_S),
  ];
}

export function clearedCookies(): string[] {
  return [serialize(ACCESS_COOKIE, "", 0), serialize(REFRESH_COOKIE, "", 0)];
}

/** True when the access token is missing or expires within the margin. The API
 * verifies the signature; this only reads `exp` to avoid a request that would 401. */
export function accessTokenUsable(token: string | undefined, nowMs = Date.now()): boolean {
  if (!token) return false;
  const payload = token.split(".")[1];
  if (!payload) return false;
  try {
    const json = JSON.parse(Buffer.from(payload.replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8"));
    return typeof json.exp === "number" && json.exp * 1000 - EXPIRY_MARGIN_S * 1000 > nowMs;
  } catch {
    return false;
  }
}
