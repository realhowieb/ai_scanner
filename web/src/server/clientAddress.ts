// The API limits sign-ins and sign-ups per visitor address. Every call from this server
// reaches the API from the server's own address, so without help all visitors would
// share one limit. When HSF_PROXY_SECRET is set (same value on hsf-api-v1), each call
// carries the visitor's address plus the secret; the API trusts the address only with it.
import type { Upstream } from "./bff";

export const CLIENT_IP_HEADER = "x-hsf-client-ip";
export const PROXY_KEY_HEADER = "x-hsf-proxy-key";
const IP_RE = /^[0-9A-Fa-f:.]{2,45}$/;

/** The visitor's address. Render's proxy appends the real client address as the LAST
 * X-Forwarded-For entry; earlier entries come from the client and can be forged. */
export function clientAddress(req: Request): string | null {
  const xff = req.headers.get("x-forwarded-for") || "";
  const last = xff.split(",").map((s) => s.trim()).filter(Boolean).pop() || "";
  return IP_RE.test(last) ? last : null;
}

export function withClientAddress(req: Request, upstream: Upstream, secret: string | undefined = process.env.HSF_PROXY_SECRET): Upstream {
  const key = (secret || "").trim();
  const ip = clientAddress(req);
  if (!key || !ip) return upstream;
  return (path, init) => {
    const headers = new Headers(init.headers);
    headers.set(CLIENT_IP_HEADER, ip);
    headers.set(PROXY_KEY_HEADER, key);
    return upstream(path, { ...init, headers });
  };
}
