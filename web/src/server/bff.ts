// Backend-for-frontend: the browser calls /api/hsf/v1/... on this origin; this
// server attaches the access token from its HttpOnly cookie and forwards to the API.
// It refreshes before forwarding when the access token is missing or expiring,
// retries an authenticated request once after a 401, and ends the session (401
// code session_expired, cookies cleared) only when refresh credentials are rejected.
import { ACCESS_COOKIE, REFRESH_COOKIE, accessTokenUsable, clearedCookies, parseCookies, sessionCookies } from "./cookies";
import type { TokenPair } from "./cookies";
import { refreshOnce } from "./refresh";
import type { RefreshOutcome } from "./refresh";

export type Upstream = (path: string, init: RequestInit) => Promise<Response>;

const METHODS = new Set(["GET", "POST", "PATCH", "DELETE"]);
const PATH_RE = /^v1\/[A-Za-z0-9._~\-/]{0,200}$/;
const REQUEST_ID_RE = /^[A-Za-z0-9._-]{8,64}$/;
const MAX_BODY_BYTES = 256 * 1024;
const PASS_HEADERS = ["content-type", "retry-after"];
const ETAG_RE = /^(W\/)?"[A-Za-z0-9]{1,64}"$/;

/** API routes that answer with a token pair. They go through /api/auth/* instead, which
 * keeps the tokens in cookies; proxied, the pair would reach browser JavaScript. */
const TOKEN_ROUTES = new Set(["v1/me/password"]);

export const SESSION_EXPIRED = { detail: "Your session has ended. Sign in again.", code: "session_expired" };

export function requestIdFor(req: Request): string {
  const given = req.headers.get("x-request-id");
  return given && REQUEST_ID_RE.test(given) ? given : `web-${crypto.randomUUID()}`;
}

/** Mutating calls must come from this site (cookies are SameSite=Lax; this closes the rest). */
export function sameOrigin(req: Request): boolean {
  if (req.method === "GET" || req.method === "HEAD") return true;
  const origin = req.headers.get("origin");
  const host = req.headers.get("x-forwarded-host") || req.headers.get("host");
  if (origin) {
    try {
      return !!host && new URL(origin).host === host;
    } catch {
      return false;
    }
  }
  return req.headers.get("sec-fetch-site") === "same-origin";
}

export function json(body: unknown, status: number, requestId: string, cookies: string[] = [], extra: Record<string, string> = {}): Response {
  const headers = new Headers({ "content-type": "application/json", "cache-control": "no-store", "x-request-id": requestId, ...extra });
  for (const c of cookies) headers.append("set-cookie", c);
  return new Response(JSON.stringify(body), { status, headers });
}

export function upstreamRefresh(upstream: Upstream, requestId: string) {
  return async (refreshToken: string): Promise<RefreshOutcome> => {
    const r = await upstream("/v1/auth/refresh", {
      method: "POST",
      headers: { "content-type": "application/json", "x-request-id": requestId },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!r.ok) return { ok: false, status: r.status, retryAfter: r.headers.get("retry-after") ?? undefined };
    return { ok: true, pair: (await r.json()) as TokenPair };
  };
}

/** A service outage must not destroy a valid refresh session. */
export function refreshFailure(outcome: Extract<RefreshOutcome, { ok: false }> | null, rid: string): Response {
  if (!outcome || outcome.status === 401) return json(SESSION_EXPIRED, 401, rid, clearedCookies());
  const status = [429, 502, 503, 504].includes(outcome.status) ? outcome.status : 503;
  return json({ detail: "Your session couldn't be refreshed right now. Please try again." }, status, rid, [],
    { "retry-after": outcome.retryAfter || "5" });
}

export async function proxy(req: Request, apiPath: string, upstream: Upstream): Promise<Response> {
  const rid = requestIdFor(req);
  if (!METHODS.has(req.method) || !PATH_RE.test(apiPath) || apiPath.startsWith("v1/auth/") || apiPath.includes("..")
    || TOKEN_ROUTES.has(apiPath.replace(/\/+$/, ""))) {
    return json({ detail: "Not found" }, 404, rid);
  }
  if (!sameOrigin(req)) return json({ detail: "Cross-site request refused." }, 403, rid);

  const jar = parseCookies(req.headers.get("cookie"));
  let access = jar[ACCESS_COOKIE];
  const refresh = jar[REFRESH_COOKIE];
  const setCookies: string[] = [];
  let refreshed = false;

  const doRefresh = async (): Promise<Response | null> => {
    if (!refresh) return refreshFailure(null, rid);
    const out = await refreshOnce(refresh, upstreamRefresh(upstream, rid));
    if (!out.ok) return refreshFailure(out, rid);
    access = out.pair.access_token;
    setCookies.splice(0, setCookies.length, ...sessionCookies(out.pair));
    refreshed = true;
    return null;
  };

  if (!accessTokenUsable(access)) {
    const failure = await doRefresh();
    if (failure) return failure;
  }

  let body: ArrayBuffer | undefined;
  if (req.method !== "GET") {
    body = await req.arrayBuffer();
    if (body.byteLength > MAX_BODY_BYTES) return json({ detail: "Request too large." }, 413, rid);
  }
  const search = new URL(req.url).search;
  // ETag revalidation (api/etag.py): the browser's If-None-Match goes upstream, and an
  // unchanged answer comes back as an empty 304 instead of the whole body.
  const ifNoneMatch = req.method === "GET" ? req.headers.get("if-none-match") : null;
  const send = () =>
    upstream(`/${apiPath}${search}`, {
      method: req.method,
      headers: {
        authorization: `Bearer ${access}`,
        accept: "application/json",
        "x-request-id": rid,
        ...(ifNoneMatch && ifNoneMatch.length <= 512 ? { "if-none-match": ifNoneMatch } : {}),
        ...(body && body.byteLength ? { "content-type": "application/json" } : {}),
      },
      body: body && body.byteLength ? body : undefined,
    });

  let res: Response;
  try {
    res = await send();
    if (res.status === 401 && !refreshed) {
      const failure = await doRefresh();
      if (failure) return failure;
      res = await send();
    }
  } catch (e) {
    const timeout = e instanceof Error && (e.name === "TimeoutError" || e.name === "AbortError");
    return json({ detail: timeout ? "The HSF service took too long to answer. Try again." : "Couldn't reach the HSF service. Try again." },
      timeout ? 504 : 502, rid, setCookies);
  }
  if (res.status === 401) return json(SESSION_EXPIRED, 401, rid, clearedCookies());

  const headers = new Headers({ "cache-control": "no-store", "x-request-id": res.headers.get("x-request-id") || rid });
  for (const h of PASS_HEADERS) {
    const v = res.headers.get(h);
    if (v) headers.set(h, v);
  }
  // A GET answer the API tagged may be kept by this browser only (private) and must be
  // revalidated on every use (no-cache); everything else stays no-store.
  const etag = res.headers.get("etag");
  if (req.method === "GET" && (res.status === 200 || res.status === 304) && etag && ETAG_RE.test(etag)) {
    headers.set("etag", etag);
    headers.set("cache-control", "private, no-cache");
  }
  for (const c of setCookies) headers.append("set-cookie", c);
  const payload = res.status === 204 || res.status === 304 ? null : await res.arrayBuffer();
  return new Response(payload, { status: res.status, headers });
}
