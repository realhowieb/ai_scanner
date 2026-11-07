// Signed-out API calls for the landing, pricing and unsubscribe pages. Only the
// routes listed here pass, with no cookies or tokens attached either way.
import type { Upstream } from "./bff";
import { json, requestIdFor, sameOrigin } from "./bff";

const ROUTES: Record<string, ReadonlyArray<"GET" | "POST">> = {
  "v1/plans": ["GET"],
  "v1/public/founding": ["GET"],
  "v1/legal/status": ["GET"],
  "v1/public/top-setups": ["GET"],
  "v1/events": ["POST"],
  "v1/client-errors": ["POST"],
  "v1/email-preferences/unsubscribe": ["GET", "POST"],
  "v1/public/list": ["GET"],
  "v1/public/subscribe": ["POST"],
  "v1/public/subscribe/status": ["GET"],
  "v1/public/subscribe/action": ["POST"],
};
const MAX_BODY = 4096;
const GATEWAY = new Set([502, 503, 504]);

// Plans change rarely, so the web server keeps the last good GET /v1/plans for five
// minutes: one slow or failed API call can't blank the pricing section. After that the
// last good copy (up to a day old) is served at once while a fresh one loads.
const PLANS_PATH = "v1/plans";
export const PLANS_TTL_MS = 5 * 60_000;
const PLANS_STALE_MS = 24 * 60 * 60_000;
let plansCache: { body: string; at: number } | null = null;
let plansInFlight: Promise<Response> | null = null;

/** Tests only: forget the cached plans. */
export function _resetPlansCache(): void {
  plansCache = null;
  plansInFlight = null;
}

function plansResponse(body: string, rid: string, cache: "hit" | "miss" | "stale"): Response {
  return new Response(body, {
    status: 200,
    headers: { "content-type": "application/json", "x-request-id": rid, "cache-control": "no-store", "x-hsf-cache": cache },
  });
}

/** Utm tags, the referrer and an invite code (ref), cleaned for the API (it cleans them again). */
export function cleanAttribution(raw: unknown): Record<string, string> | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!/^(utm_(source|medium|campaign|content|term)|referrer|ref)$/.test(k) || typeof v !== "string" || !v) continue;
    out[k] = v.slice(0, 200);
  }
  return Object.keys(out).length ? out : null;
}

export async function publicApi(req: Request, segments: string[], upstream: Upstream): Promise<Response> {
  const rid = requestIdFor(req);
  const path = segments.join("/");
  const method = req.method.toUpperCase();
  if (!ROUTES[path]) return json({ detail: "Not found." }, 404, rid);
  if (!ROUTES[path].includes(method as "GET" | "POST")) return json({ detail: "Method not allowed." }, 405, rid);
  let body: string | undefined;
  if (method === "POST") {
    if (!sameOrigin(req)) return json({ detail: "Cross-site request refused." }, 403, rid);
    body = await req.text();
    if (body.length > MAX_BODY) return json({ detail: "Request too large." }, 413, rid);
  }
  const query = new URL(req.url).search;
  if (path === PLANS_PATH && method === "GET" && !query) return cachedPlans(rid, upstream);
  return forward(path, query, method, body, rid, upstream);
}

async function cachedPlans(rid: string, upstream: Upstream): Promise<Response> {
  const age = plansCache ? Date.now() - plansCache.at : Infinity;
  if (plansCache && age < PLANS_TTL_MS) return plansResponse(plansCache.body, rid, "hit");
  const refresh = refreshPlans(rid, upstream);
  // Past five minutes, answer at once with the last good copy and refresh behind it,
  // so a slow API call never holds up the page.
  if (plansCache && age < PLANS_STALE_MS) {
    refresh.catch(() => undefined);
    return plansResponse(plansCache.body, rid, "stale");
  }
  return refresh;
}

/** One API call at a time; keeps the body when it is good JSON. */
function refreshPlans(rid: string, upstream: Upstream): Promise<Response> {
  plansInFlight ??= (async () => {
    const res = await forward(PLANS_PATH, "", "GET", undefined, rid, upstream);
    if (!res.ok) return res;
    const text = await res.text();
    try {
      JSON.parse(text);
    } catch {
      return json({ detail: "Couldn't read plans from the HSF service. Try again." }, 502, rid);
    }
    plansCache = { body: text, at: Date.now() };
    return plansResponse(text, rid, "miss");
  })().finally(() => { plansInFlight = null; });
  return plansInFlight.then((r) => r.clone());
}

async function forward(path: string, query: string, method: string, body: string | undefined, rid: string, upstream: Upstream): Promise<Response> {
  let res: Response;
  try {
    res = await upstream(`/${path}${query}`, {
      method,
      headers: { accept: "application/json", "x-request-id": rid, ...(body !== undefined ? { "content-type": "application/json" } : {}) },
      body,
    });
  } catch {
    return json({ detail: "Couldn't reach the HSF service. Try again." }, 502, rid);
  }
  const id = res.headers.get("x-request-id") || rid;
  if (GATEWAY.has(res.status)) return json({ detail: "The HSF service is starting up. Try again in a moment." }, res.status, id);
  const text = await res.text();
  return new Response(res.status === 204 || res.status === 202 ? null : text || null, {
    status: res.status,
    headers: {
      "content-type": "application/json", "x-request-id": id, "cache-control": "no-store",
      ...(res.headers.get("retry-after") ? { "retry-after": res.headers.get("retry-after")! } : {}),
    },
  });
}
