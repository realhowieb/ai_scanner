// Signed-out API calls for the landing, pricing and unsubscribe pages. Only the
// routes listed here pass, with no cookies or tokens attached either way.
import type { Upstream } from "./bff";
import { json, requestIdFor, sameOrigin } from "./bff";

const ROUTES: Record<string, ReadonlyArray<"GET" | "POST">> = {
  "v1/plans": ["GET"],
  "v1/events": ["POST"],
  "v1/email-preferences/unsubscribe": ["GET", "POST"],
};
const MAX_BODY = 4096;
const GATEWAY = new Set([502, 503, 504]);

/** Utm tags and the referrer, cleaned for the API (it cleans them again). */
export function cleanAttribution(raw: unknown): Record<string, string> | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!/^(utm_(source|medium|campaign|content|term)|referrer)$/.test(k) || typeof v !== "string" || !v) continue;
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
