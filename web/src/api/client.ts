// Typed API client, generated from the API's OpenAPI contract (src/api/schema.d.ts,
// `npm run api:types`). It calls this site's BFF (/api/hsf), never the API directly,
// so no token is ever visible to browser code.
import createClient from "openapi-fetch";
import type { Middleware } from "openapi-fetch";

import { parseRetryAfter } from "@/lib/retryAfter";
import { clearCachedMe } from "@/session/meCache";

import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly requestId: string | null,
    readonly retryAfterS: number | null,
    readonly body: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }

  get code(): string | null {
    const b = this.body as { code?: unknown } | null;
    return b && typeof b.code === "string" ? b.code : null;
  }
}

let onSessionExpired: () => void = () => {
  if (typeof window === "undefined") return;
  clearCachedMe();
  const here = window.location.pathname + window.location.search;
  // A full page load on purpose: it drops every piece of client state from the old session.
  // eslint-disable-next-line @next/next/no-location-assign-relative-destination
  window.location.assign(`/login?next=${encodeURIComponent(here)}&expired=1`);
};
let expiring = false;

/** Tests and the session provider can replace what happens when the session ends. */
export function setSessionExpiredHandler(fn: () => void): void {
  onSessionExpired = fn;
  expiring = false;
}

export function newRequestId(): string {
  return `web-${crypto.randomUUID()}`;
}

const requestIds: Middleware = {
  onRequest({ request }) {
    if (!request.headers.get("x-request-id")) request.headers.set("x-request-id", newRequestId());
    return request;
  },
};

export function messageFrom(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) return "Some of the values aren't valid.";
  if (status === 503) return "The HSF service is unavailable right now.";
  if (status === 429) return "Too many requests. Wait a moment and try again.";
  return `Something went wrong (HTTP ${status}).`;
}

export function toApiError(response: Response, body: unknown): ApiError {
  const err = new ApiError(response.status, messageFrom(body, response.status), response.headers.get("x-request-id"),
    parseRetryAfter(response.headers.get("retry-after")), body);
  if (response.status === 401 && err.code === "session_expired" && !expiring) {
    expiring = true;
    onSessionExpired();
  }
  return err;
}

export function makeClient(fetchImpl?: typeof fetch, base = "/api/hsf") {
  const origin = typeof window === "undefined" ? "" : window.location.origin;
  const c = createClient<paths>({
    baseUrl: `${origin}${base}`,
    fetch: fetchImpl ?? ((input: Request) => globalThis.fetch(input)), // late-bound, so a patched fetch is used
    credentials: "same-origin",
  });
  c.use(requestIds);
  return c;
}

export const api = makeClient();
/** Signed-out routes (plans, funnel events, unsubscribe links), through /api/public. */
export const publicApi = makeClient(undefined, "/api/public");

type Result<T> = { data?: T; error?: unknown; response: Response };

/** Unwraps an openapi-fetch result, throwing ApiError for any non-2xx answer. */
export async function unwrap<T>(p: Promise<Result<T>>): Promise<T> {
  let r: Result<T>;
  try {
    r = await p;
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "You appear to be offline, or the site couldn't be reached.", null, null, null);
  }
  if (!r.response.ok) throw toApiError(r.response, r.error ?? null);
  return r.data as T;
}
