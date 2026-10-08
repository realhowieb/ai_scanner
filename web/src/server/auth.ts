// Sign-in and sign-out for the BFF. Credentials pass through to the API (retried once
// only when a gateway error shows the request never reached it) and are
// never stored or logged; the token pair goes straight into HttpOnly cookies.
import type { Upstream } from "./bff";
import { SESSION_EXPIRED, json, requestIdFor, sameOrigin, upstreamRefresh } from "./bff";
import { ACCESS_COOKIE, REFRESH_COOKIE, accessTokenUsable, clearedCookies, parseCookies, sessionCookies } from "./cookies";
import type { TokenPair } from "./cookies";
import { refreshOnce } from "./refresh";

const FAILED = "Sign-in failed.";

async function detailOf(res: Response): Promise<unknown> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return body.detail ?? FAILED;
  } catch {
    return FAILED;
  }
}

/** WEB_BETA_ALLOWED_EMAILS (server-only, comma-separated): when set, only these accounts
 * can sign in to this deployment. Unset = everyone with an HSF account. */
export function betaAllowlist(raw: string | undefined = process.env.WEB_BETA_ALLOWED_EMAILS): Set<string> | null {
  const items = (raw || "").split(",").map((e) => e.trim().toLowerCase()).filter(Boolean);
  return items.length ? new Set(items) : null;
}

/** Shown when the API is still starting (Render answers 502/503/504 while it wakes up). */
export const API_STARTING = "The HSF service is starting up. Wait a moment and sign in again.";
const GATEWAY = new Set([502, 503, 504]);
export const LOGIN_RETRY_MS = 3000;

export const NOT_INVITED = "This preview of the new HSF app is invite-only for now. Keep using the classic app; we'll let you know when it opens.";

export async function login(req: Request, upstream: Upstream, allow: Set<string> | null = betaAllowlist()): Promise<Response> {
  const rid = requestIdFor(req);
  if (!sameOrigin(req)) return json({ detail: "Cross-site request refused." }, 403, rid);
  let email = "";
  let password = "";
  try {
    const body = (await req.json()) as { email?: unknown; password?: unknown };
    email = typeof body.email === "string" ? body.email.trim() : "";
    password = typeof body.password === "string" ? body.password : "";
  } catch {
    /* handled below */
  }
  if (!email || !password || email.length > 320 || password.length > 1024) {
    return json({ detail: "Enter your email and password." }, 400, rid);
  }
  const attempt = () => upstream("/v1/auth/login", {
    method: "POST",
    headers: { "content-type": "application/json", "x-request-id": rid },
    body: JSON.stringify({ email, password, client: "web" }),
  });
  let res: Response;
  try {
    res = await attempt();
    // A gateway error means the request never reached the API (it was waking up or
    // restarting), so no session was created and one retry is safe.
    if (GATEWAY.has(res.status)) {
      await new Promise((r) => setTimeout(r, LOGIN_RETRY_MS));
      res = await attempt();
    }
  } catch {
    return json({ detail: "Couldn't reach the HSF service. Try again." }, 502, rid);
  }
  if (GATEWAY.has(res.status)) {
    // Render's own error page isn't JSON; an API 503 ("database unavailable") keeps its message.
    const detail = await detailOf(res);
    return json({ detail: detail === FAILED ? API_STARTING : detail }, 503, rid, [], { "retry-after": "30" });
  }
  const id = res.headers.get("x-request-id") || rid;
  if (!res.ok) {
    const extra: Record<string, string> = {};
    const ra = res.headers.get("retry-after");
    if (ra) extra["retry-after"] = ra;
    const detail = res.status === 401 ? "Wrong email or password." : await detailOf(res);
    return json({ detail }, res.status, id, [], extra);
  }
  const pair = (await res.json()) as TokenPair;
  if (allow) {
    // The account's email comes from the API, so signing in with a username works too.
    let invited = false;
    try {
      const me = await upstream("/v1/me", { headers: { authorization: `Bearer ${pair.access_token}`, "x-request-id": rid } });
      const account = me.ok ? ((await me.json()) as { email?: string }) : {};
      invited = !!account.email && allow.has(account.email.trim().toLowerCase());
    } catch {
      return json({ detail: "Couldn't reach the HSF service. Try again." }, 502, id);
    }
    if (!invited) {
      try {
        await upstream("/v1/auth/logout", { method: "POST", headers: { "content-type": "application/json", "x-request-id": rid },
          body: JSON.stringify({ refresh_token: pair.refresh_token }) });
      } catch {
        /* the session was never handed to the browser either way */
      }
      return json({ detail: NOT_INVITED, code: "not_invited" }, 403, id);
    }
  }
  return json({ ok: true }, 200, id, sessionCookies(pair));
}

export async function logout(req: Request, upstream: Upstream): Promise<Response> {
  const rid = requestIdFor(req);
  if (!sameOrigin(req)) return json({ detail: "Cross-site request refused." }, 403, rid);
  const refresh = parseCookies(req.headers.get("cookie"))[REFRESH_COOKIE];
  if (refresh) {
    try {
      await upstream("/v1/auth/logout", {
        method: "POST",
        headers: { "content-type": "application/json", "x-request-id": rid },
        body: JSON.stringify({ refresh_token: refresh }),
      });
    } catch {
      /* the cookies are cleared either way */
    }
  }
  const headers = new Headers({ "cache-control": "no-store", "x-request-id": rid });
  for (const c of clearedCookies()) headers.append("set-cookie", c);
  return new Response(null, { status: 204, headers });
}

/** Change the password. The API signs every other session out and answers with a new
 * token pair for this one, which goes straight into the cookies (never to the browser). */
export async function changePassword(req: Request, upstream: Upstream): Promise<Response> {
  const rid = requestIdFor(req);
  if (!sameOrigin(req)) return json({ detail: "Cross-site request refused." }, 403, rid);
  let current = "";
  let next = "";
  try {
    const body = (await req.json()) as { current_password?: unknown; new_password?: unknown };
    current = typeof body.current_password === "string" ? body.current_password : "";
    next = typeof body.new_password === "string" ? body.new_password : "";
  } catch {
    /* handled below */
  }
  if (!current || !next || current.length > 256 || next.length > 256) {
    return json({ detail: "Enter your current password and a new one." }, 400, rid);
  }
  const jar = parseCookies(req.headers.get("cookie"));
  let access = jar[ACCESS_COOKIE];
  const refresh = jar[REFRESH_COOKIE];
  let pending: string[] = [];
  try {
    if (!accessTokenUsable(access)) {
      const out = refresh ? await refreshOnce(refresh, upstreamRefresh(upstream, rid)) : null;
      if (!out || !out.ok) return json(SESSION_EXPIRED, 401, rid, clearedCookies());
      access = out.pair.access_token;
      pending = sessionCookies(out.pair);
    }
    const res = await upstream("/v1/me/password", {
      method: "POST",
      headers: { authorization: `Bearer ${access}`, "content-type": "application/json", accept: "application/json", "x-request-id": rid },
      body: JSON.stringify({ current_password: current, new_password: next }),
    });
    const id = res.headers.get("x-request-id") || rid;
    if (res.status === 401) return json(SESSION_EXPIRED, 401, id, clearedCookies());
    if (!res.ok) {
      const extra: Record<string, string> = {};
      const ra = res.headers.get("retry-after");
      if (ra) extra["retry-after"] = ra;
      const detail = await detailOf(res);
      return json({ detail: detail === FAILED ? "Couldn't change the password." : detail }, res.status, id, pending, extra);
    }
    return json({ ok: true }, 200, id, sessionCookies((await res.json()) as TokenPair));
  } catch {
    return json({ detail: "Couldn't reach the HSF service. Try again." }, 502, rid, pending);
  }
}

type Fields = Record<string, "string" | "boolean">;

async function readFields(req: Request, fields: Fields): Promise<Record<string, string | boolean> | null> {
  let body: Record<string, unknown>;
  try {
    body = (await req.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
  const out: Record<string, string | boolean> = {};
  for (const [k, t] of Object.entries(fields)) {
    const v = body[k];
    if (t === "boolean") out[k] = v === true;
    else if (typeof v === "string" && v.length > 0 && v.length <= 1024) out[k] = v;
    else return null;
  }
  return out;
}

async function forward(path: string, body: unknown, rid: string, upstream: Upstream): Promise<Response> {
  return upstream(path, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/json", "x-request-id": rid },
    body: JSON.stringify(body),
  });
}

async function passError(res: Response, rid: string): Promise<Response> {
  const extra: Record<string, string> = {};
  const ra = res.headers.get("retry-after");
  if (ra) extra["retry-after"] = ra;
  const detail = GATEWAY.has(res.status) ? API_STARTING : await detailOf(res);
  return json({ detail: detail === FAILED ? "Something went wrong. Try again." : detail }, res.status, res.headers.get("x-request-id") || rid, [], extra);
}

/** Public account flows that need no session: password reset (request and confirm) and
 * email verification. The API's answer passes through; nothing is stored. */
const PUBLIC: Record<string, { path: string; fields: Fields }> = {
  "password-reset": { path: "/v1/auth/password-reset", fields: { email: "string" } },
  "password-reset-confirm": { path: "/v1/auth/password-reset/confirm", fields: { token: "string", new_password: "string" } },
  "verify-email": { path: "/v1/auth/verify-email", fields: { token: "string" } },
};

export async function publicAuth(req: Request, flow: keyof typeof PUBLIC, upstream: Upstream): Promise<Response> {
  const rid = requestIdFor(req);
  if (!sameOrigin(req)) return json({ detail: "Cross-site request refused." }, 403, rid);
  const spec = PUBLIC[flow]!;
  const body = await readFields(req, spec.fields);
  if (!body) return json({ detail: "Fill in every field." }, 400, rid);
  let res: Response;
  try {
    res = await forward(spec.path, body, rid, upstream);
  } catch {
    return json({ detail: "Couldn't reach the HSF service. Try again." }, 502, rid);
  }
  if (!res.ok) return passError(res, rid);
  const out = (await res.json().catch(() => ({}))) as { message?: string };
  return json({ ok: true, message: out.message ?? "Done." }, 200, res.headers.get("x-request-id") || rid);
}

/** Create a Free account and sign in: the API's token pair goes into the cookies only.
 * With an invite list (WEB_BETA_ALLOWED_EMAILS) other emails are refused before any account is created. */
export async function signup(req: Request, upstream: Upstream, allow: Set<string> | null = betaAllowlist()): Promise<Response> {
  const rid = requestIdFor(req);
  if (!sameOrigin(req)) return json({ detail: "Cross-site request refused." }, 403, rid);
  const body = await readFields(req, { email: "string", password: "string", username: "string", accept_terms: "boolean" });
  if (!body) return json({ detail: "Fill in every field." }, 400, rid);
  const email = String(body.email).trim().toLowerCase();
  if (allow && !allow.has(email)) return json({ detail: NOT_INVITED, code: "not_invited" }, 403, rid);
  let res: Response;
  try {
    res = await forward("/v1/auth/signup", { ...body, email, client: "web" }, rid, upstream);
  } catch {
    return json({ detail: "Couldn't reach the HSF service. Try again." }, 502, rid);
  }
  if (!res.ok) return passError(res, rid);
  const out = (await res.json()) as TokenPair & { email?: string; verification_sent?: boolean };
  return json({ ok: true, email: out.email ?? email, verification_sent: !!out.verification_sent }, 201,
    res.headers.get("x-request-id") || rid, sessionCookies(out));
}
