// Sign-in and sign-out for the BFF. Credentials pass through to the API once and are
// never stored or logged; the token pair goes straight into HttpOnly cookies.
import type { Upstream } from "./bff";
import { json, requestIdFor, sameOrigin } from "./bff";
import { REFRESH_COOKIE, clearedCookies, parseCookies, sessionCookies } from "./cookies";
import type { TokenPair } from "./cookies";

async function detailOf(res: Response): Promise<unknown> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    return body.detail ?? "Sign-in failed.";
  } catch {
    return "Sign-in failed.";
  }
}

/** WEB_BETA_ALLOWED_EMAILS (server-only, comma-separated): when set, only these accounts
 * can sign in to this deployment. Unset = everyone with an HSF account. */
export function betaAllowlist(raw: string | undefined = process.env.WEB_BETA_ALLOWED_EMAILS): Set<string> | null {
  const items = (raw || "").split(",").map((e) => e.trim().toLowerCase()).filter(Boolean);
  return items.length ? new Set(items) : null;
}

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
  let res: Response;
  try {
    res = await upstream("/v1/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json", "x-request-id": rid },
      body: JSON.stringify({ email, password, client: "web" }),
    });
  } catch {
    return json({ detail: "Couldn't reach the HSF service. Try again." }, 502, rid);
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
