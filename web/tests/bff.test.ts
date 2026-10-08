// @vitest-environment node
import { beforeEach, describe, expect, it, vi } from "vitest";

import { API_STARTING, NOT_INVITED, changePassword, login, logout, publicAuth, signup } from "@/server/auth";
import { SESSION_EXPIRED, proxy } from "@/server/bff";
import type { Upstream } from "@/server/bff";
import { ACCESS_COOKIE, REFRESH_COOKIE, accessTokenUsable, secureCookies } from "@/server/cookies";
import { _resetRefreshState } from "@/server/refresh";

import { jsonResponse, jwt, setCookies } from "./helpers";

const HOST = "app.example.test";

function req(path: string, { method = "GET", cookies = {}, origin = `https://${HOST}`, body }: {
  method?: string; cookies?: Record<string, string>; origin?: string | null; body?: unknown;
} = {}): Request {
  const headers: Record<string, string> = { host: HOST };
  const jar = Object.entries(cookies).map(([k, v]) => `${k}=${v}`).join("; ");
  if (jar) headers.cookie = jar;
  if (origin && method !== "GET") headers.origin = origin;
  if (body !== undefined) headers["content-type"] = "application/json";
  return new Request(`https://${HOST}${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
}

const pair = (n: number) => ({ access_token: jwt(900), refresh_token: `rt-${n}`, expires_in: 900, token_type: "bearer" });

type Call = { path: string; init: RequestInit };

function fakeUpstream(handler: (c: Call, calls: Call[]) => Response | Promise<Response>) {
  const calls: Call[] = [];
  const up: Upstream = async (path, init) => {
    const c = { path, init };
    calls.push(c);
    return handler(c, calls);
  };
  return { up, calls };
}

const auth = (c: Call) => (c.init.headers as Record<string, string>).authorization;

beforeEach(() => _resetRefreshState());

describe("BFF proxy", () => {
  it("forwards with the cookie's access token and passes status, Retry-After and request id", async () => {
    const at = jwt(600);
    const { up, calls } = fakeUpstream(() => jsonResponse({ detail: "busy" }, 503, { "retry-after": "30", "x-request-id": "srv-12345678" }));
    const res = await proxy(req("/api/hsf/v1/today", { cookies: { [ACCESS_COOKIE]: at, [REFRESH_COOKIE]: "rt-0" } }), "v1/today", up);
    expect(res.status).toBe(503);
    expect(res.headers.get("retry-after")).toBe("30");
    expect(res.headers.get("x-request-id")).toBe("srv-12345678");
    expect(calls).toHaveLength(1);
    expect(auth(calls[0]!)).toBe(`Bearer ${at}`);
    expect(setCookies(res)).toEqual([]);
  });

  it("refreshes first when the access token is missing, then sets rotated cookies", async () => {
    const { up, calls } = fakeUpstream((c) => (c.path === "/v1/auth/refresh" ? jsonResponse(pair(1)) : jsonResponse({ ok: 1 })));
    const res = await proxy(req("/api/hsf/v1/me", { cookies: { [REFRESH_COOKIE]: "rt-0" } }), "v1/me", up);
    expect(res.status).toBe(200);
    expect(calls.map((c) => c.path)).toEqual(["/v1/auth/refresh", "/v1/me"]);
    const cookies = setCookies(res).join("\n");
    expect(cookies).toContain(`${REFRESH_COOKIE}=rt-1`);
    expect(cookies).toMatch(/HttpOnly; Secure; SameSite=Lax/);
  });

  it("retries an authenticated request once after a 401, with the refreshed token", async () => {
    const { up, calls } = fakeUpstream((c, all) => {
      if (c.path === "/v1/auth/refresh") return jsonResponse(pair(2));
      return all.filter((x) => x.path === "/v1/me").length === 1 ? jsonResponse({ detail: "expired" }, 401) : jsonResponse({ ok: 1 });
    });
    const res = await proxy(req("/api/hsf/v1/me", { cookies: { [ACCESS_COOKIE]: jwt(600), [REFRESH_COOKIE]: "rt-1" } }), "v1/me", up);
    expect(res.status).toBe(200);
    expect(calls.map((c) => c.path)).toEqual(["/v1/me", "/v1/auth/refresh", "/v1/me"]);
    expect(auth(calls[2]!)).not.toBe(auth(calls[0]!));
  });

  it("ends the session (401 session_expired, cookies cleared) when the refresh fails", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse({ detail: "invalid" }, 401));
    const res = await proxy(req("/api/hsf/v1/me", { cookies: { [REFRESH_COOKIE]: "rt-revoked" } }), "v1/me", up);
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual(SESSION_EXPIRED);
    expect(setCookies(res).every((c) => c.includes("Max-Age=0"))).toBe(true);
    expect(calls.map((c) => c.path)).toEqual(["/v1/auth/refresh"]);
  });

  it("ends the session when the API still answers 401 after a refresh (no retry loop)", async () => {
    const { up, calls } = fakeUpstream((c) => (c.path === "/v1/auth/refresh" ? jsonResponse(pair(3)) : jsonResponse({ detail: "no" }, 401)));
    const res = await proxy(req("/api/hsf/v1/me", { cookies: { [ACCESS_COOKIE]: jwt(600), [REFRESH_COOKIE]: "rt-2" } }), "v1/me", up);
    expect(res.status).toBe(401);
    expect(calls.filter((c) => c.path === "/v1/me")).toHaveLength(2);
    expect(calls.filter((c) => c.path === "/v1/auth/refresh")).toHaveLength(1);
  });

  it("answers 401 without calling the API when there is no session at all", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse({}));
    const res = await proxy(req("/api/hsf/v1/me"), "v1/me", up);
    expect(res.status).toBe(401);
    expect(calls).toHaveLength(0);
  });

  it("shares one refresh between concurrent requests holding the same refresh token", async () => {
    let refreshes = 0;
    const { up } = fakeUpstream(async (c) => {
      if (c.path === "/v1/auth/refresh") {
        refreshes += 1;
        await new Promise((r) => setTimeout(r, 20));
        return jsonResponse(pair(9));
      }
      return jsonResponse({ ok: 1 });
    });
    const results = await Promise.all(Array.from({ length: 6 }, () =>
      proxy(req("/api/hsf/v1/me", { cookies: { [REFRESH_COOKIE]: "rt-shared" } }), "v1/me", up)));
    expect(refreshes).toBe(1);
    expect(results.every((r) => r.status === 200)).toBe(true);
    expect(new Set(results.map((r) => setCookies(r).find((c) => c.startsWith(REFRESH_COOKIE))))).toEqual(new Set([`${REFRESH_COOKIE}=rt-9; Path=/; Max-Age=2592000; HttpOnly; Secure; SameSite=Lax`]));
    // A request arriving just after the rotation reuses it instead of replaying the old token.
    await proxy(req("/api/hsf/v1/me", { cookies: { [REFRESH_COOKIE]: "rt-shared" } }), "v1/me", up);
    expect(refreshes).toBe(1);
  });

  it("refuses auth routes, non-v1 paths and traversal", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse({}));
    const cookies = { [ACCESS_COOKIE]: jwt(600) };
    for (const p of ["v1/auth/refresh", "v1/auth/login", "healthz", "v2/x", "v1/../admin"]) {
      expect((await proxy(req(`/api/hsf/${p}`, { cookies }), p, up)).status).toBe(404);
    }
    expect(calls).toHaveLength(0);
  });

  it("refuses cross-site writes and accepts same-origin ones", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse({ scan_id: "a" }, 202));
    const cookies = { [ACCESS_COOKIE]: jwt(600) };
    const evil = await proxy(req("/api/hsf/v1/scans", { method: "POST", cookies, origin: "https://evil.example", body: {} }), "v1/scans", up);
    expect(evil.status).toBe(403);
    const none = await proxy(req("/api/hsf/v1/scans", { method: "POST", cookies, origin: null, body: {} }), "v1/scans", up);
    expect(none.status).toBe(403);
    const ok = await proxy(req("/api/hsf/v1/scans", { method: "POST", cookies, body: { universe: "sp500" } }), "v1/scans", up);
    expect(ok.status).toBe(202);
    expect(calls).toHaveLength(1);
    expect(new TextDecoder().decode(calls[0]!.init.body as ArrayBuffer)).toBe('{"universe":"sp500"}');
  });

  it("keeps the query string and maps a timeout to 504", async () => {
    const { up, calls } = fakeUpstream(() => { throw Object.assign(new Error("t"), { name: "TimeoutError" }); });
    const res = await proxy(req("/api/hsf/v1/scans/latest?limit=5&signal=breakout", { cookies: { [ACCESS_COOKIE]: jwt(600) } }), "v1/scans/latest", up);
    expect(calls[0]!.path).toBe("/v1/scans/latest?limit=5&signal=breakout");
    expect(res.status).toBe(504);
  });
});

describe("cookie mode", () => {
  it("is Secure with __Host- names in production, plain on local http dev", () => {
    expect(ACCESS_COOKIE).toBe("__Host-hsf_at");
    expect(secureCookies({ NODE_ENV: "production" })).toBe(true);
    expect(secureCookies({ NODE_ENV: "development" })).toBe(false);
    expect(secureCookies({ NODE_ENV: "development", HSF_COOKIE_SECURE: "1" })).toBe(true);
  });
});

describe("access token expiry check", () => {
  it("treats expiring, malformed and missing tokens as unusable", () => {
    expect(accessTokenUsable(jwt(600))).toBe(true);
    expect(accessTokenUsable(jwt(10))).toBe(false);
    expect(accessTokenUsable("garbage")).toBe(false);
    expect(accessTokenUsable(undefined)).toBe(false);
  });
});

describe("login and logout", () => {
  const body = { email: "a@example.invalid", password: "pw" };

  it("puts the token pair only in HttpOnly cookies, never in the body", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse(pair(5)));
    const res = await login(req("/api/auth/login", { method: "POST", body }), up);
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).not.toContain("rt-5");
    expect(text).not.toContain("eyJ");
    expect(setCookies(res).every((c) => c.includes("HttpOnly") && c.includes("Secure"))).toBe(true);
    expect(JSON.parse(calls[0]!.init.body as string)).toMatchObject({ client: "web" });
  });

  it("maps a wrong password and passes the lockout's Retry-After", async () => {
    let { up } = fakeUpstream(() => jsonResponse({ detail: "Invalid credentials" }, 401));
    let res = await login(req("/api/auth/login", { method: "POST", body }), up);
    expect(res.status).toBe(401);
    expect((await res.json()).detail).toBe("Wrong email or password.");
    ({ up } = fakeUpstream(() => jsonResponse({ detail: "Too many attempts" }, 429, { "retry-after": "120" })));
    res = await login(req("/api/auth/login", { method: "POST", body }), up);
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("120");
    expect(setCookies(res)).toEqual([]);
  });

  it("retries once when the API is waking up, then signs in", async () => {
    vi.useFakeTimers();
    try {
      const { up, calls } = fakeUpstream((_c, all) =>
        all.length === 1 ? new Response("<html>502</html>", { status: 502 }) : jsonResponse(pair(7)));
      const pending = login(req("/api/auth/login", { method: "POST", body }), up);
      await vi.advanceTimersByTimeAsync(3000);
      const res = await pending;
      expect(res.status).toBe(200);
      expect(calls).toHaveLength(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("explains a cold start instead of a bare failure, and keeps an API 503's own message", async () => {
    vi.useFakeTimers();
    try {
      let { up, calls } = fakeUpstream(() => new Response("<html>503</html>", { status: 503 }));
      let pending = login(req("/api/auth/login", { method: "POST", body }), up);
      await vi.advanceTimersByTimeAsync(3000);
      let res = await pending;
      expect(res.status).toBe(503);
      expect(calls).toHaveLength(2);
      expect((await res.json()).detail).toBe(API_STARTING);
      expect(res.headers.get("retry-after")).toBe("30");
      expect(setCookies(res)).toEqual([]);
      ({ up, calls } = fakeUpstream(() => jsonResponse({ detail: "database unavailable" }, 503)));
      pending = login(req("/api/auth/login", { method: "POST", body }), up);
      await vi.advanceTimersByTimeAsync(3000);
      res = await pending;
      expect((await res.json()).detail).toBe("database unavailable");
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not retry a wrong password", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse({ detail: "Invalid credentials" }, 401));
    await login(req("/api/auth/login", { method: "POST", body }), up);
    expect(calls).toHaveLength(1);
  });

  it("refuses a cross-site sign-in", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse(pair(6)));
    const res = await login(req("/api/auth/login", { method: "POST", body, origin: "https://evil.example" }), up);
    expect(res.status).toBe(403);
    expect(calls).toHaveLength(0);
  });

  it("revokes the refresh token and clears cookies, even when the API is down", async () => {
    const handler = vi.fn(() => { throw new Error("down"); });
    const { up, calls } = fakeUpstream(handler);
    const res = await logout(req("/api/auth/logout", { method: "POST", cookies: { [REFRESH_COOKIE]: "rt-7" } }), up);
    expect(res.status).toBe(204);
    expect(JSON.parse(calls[0]!.init.body as string)).toEqual({ refresh_token: "rt-7" });
    expect(setCookies(res).every((c) => c.includes("Max-Age=0"))).toBe(true);
  });
});

describe("password change", () => {
  const cookies = { [ACCESS_COOKIE]: jwt(600), [REFRESH_COOKIE]: "rt-0" };
  const body = { current_password: "old-pass", new_password: "new-pass-123" };

  it("is never proxied, because the API answers with a token pair", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse(pair(1)));
    const res = await proxy(req("/api/hsf/v1/me/password", { method: "POST", cookies, body }), "v1/me/password", up);
    expect(res.status).toBe(404);
    expect(calls).toHaveLength(0);
  });

  it("stores the new pair in HttpOnly cookies and returns no token", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse(pair(7)));
    const res = await changePassword(req("/api/auth/password", { method: "POST", cookies, body }), up);
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).not.toContain("rt-7");
    expect(calls[0]!.path).toBe("/v1/me/password");
    expect(auth(calls[0]!)).toBe(`Bearer ${cookies[ACCESS_COOKIE]}`);
    const set = setCookies(res);
    expect(set.some((c) => c.startsWith(`${REFRESH_COOKIE}=rt-7`) && c.includes("HttpOnly"))).toBe(true);
  });

  it("passes the API's rule message and keeps the session on a wrong password", async () => {
    const { up } = fakeUpstream(() => jsonResponse({ detail: "Current password is incorrect." }, 400));
    const res = await changePassword(req("/api/auth/password", { method: "POST", cookies, body }), up);
    expect(res.status).toBe(400);
    expect((await res.json()).detail).toBe("Current password is incorrect.");
    expect(setCookies(res)).toEqual([]);
  });

  it("refreshes an expired access token first, and ends a dead session", async () => {
    const { up, calls } = fakeUpstream((c) => (c.path === "/v1/auth/refresh" ? jsonResponse(pair(2)) : jsonResponse(pair(3))));
    const ok = await changePassword(req("/api/auth/password", { method: "POST", cookies: { [REFRESH_COOKIE]: "rt-0" }, body }), up);
    expect(ok.status).toBe(200);
    expect(calls.map((c) => c.path)).toEqual(["/v1/auth/refresh", "/v1/me/password"]);
    _resetRefreshState();
    const dead = fakeUpstream(() => jsonResponse({ detail: "no" }, 401));
    const res = await changePassword(req("/api/auth/password", { method: "POST", cookies: { [REFRESH_COOKIE]: "rt-9" }, body }), dead.up);
    expect(res.status).toBe(401);
    expect(await res.json()).toEqual(SESSION_EXPIRED);
  });

  it("refuses cross-site requests and empty fields without calling the API", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse(pair(1)));
    expect((await changePassword(req("/api/auth/password", { method: "POST", cookies, body, origin: "https://evil.example" }), up)).status).toBe(403);
    expect((await changePassword(req("/api/auth/password", { method: "POST", cookies, body: { current_password: "x" } }), up)).status).toBe(400);
    expect(calls).toHaveLength(0);
  });
});

describe("public account routes", () => {
  const body = { email: "New@Example.com", password: "a-long-password", username: "ann", accept_terms: true };

  it("sign-up keeps the tokens in cookies and refuses uninvited emails before creating anything", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse({ ...pair(4), email: "new@example.com", verification_sent: true }, 201));
    const res = await signup(req("/api/auth/signup", { method: "POST", body }), up, null);
    expect(res.status).toBe(201);
    expect(await res.text()).not.toContain("rt-4");
    expect(setCookies(res).some((c) => c.startsWith(`${REFRESH_COOKIE}=rt-4`))).toBe(true);
    expect(JSON.parse(calls[0]!.init.body as string)).toMatchObject({ email: "new@example.com", client: "web" });
    const refused = await signup(req("/api/auth/signup", { method: "POST", body }), up, new Set(["someone@else.com"]));
    expect(refused.status).toBe(403);
    expect((await refused.json()).detail).toBe(NOT_INVITED);
    expect(calls).toHaveLength(1);
  });

  it("passes the API's errors and messages for reset and verification, and refuses cross-site posts", async () => {
    const { up, calls } = fakeUpstream((c) => (c.path === "/v1/auth/verify-email"
      ? jsonResponse({ detail: "This link is invalid or has expired." }, 400)
      : jsonResponse({ ok: true, message: "If that email is registered, a reset link has been sent." }, 202)));
    const ok = await publicAuth(req("/api/auth/password-reset", { method: "POST", body: { email: "a@b.co" } }), "password-reset", up);
    expect((await ok.json()).message).toContain("reset link");
    const bad = await publicAuth(req("/api/auth/verify-email", { method: "POST", body: { token: "expired" } }), "verify-email", up);
    expect(bad.status).toBe(400);
    expect((await bad.json()).detail).toContain("invalid");
    expect((await publicAuth(req("/api/auth/verify-email", { method: "POST", body: { token: "x" }, origin: "https://evil.example" }), "verify-email", up)).status).toBe(403);
    expect((await publicAuth(req("/api/auth/password-reset-confirm", { method: "POST", body: { token: "x" } }), "password-reset-confirm", up)).status).toBe(400);
    expect(calls.map((c) => c.path)).toEqual(["/v1/auth/password-reset", "/v1/auth/verify-email"]);
  });
});
