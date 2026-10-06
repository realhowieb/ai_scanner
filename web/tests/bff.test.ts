// @vitest-environment node
import { beforeEach, describe, expect, it, vi } from "vitest";

import { login, logout } from "@/server/auth";
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
