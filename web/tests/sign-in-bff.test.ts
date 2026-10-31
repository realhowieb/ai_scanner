// @vitest-environment node
import { beforeEach, describe, expect, it } from "vitest";

import { NOT_INVITED, emailLinkSignIn, logoutEverywhere, publicAuth } from "@/server/auth";
import type { Upstream } from "@/server/bff";
import { ACCESS_COOKIE, REFRESH_COOKIE } from "@/server/cookies";
import { _resetRefreshState } from "@/server/refresh";

import { jsonResponse, jwt, setCookies } from "./helpers";

const HOST = "app.example.test";

function req(path: string, { cookies = {}, origin = `https://${HOST}`, body }: { cookies?: Record<string, string>; origin?: string; body?: unknown } = {}): Request {
  const headers: Record<string, string> = { host: HOST, origin };
  const jar = Object.entries(cookies).map(([k, v]) => `${k}=${v}`).join("; ");
  if (jar) headers.cookie = jar;
  if (body !== undefined) headers["content-type"] = "application/json";
  return new Request(`https://${HOST}${path}`, { method: "POST", headers, body: body === undefined ? undefined : JSON.stringify(body) });
}

type Call = { path: string; init: RequestInit };
function fakeUpstream(handler: (c: Call) => Response) {
  const calls: Call[] = [];
  const up: Upstream = async (path, init) => { const c = { path, init }; calls.push(c); return handler(c); };
  return { up, calls };
}
const pair = (n: number) => ({ access_token: jwt(900), refresh_token: `rt-${n}`, expires_in: 900, token_type: "bearer" });

beforeEach(() => _resetRefreshState());

describe("emailed sign-in link", () => {
  it("asking for a link passes the email and the Turnstile token through", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse({ message: "If that email has an HSF account, a sign-in link is on its way." }, 202));
    const res = await publicAuth(req("/api/auth/email-link", { body: { email: "a@b.co", turnstile_token: "cf-1" } }), "email-link", up);
    expect(res.status).toBe(200);
    expect((await res.json()).message).toContain("sign-in link");
    expect(calls[0]!.path).toBe("/v1/auth/email-link");
    expect(JSON.parse(calls[0]!.init.body as string)).toEqual({ email: "a@b.co", turnstile_token: "cf-1" });
  });

  it("using the link puts the token pair only in cookies", async () => {
    const { up, calls } = fakeUpstream(() => jsonResponse(pair(3)));
    const res = await emailLinkSignIn(req("/api/auth/email-link-confirm", { body: { token: "tok-123456789" } }), up, null);
    expect(res.status).toBe(200);
    expect(await res.text()).not.toContain("rt-3");
    expect(setCookies(res).some((c) => c.startsWith(`${REFRESH_COOKIE}=rt-3`))).toBe(true);
    expect(JSON.parse(calls[0]!.init.body as string)).toEqual({ token: "tok-123456789", client: "web" });
  });

  it("passes an expired link's message, refuses cross-site posts and uninvited accounts", async () => {
    let { up } = fakeUpstream(() => jsonResponse({ detail: "This sign-in link is invalid, expired or already used. Ask for a new one." }, 400));
    const bad = await emailLinkSignIn(req("/api/auth/email-link-confirm", { body: { token: "old-123456789" } }), up, null);
    expect(bad.status).toBe(400);
    expect((await bad.json()).detail).toContain("already used");
    expect(setCookies(bad)).toEqual([]);
    expect((await emailLinkSignIn(req("/api/auth/email-link-confirm", { body: { token: "x-123456789" }, origin: "https://evil.example" }), up, null)).status).toBe(403);
    expect((await emailLinkSignIn(req("/api/auth/email-link-confirm", { body: {} }), up, null)).status).toBe(400);
    ({ up } = fakeUpstream((c) => (c.path === "/v1/me" ? jsonResponse({ email: "other@x.co" }) : c.path === "/v1/auth/logout" ? new Response(null, { status: 204 }) : jsonResponse(pair(4)))));
    const refused = await emailLinkSignIn(req("/api/auth/email-link-confirm", { body: { token: "tok-123456789" } }), up, new Set(["me@x.co"]));
    expect(refused.status).toBe(403);
    expect((await refused.json()).detail).toBe(NOT_INVITED);
  });
});

describe("sign out everywhere", () => {
  it("calls the API with this session and clears the cookies", async () => {
    const access = jwt(900);
    const { up, calls } = fakeUpstream(() => new Response(null, { status: 204 }));
    const res = await logoutEverywhere(req("/api/auth/logout-everywhere", { cookies: { [ACCESS_COOKIE]: access, [REFRESH_COOKIE]: "rt-1" } }), up);
    expect(res.status).toBe(204);
    expect(calls[0]!.path).toBe("/v1/me/sign-out-everywhere");
    expect((calls[0]!.init.headers as Record<string, string>).authorization).toBe(`Bearer ${access}`);
    expect(setCookies(res).every((c) => c.includes("Max-Age=0"))).toBe(true);
  });

  it("answers 401 without a session and refuses cross-site posts", async () => {
    const { up, calls } = fakeUpstream(() => new Response(null, { status: 204 }));
    expect((await logoutEverywhere(req("/api/auth/logout-everywhere"), up)).status).toBe(401);
    expect((await logoutEverywhere(req("/api/auth/logout-everywhere", { cookies: { [REFRESH_COOKIE]: "rt-1" }, origin: "https://evil.example" }), up)).status).toBe(403);
    expect(calls).toHaveLength(0);
  });
});
