import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GET as healthz } from "@/app/api/healthz/route";
import { Skeleton, WAKING_UP } from "@/components/ui";
import { NOT_INVITED, betaAllowlist, login } from "@/server/auth";
import type { Upstream } from "@/server/bff";

import { jsonResponse, jwt, setCookies } from "./helpers";

const HOST = "beta.example.test";
const loginReq = (email: string) => new Request(`https://${HOST}/api/auth/login`, {
  method: "POST", headers: { host: HOST, origin: `https://${HOST}`, "content-type": "application/json" },
  body: JSON.stringify({ email, password: "pw" }),
});

function upstreamFor(accountEmail: string) {
  const calls: string[] = [];
  const up: Upstream = async (path, init) => {
    calls.push(path);
    if (path === "/v1/auth/login") return jsonResponse({ access_token: jwt(900), refresh_token: "rt-1", expires_in: 900, token_type: "bearer" });
    if (path === "/v1/me") {
      expect((init.headers as Record<string, string>).authorization).toMatch(/^Bearer /);
      return jsonResponse({ email: accountEmail, plan: "pro" });
    }
    if (path === "/v1/auth/logout") return new Response(null, { status: 204 });
    return jsonResponse({}, 404);
  };
  return { up, calls };
}

describe("invite-only beta (WEB_BETA_ALLOWED_EMAILS)", () => {
  it("parses a comma-separated list; unset or blank means everyone", () => {
    expect(betaAllowlist(" A@x.com, b@y.com ,,")).toEqual(new Set(["a@x.com", "b@y.com"]));
    expect(betaAllowlist(undefined)).toBeNull();
    expect(betaAllowlist(" , ")).toBeNull();
  });

  it("lets an invited account in (matched on the API's email, case-insensitive)", async () => {
    const { up, calls } = upstreamFor("Invited@Example.com");
    const res = await login(loginReq("invited-username"), up, new Set(["invited@example.com"]));
    expect(res.status).toBe(200);
    expect(setCookies(res).some((c) => c.includes("hsf_rt=rt-1"))).toBe(true);
    expect(calls).toEqual(["/v1/auth/login", "/v1/me"]);
  });

  it("refuses an account not on the list, revokes the new session and sets no cookies", async () => {
    const { up, calls } = upstreamFor("stranger@example.com");
    const res = await login(loginReq("stranger@example.com"), up, new Set(["invited@example.com"]));
    expect(res.status).toBe(403);
    expect(await res.json()).toEqual({ detail: NOT_INVITED, code: "not_invited" });
    expect(setCookies(res)).toEqual([]);
    expect(calls).toEqual(["/v1/auth/login", "/v1/me", "/v1/auth/logout"]);
  });

  it("without a list, sign-in doesn't call /v1/me", async () => {
    const { up, calls } = upstreamFor("anyone@example.com");
    expect((await login(loginReq("anyone@example.com"), up, null)).status).toBe(200);
    expect(calls).toEqual(["/v1/auth/login"]);
  });
});

describe("health check", () => {
  it("answers 200 without touching the API", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const res = healthz();
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true });
    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });
});

describe("cold starts", () => {
  afterEach(() => vi.useRealTimers());
  it("loading states explain a slow wake-up after 8 s", () => {
    vi.useFakeTimers();
    render(<Skeleton />);
    expect(screen.queryByText(WAKING_UP)).not.toBeInTheDocument();
    act(() => { vi.advanceTimersByTime(8000); });
    expect(screen.getByText(WAKING_UP)).toBeInTheDocument();
  });
});
