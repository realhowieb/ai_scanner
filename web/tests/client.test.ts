import { describe, expect, it, vi } from "vitest";

import { ApiError, makeClient, setSessionExpiredHandler, unwrap } from "@/api/client";
import { safeNext } from "@/lib/nextPath";
import { parseRetryAfter } from "@/lib/retryAfter";
import { freshness } from "@/lib/format";

import { jsonResponse } from "./helpers";

describe("Retry-After", () => {
  it("reads seconds and HTTP dates, caps them, ignores junk", () => {
    expect(parseRetryAfter("30")).toBe(30);
    expect(parseRetryAfter("99999")).toBe(3600);
    const now = Date.parse("2026-10-06T12:00:00Z");
    expect(parseRetryAfter("Tue, 06 Oct 2026 12:00:45 GMT", now)).toBe(45);
    expect(parseRetryAfter("soon")).toBeNull();
    expect(parseRetryAfter(null)).toBeNull();
  });
});

describe("typed client", () => {
  it("sends a request id and turns errors into ApiError with Retry-After and the support id", async () => {
    const seen: Request[] = [];
    const fetchImpl = vi.fn(async (input: Request) => {
      seen.push(input);
      return jsonResponse({ detail: "The scanner is busy." }, 503, { "retry-after": "20", "x-request-id": "web-abcdef12" });
    });
    const c = makeClient(fetchImpl as unknown as typeof fetch);
    const err = await unwrap(c.GET("/v1/today")).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 503, retryAfterS: 20, requestId: "web-abcdef12", message: "The scanner is busy." });
    expect(seen[0]!.headers.get("x-request-id")).toMatch(/^web-[0-9a-f-]{36}$/);
    expect(new URL(seen[0]!.url).pathname).toBe("/api/hsf/v1/today");
  });

  it("calls the session-expired handler once, however many requests fail at the same time", async () => {
    const onExpired = vi.fn();
    setSessionExpiredHandler(onExpired);
    const c = makeClient((async () => jsonResponse({ detail: "ended", code: "session_expired" }, 401)) as unknown as typeof fetch);
    const errs = await Promise.all([1, 2, 3].map(() => unwrap(c.GET("/v1/me")).catch((e) => e)));
    expect(errs.every((e) => e instanceof ApiError && e.code === "session_expired")).toBe(true);
    expect(onExpired).toHaveBeenCalledTimes(1);
  });

  it("reports a network failure as status 0", async () => {
    const c = makeClient((async () => { throw new TypeError("offline"); }) as unknown as typeof fetch);
    await expect(unwrap(c.GET("/v1/me"))).rejects.toMatchObject({ status: 0 });
  });
});

describe("helpers", () => {
  it("only allows same-site paths after sign-in", () => {
    expect(safeNext("/scanner?signal=breakout")).toBe("/scanner?signal=breakout");
    for (const bad of ["https://evil.example", "//evil.example", "/\\evil", "/api/hsf/v1/me", null, ""]) expect(safeNext(bad)).toBe("/today");
  });

  it("marks scans older than 6 hours stale", () => {
    const now = Date.parse("2026-10-06T18:00:00Z");
    expect(freshness("2026-10-06T17:30:00Z", now)).toEqual({ label: "Updated 30m ago", stale: false });
    expect(freshness("2026-10-06T10:00:00Z", now).stale).toBe(true);
    expect(freshness(null, now).stale).toBe(true);
  });
});
