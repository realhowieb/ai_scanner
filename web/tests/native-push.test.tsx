import { afterEach, describe, expect, it, vi } from "vitest";

import { isNativeApp, isStandalone } from "@/lib/homeScreen";
import { currentPushState, disablePush, enablePush } from "@/lib/webPush";

import { jsonResponse } from "./helpers";

type Handler = (arg: unknown) => void;

function fakeApp(permission: "granted" | "denied" | "prompt" = "granted") {
  const handlers: Record<string, Handler[]> = {};
  const push = {
    checkPermissions: vi.fn(async () => ({ receive: permission })),
    requestPermissions: vi.fn(async () => ({ receive: permission })),
    register: vi.fn(async () => { for (const h of handlers.registration ?? []) h({ value: "ab".repeat(32) }); }),
    unregister: vi.fn(async () => undefined),
    addListener: vi.fn(async (event: string, cb: Handler) => {
      (handlers[event] ??= []).push(cb);
      return { remove: async () => { handlers[event] = (handlers[event] ?? []).filter((h) => h !== cb); } };
    }),
  };
  (window as unknown as { Capacitor: unknown }).Capacitor = { isNativePlatform: () => true, Plugins: { PushNotifications: push } };
  return push;
}

afterEach(() => {
  delete (window as unknown as { Capacitor?: unknown }).Capacitor;
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("iPhone app notifications", () => {
  it("only counts as the app inside the native shell", () => {
    expect(isNativeApp()).toBe(false);
    fakeApp();
    expect(isNativeApp()).toBe(true);
    expect(isStandalone()).toBe(true);
  });

  it("registers the APNs token with the API, then removes it when turned off", async () => {
    const push = fakeApp();
    const calls: { url: string; method: string; body: unknown }[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const req = input instanceof Request ? input : new Request(String(input), init);
      const text = await req.text();
      calls.push({ url: req.url, method: req.method, body: text ? JSON.parse(text) : null });
      if (req.method === "DELETE") return new Response(null, { status: 204 });
      return jsonResponse({ id: 42, provider: "apns", platform: "ios", device_name: "HSF iPhone app", app_version: null, created_at: null, last_seen_at: null });
    });
    expect(await currentPushState()).toBe("off");
    expect(await enablePush("")).toBe("on");
    expect(calls[0]?.url).toMatch(/\/v1\/me\/devices$/);
    expect(calls[0]?.body).toEqual({ push_token: "ab".repeat(32), platform: "ios", provider: "apns", device_name: "HSF iPhone app" });
    expect(await currentPushState()).toBe("on");
    await disablePush();
    expect(calls[1]?.method).toBe("DELETE");
    expect(calls[1]?.url).toMatch(/\/v1\/me\/devices\/42$/);
    expect(push.unregister).toHaveBeenCalled();
    expect(await currentPushState()).toBe("off");
  });

  it("reports notifications blocked in iPhone Settings", async () => {
    fakeApp("denied");
    expect(await currentPushState()).toBe("denied");
    expect(await enablePush("")).toBe("denied");
  });
});
