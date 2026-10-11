import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PushToggle } from "@/features/AlertsView";

import { jsonResponse } from "./helpers";

// Any uncompressed P-256 point works here: the fake push manager never checks it.
const SERVER_POINT = "B" + "A".repeat(86);
const calls: { method: string; path: string; body: unknown }[] = [];
let enabled = true;
let allowed = true;
let sub: { endpoint: string; toJSON: () => unknown; unsubscribe: () => Promise<boolean> } | null = null;
const subscribe = vi.fn();

beforeEach(() => {
  calls.length = 0;
  enabled = true;
  allowed = true;
  sub = null;
  subscribe.mockReset().mockImplementation(async () => {
    sub = {
      endpoint: "https://push.example/abc",
      toJSON: () => ({ endpoint: "https://push.example/abc", keys: { p256dh: "p".repeat(87), auth: "a".repeat(22) } }),
      unsubscribe: async () => { sub = null; return true; },
    };
    return sub;
  });
  const reg = { pushManager: { getSubscription: async () => sub, subscribe } };
  vi.stubGlobal("PushManager", function PushManager() {});
  vi.stubGlobal("Notification", Object.assign(function Notification() {}, { permission: "default", requestPermission: vi.fn(async () => "granted") }));
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: { getRegistration: async () => reg, register: async () => reg, ready: Promise.resolve(reg) },
  });
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string, init?: RequestInit) => {
    const req = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const path = new URL(req.url).pathname.replace(/^\/api\/hsf/, "");
    const text = req.method === "GET" ? "" : await req.text();
    calls.push({ method: req.method, path, body: text ? JSON.parse(text) : null });
    if (path === "/v1/web-push/config") return jsonResponse({ enabled, allowed, public_key: enabled && allowed ? SERVER_POINT : null });
    if (path === "/v1/me/web-push" && req.method === "POST") return jsonResponse({ id: 1, provider: "webpush", platform: "web" });
    if (path === "/v1/me/web-push" && req.method === "DELETE") return new Response(null, { status: 204 });
    return jsonResponse({ detail: "no fake" }, 404);
  }));
});
afterEach(() => {
  vi.unstubAllGlobals();
  Reflect.deleteProperty(navigator, "serviceWorker");
});

describe("browser notifications toggle", () => {
  it("is hidden until the server has push keys", async () => {
    enabled = false;
    const { container } = render(<PushToggle />);
    await waitFor(() => expect(calls.some((c) => c.path === "/v1/web-push/config")).toBe(true));
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the Pro note instead of the toggle on Free", async () => {
    allowed = false;
    render(<PushToggle />);
    expect(await screen.findByText("Browser notifications are part of Pro.")).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("subscribes with the server key and registers the browser, then turns off again", async () => {
    const u = userEvent.setup();
    render(<PushToggle />);
    const box = await screen.findByRole("checkbox", { name: /Notify me in this browser/ });
    expect(box).not.toBeChecked();
    await u.click(box);
    await waitFor(() => expect(box).toBeChecked());
    expect(subscribe).toHaveBeenCalledWith(expect.objectContaining({ userVisibleOnly: true }));
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.path).toBe("/v1/me/web-push");
    expect(post.body).toMatchObject({ endpoint: "https://push.example/abc", keys: { auth: "a".repeat(22) } });
    await u.click(box);
    await waitFor(() => expect(box).not.toBeChecked());
    expect(calls.find((c) => c.method === "DELETE")!.body).toEqual({ endpoint: "https://push.example/abc" });
    expect(sub).toBeNull();
  });
});
