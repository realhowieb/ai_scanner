// @vitest-environment node
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { expect, it, vi } from "vitest";

const source = readFileSync(new URL("../public/sw.js", import.meta.url), "utf8");
function worker() {
  const handlers: Record<string, (event: unknown) => void> = {};
  const showNotification = vi.fn(async () => undefined);
  const openWindow = vi.fn(async () => undefined);
  vm.runInNewContext(source, { URL, self: {
    location: { origin: "https://app.example.test" },
    addEventListener: (name: string, handler: (event: unknown) => void) => { handlers[name] = handler; },
    registration: { showNotification }, clients: { matchAll: async () => [], openWindow },
  } });
  return { handlers, showNotification, openWindow };
}
it.each(["//evil.example/path", "/\\evil.example/path", "https://evil.example/path", "javascript:alert(1)"])("keeps push/click target inside the app: %s", async (url) => {
  const w = worker();
  let done: Promise<unknown> | undefined;
  const waitUntil = (p: Promise<unknown>) => { done = p; };
  w.handlers.push!({ data: { json: () => ({ url }) }, waitUntil });
  await done;
  expect(w.showNotification).toHaveBeenCalledWith("HSF alert", expect.objectContaining({ data: { url: "/alerts" } }));
  w.handlers.notificationclick!({ notification: { close: vi.fn(), data: { url } }, waitUntil });
  await done;
  expect(w.openWindow).toHaveBeenCalledWith("https://app.example.test/alerts");
});
it("accepts an internal alert target and handles empty payloads", async () => {
  const w = worker();
  let done: Promise<unknown> | undefined;
  const waitUntil = (p: Promise<unknown>) => { done = p; };
  w.handlers.push!({ data: { json: () => ({ url: "/alerts?view=activity" }) }, waitUntil });
  await done;
  expect(w.showNotification).toHaveBeenLastCalledWith("HSF alert", expect.objectContaining({ data: { url: "/alerts?view=activity" } }));
  w.handlers.push!({ data: { json: () => null }, waitUntil });
  await done;
  expect(w.showNotification).toHaveBeenLastCalledWith("HSF alert", expect.objectContaining({ data: { url: "/alerts" } }));
});
