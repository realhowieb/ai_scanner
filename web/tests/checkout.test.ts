import { afterEach, expect, it, vi } from "vitest";

import { account } from "@/api/account";
import { checkout } from "@/lib/checkout";

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it("refuses a second window's checkout while the browser lock is held", async () => {
  const send = vi.spyOn(account, "checkout");
  vi.stubGlobal("navigator", { locks: { request: async (_name: string, _options: unknown, callback: (lock: null) => unknown) => callback(null) } });
  await expect(checkout("premium", "year")).rejects.toMatchObject({ status: 409 });
  expect(send).not.toHaveBeenCalled();
});

it("passes the selected interval through the lock and supports browsers without Web Locks", async () => {
  const send = vi.spyOn(account, "checkout").mockResolvedValue({ url: "https://checkout.stripe.com/session", mode: "checkout" });
  const request = vi.fn(async (_name: string, _options: unknown, callback: (lock: object) => unknown) => callback({ name: "hsf-checkout" }));
  vi.stubGlobal("navigator", { locks: { request } });
  await checkout("pro", "year");
  expect(request).toHaveBeenCalledWith("hsf-checkout", { ifAvailable: true }, expect.any(Function));
  expect(send).toHaveBeenLastCalledWith("pro", "year");
  vi.stubGlobal("navigator", {});
  await checkout("premium", "month");
  expect(send).toHaveBeenLastCalledWith("premium", "month");
});
