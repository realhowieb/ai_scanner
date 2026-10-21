// @vitest-environment node
import { beforeEach, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ proxy: vi.fn(), after: vi.fn(), enabled: vi.fn(), flush: vi.fn() }));
vi.mock("next/server", () => ({ after: mocks.after }));
vi.mock("@/server/bff", () => ({ proxy: mocks.proxy }));
vi.mock("@/server/upstream", () => ({ upstreamFor: () => "fixture" }));
vi.mock("@/server/posthogLogs", () => ({
  posthogLogsEnabled: mocks.enabled, flushPostHogLogs: mocks.flush,
  logProxyRequestStarted: vi.fn(), logProxyResponse: vi.fn(),
}));
beforeEach(() => vi.resetAllMocks());

it("returns the unchanged proxy response without scheduling disabled logs", async () => {
  const response = new Response("fixture", { status: 202 });
  mocks.proxy.mockResolvedValue(response);
  mocks.enabled.mockReturnValue(false);
  const { GET } = await import("@/app/api/hsf/[...path]/route");
  const req = new Request("https://app.example.test/api/hsf/v1/me");
  expect(await GET(req, { params: Promise.resolve({ path: ["v1", "me"] }) })).toBe(response);
  expect(mocks.proxy).toHaveBeenCalledWith(req, "v1/me", "fixture");
  expect(mocks.after).not.toHaveBeenCalled();
});
it("preserves the API response when scheduling a log flush fails", async () => {
  const response = new Response(null, { status: 204 });
  mocks.proxy.mockResolvedValue(response);
  mocks.enabled.mockReturnValue(true);
  mocks.after.mockImplementation(() => { throw new Error("unavailable"); });
  const { DELETE } = await import("@/app/api/hsf/[...path]/route");
  expect(await DELETE(new Request("https://app.example.test/api/hsf/fixture", { method: "DELETE" }),
    { params: Promise.resolve({ path: ["fixture"] }) })).toBe(response);
  expect(mocks.after).toHaveBeenCalledWith(mocks.flush);
});
