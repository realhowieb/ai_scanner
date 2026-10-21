import { afterEach, beforeEach, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ provider: vi.fn(), emit: vi.fn(), flush: vi.fn(), exporter: vi.fn() }));
vi.mock("@opentelemetry/sdk-logs", () => ({
  LoggerProvider: class { constructor(options: unknown) { mocks.provider(options); } getLogger() { return { emit: mocks.emit }; } forceFlush() { return mocks.flush(); } },
  BatchLogRecordProcessor: class { constructor(public options: unknown) {} },
}));
vi.mock("@opentelemetry/exporter-logs-otlp-http", () => ({ OTLPLogExporter: class { constructor(options: unknown) { mocks.exporter(options); } } }));
beforeEach(() => { vi.resetModules(); vi.resetAllMocks(); });
afterEach(() => vi.unstubAllEnvs());

it.each(["", "0"])("does not create an exporter when logging is disabled (%s)", async (enabled) => {
  vi.stubEnv("POSTHOG_LOGS_ENABLED", enabled);
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "fixture");
  expect((await import("../instrumentation")).posthogLogProvider).toBeNull();
  expect(mocks.exporter).not.toHaveBeenCalled();
});
it("does not require analytics keys to start the app", async () => {
  vi.stubEnv("POSTHOG_LOGS_ENABLED", "1");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "");
  expect((await import("../instrumentation")).posthogLogProvider).toBeNull();
});
it("exports bounded method/status logs to the selected region and tolerates failures", async () => {
  vi.stubEnv("POSTHOG_LOGS_ENABLED", "1");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "fixture");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", "https://eu.i.posthog.com");
  const logs = await import("@/server/posthogLogs");
  expect(logs.posthogLogsEnabled()).toBe(true);
  expect(mocks.exporter).toHaveBeenCalledWith(expect.objectContaining({ url: "https://eu.i.posthog.com/i/v1/logs" }));
  logs.logProxyRequestStarted("GET");
  logs.logProxyResponse("GET", 503);
  expect(mocks.emit.mock.calls.map(([record]) => record.attributes)).toEqual([
    { "http.request.method": "GET" }, { "http.request.method": "GET", "http.response.status_code": 503 },
  ]);
  mocks.emit.mockImplementation(() => { throw new Error("unavailable"); });
  mocks.flush.mockRejectedValue(new Error("unavailable"));
  expect(() => logs.logProxyRequestStarted("GET")).not.toThrow();
  expect(() => logs.logProxyResponse("GET", 200)).not.toThrow();
  await expect(logs.flushPostHogLogs()).resolves.toBeUndefined();
});
