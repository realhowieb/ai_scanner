import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { waitFor } from "@testing-library/react";
import { fakeApi } from "./fakeApi";

const ph = vi.hoisted(() => ({ init: vi.fn(), capture: vi.fn(), reset: vi.fn(), opt_out_capturing: vi.fn(), opt_in_capturing: vi.fn() }));
vi.mock("posthog-js", () => ({ default: ph }));

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  localStorage.clear();
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_fixture_only");
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_HOST", "https://eu.i.posthog.com");
});
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); localStorage.clear(); });

it("does not initialize or send anything without a project key", async () => {
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "");
  const a = await import("@/lib/usageAnalytics");
  a.captureUsage("stock_viewed", { route: "/stocks/NVDA" });
  await Promise.resolve();
  expect(ph.init).not.toHaveBeenCalled();
  expect(ph.capture).not.toHaveBeenCalled();
});

it.each([{ doNotTrack: "1" }, { globalPrivacyControl: true }])("respects browser privacy settings: %s", async (navigator) => {
  vi.stubGlobal("navigator", navigator);
  const a = await import("@/lib/usageAnalytics");
  a.captureUsage("scanner_viewed");
  expect(a.usageAllowed()).toBe(false);
  expect(ph.init).not.toHaveBeenCalled();
});

it("initializes once with automatic collection disabled and only the selected region", async () => {
  const a = await import("@/lib/usageAnalytics");
  a.captureUsage("page_viewed", { route: "/scanner" });
  a.captureUsage("scanner_viewed");
  await waitFor(() => expect(ph.capture).toHaveBeenCalledTimes(2));
  expect(ph.init).toHaveBeenCalledTimes(1);
  expect(ph.init.mock.calls[0]![1]).toMatchObject({ api_host: "https://eu.i.posthog.com", autocapture: false,
    capture_pageview: false, disable_session_recording: true, person_profiles: "never", advanced_disable_flags: true,
    disable_external_dependency_loading: true, capture_exceptions: false, save_referrer: false, save_campaign_params: false });
});

it("strips SDK URLs, person properties, tokens in URLs and product data before sending", async () => {
  const { sanitizeUsage, usageRoute } = await import("@/lib/usageAnalytics");
  const event = { uuid: "test-event", event: "stock_viewed", $set: { email: "private@example.com" }, $set_once: { note: "private" }, properties: { distinct_id: "anonymous-123", route: "/stocks/NVDA?token=secret",
    $current_url: "https://example.com/reset-password?token=secret", $referrer: "https://example.com?email=private",
    email: "private@example.com", ticker: "NVDA", note: "private journal", $set: { email: "private@example.com" } } };
  const safe = sanitizeUsage(event);
  expect(safe).not.toHaveProperty("$set");
  expect(safe).not.toHaveProperty("$set_once");
  expect(safe?.properties).toEqual({ distinct_id: "anonymous-123", route: "/stocks/:ticker", $process_person_profile: false });
  expect(sanitizeUsage({ uuid: "test-event", event: "$autocapture", properties: { text: "private" } })).toBeNull();
  expect(usageRoute("/reset-password?token=secret")).toBeNull();
  expect(usageRoute("/admin/users/private@example.com")).toBeNull();
  expect(usageRoute("/scanner?query=private")).toBe("/scanner");
});

it("stops capture immediately on opt-out, and can resume without an automatic opt-in event", async () => {
  const a = await import("@/lib/usageAnalytics");
  a.captureUsage("page_viewed", { route: "/today" });
  await waitFor(() => expect(ph.capture).toHaveBeenCalledTimes(1));
  a.setUsageEnabled(false);
  expect(a.usageAllowed()).toBe(false);
  expect(ph.opt_out_capturing).toHaveBeenCalled();
  a.captureUsage("scanner_viewed");
  expect(ph.capture).toHaveBeenCalledTimes(1);
  const beforeSend = ph.init.mock.calls[0]![1].before_send;
  expect(beforeSend({ event: "page_viewed", properties: {} })).toBeNull();
  a.setUsageEnabled(true);
  expect(ph.opt_in_capturing).toHaveBeenCalledWith({ captureEventName: false });
  a.captureUsage("scanner_viewed");
  await waitFor(() => expect(ph.capture).toHaveBeenCalledTimes(2));
  a.resetUsageIdentity();
  expect(ph.reset).toHaveBeenCalledWith(true);
});

it("records successful actions without names or tickers and excludes rejected requests", async () => {
  const fake = fakeApi({ rules: true });
  vi.stubGlobal("fetch", fake.fetch);
  const { watchlists, alertRules } = await import("@/api/userData");
  const list = await watchlists.create("Private investing strategy");
  await watchlists.addTickers(list.id, ["NVDA"]);
  await alertRules.create({ rule_type: "HSF_SCORE_CROSS_ABOVE", ticker: "NVDA", threshold: 80, enabled: true });
  await waitFor(() => expect(ph.capture).toHaveBeenCalledTimes(3));
  expect(ph.capture.mock.calls.map(([event, properties]) => [event, properties])).toEqual([
    ["watchlist_created", {}], ["watchlist_stock_added", {}], ["alert_created", { kind: "hsf" }],
  ]);
  fake.fail("POST", /^\/v1\/watchlists$/, 403, { detail: "Limit reached" });
  await expect(watchlists.create("Do not send")).rejects.toThrow();
  expect(ph.capture).toHaveBeenCalledTimes(3);
});

it("a broken analytics SDK cannot reject a successful product action", async () => {
  ph.init.mockImplementationOnce(() => { throw new Error("Analytics unavailable"); });
  const fake = fakeApi();
  vi.stubGlobal("fetch", fake.fetch);
  const { watchlists } = await import("@/api/userData");
  await expect(watchlists.create("Main")).resolves.toMatchObject({ name: "Main" });
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(ph.capture).not.toHaveBeenCalled();
});

it("drops an action queued before sign-out and resets an SDK that finishes loading afterward", async () => {
  const a = await import("@/lib/usageAnalytics");
  a.captureUsage("watchlist_created");
  a.resetUsageIdentity();
  await waitFor(() => expect(ph.init).toHaveBeenCalled());
  expect(ph.reset).toHaveBeenCalledWith(true);
  expect(ph.capture).not.toHaveBeenCalled();
});
