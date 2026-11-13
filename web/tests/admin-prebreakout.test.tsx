import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import { PrebreakoutBody, type PrebreakoutSummary } from "@/features/AdminPrebreakout";

const data: PrebreakoutSummary = {
  model: { model_version: "pb-v1", trained_at: null, saved_at: null, dataset_version: "ds1", target_rule: "Future setup then economic outcome",
    features: 2, calibration_method: "isotonic", validation_auc: 0.62,
    training_buckets: [{ bucket: "40-50%", signals: 20, mean_confidence: 0.45, hit_rate: 0.4 }] },
  warnings: [], live_validation: "Five-day outcomes cannot verify this target.",
  latest: { scanned_at: "2026-11-12T20:00:00Z", rows: [
    { ticker: "AAA", probability: 20, hsf_score: 70, breakout_score: 7, setup: "Rebound", model_version: "pb-v1" },
    { ticker: "BBB", probability: 60, hsf_score: 80, breakout_score: 6, setup: "Recovery", model_version: "pb-v1" },
    { ticker: "MISSING", probability: null, hsf_score: null, breakout_score: null, setup: null, model_version: null },
  ] },
};
afterEach(() => vi.unstubAllGlobals());

it("sorts saved scores and distinguishes missing values and training evidence", async () => {
  render(<PrebreakoutBody d={data} />);
  const region = screen.getByRole("region", { name: "PreBreakout latest scan scores" });
  const names = () => within(region).getAllByRole("rowheader").map((e) => e.textContent);
  expect(names()).toEqual(["BBB", "AAA", "MISSING"]);
  await userEvent.click(within(region).getByRole("button", { name: "PreBreakout" }));
  expect(names()).toEqual(["AAA", "BBB", "MISSING"]);
  expect(within(region).getByText("Unavailable")).toBeInTheDocument();
  expect(screen.getByText("Five-day outcomes cannot verify this target.")).toBeInTheDocument();
  await userEvent.click(screen.getByText("Training-time validation calibration"));
  expect(screen.getByRole("table", { name: "PreBreakout training calibration" })).toHaveTextContent("40.0%");
});

it("loads inputs on demand for the exact displayed snapshot", async () => {
  const urls: string[] = [];
  vi.stubGlobal("fetch", vi.fn(async (request: Request | string) => {
    urls.push(typeof request === "string" ? request : request.url);
    return new Response(JSON.stringify({ ticker: "BBB", scanned_at: data.latest.scanned_at, model_version: "pb-v1",
      note: "Captured values are not feature contributions.",
      inputs: [{ name: "VolRel20", value: 0, defaulted: true }, { name: "RSI", value: 32, defaulted: false }] }),
    { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  render(<PrebreakoutBody d={data} />);
  expect(urls).toEqual([]);
  await userEvent.click(within(screen.getByRole("row", { name: /BBB/ })).getByRole("button", { name: "View saved inputs" }));
  expect(await screen.findByText("Defaulted")).toBeInTheDocument();
  expect(screen.getByText("Observed")).toBeInTheDocument();
  const url = new URL(urls[0] ?? "", "http://localhost");
  expect(url.pathname).toContain("/prebreakout/inputs/BBB");
  expect(url.searchParams.get("scanned_at")).toBe(data.latest.scanned_at);
});

it("shows missing model and scores explicitly", () => {
  render(<PrebreakoutBody d={{ model: null, warnings: [], latest: { scanned_at: null, rows: [] }, live_validation: "Not established" }} />);
  expect(screen.getByText("No active database model recorded")).toBeInTheDocument();
  expect(screen.getByText("No saved PreBreakout scores yet")).toBeInTheDocument();
});
