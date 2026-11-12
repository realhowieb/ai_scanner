import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ModelsTab } from "@/features/AdminModels";

import { jsonResponse } from "./helpers";

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/admin",
}));

const inputHealth = {
  status: "WATCH", sample_limit: 500, lookback_days: 7, sampled_predictions: 375,
  comparable_predictions: 375, unknown_predictions: 0, other_identity_predictions: 0,
  latest_prediction_at: "2026-11-06T17:43:00Z", model_version: "prebreakout-actual",
  input_contract: "public_snapshot_observations_v1", feature_count: 75,
  latest_real_inputs: 17, latest_defaulted_inputs: 58,
  features: [{ name: "LowSlope10D", defaulted_predictions: 375, predictions: 375, defaulted_pct: 100 }],
  note: "Input completeness only, not accuracy or calibration.",
};

const shadowModel = {
  id: 2, model_version: "ml-v4-lightgbm-production-f5d-000000000002", provider: "lightgbm", kind: "model",
  feature_set: "production", target_label: "F_clean_path_5d", horizon_days: 5, dataset_version: "hsf-ml-2026-11-13-v1",
  status: "shadow", status_changed_at: "2026-11-14T15:00:00Z", trained_at: null, created_at: null, has_artifact: true,
  artifact_sha256: "a".repeat(64), calibration: { method: "sigmoid" },
  metrics: { evaluation_version: "hsf-ml-v4-eval-1", walk_forward_raw: { n: 300, roc_auc: 0.61, brier: 0.2, ece: 0.04 },
    holdout: { raw: { n: 120, roc_auc: 0.58, brier: 0.21, ece: 0.05 } } },
};
const candidate = { ...shadowModel, id: 1, model_version: "ml-v4-xgboost-production-f5d-000000000001", provider: "xgboost", status: "candidate" };
const gates = { gates_version: "hsf-ml-v4-gates-1", passed: false, checks: [
  { gate: "labeled_shadow_rows", value: 40, threshold: ">= 200", passed: false },
  { gate: "evaluation_attached", value: "hsf-ml-v4-eval-1", threshold: "present", passed: true }] };
const shadow = { scored: 60, labeled: 40, entry_days: 8, last_scored_at: "2026-11-24T21:41:00Z", roc_auc: 0.6, hit_rate: 0.3,
  vs_ai_confidence_as_served: { rows: 40, model_auc: 0.6, served_auc: 0.55, auc_delta: 0.05, brier_delta: -0.01 } };

const aiConfidence = {
  model: { model_version: "ai-confidence-xgb-v7", trained_at: "2026-09-09T20:55:40Z", saved_at: "2026-09-09T21:00:00Z", dataset_version: null,
    target_rule: "path_first_hit_v1: +4% before -2% in 5 trading days", features: 6, calibration_method: null, calibration_points: 0,
    training_buckets: [{ bucket: "20-30%", signals: 120, mean_confidence: 0.25, hit_rate: 0.22 }] },
  warnings: ["The live model has no calibration map, so the shown percentages are raw model scores and run high."],
  latest: { scanned_at: "2026-11-12T19:35:00Z", tiers: { High: 1, Medium: 0, Low: 1 }, rows: [
    { ticker: "AMD", ai_confidence: 12.5, tier: "Low", breakout_score: 61.2, hsf_score: 70, setup: "Gapper", chg_pct: 3.1 },
    { ticker: "NVDA", ai_confidence: 46.2, tier: "High", breakout_score: 80.4, hsf_score: 88, setup: "Breakout", chg_pct: 4.2 }] },
  live: { days: 60, matured: 50, pending: 12, buckets: [
    { bucket: "40-50%", signals: 50, mean_confidence: 44.1, hits: 15, misses: 30, unclear: 5, hit_rate_low: 30, hit_rate_high: 40 }] },
  note: "Training buckets come from the validation set.",
};

let prebreakoutState: "available" | "missing" | "error" = "available";
const calls: { path: string; method: string; body: unknown }[] = [];
beforeEach(() => {
  calls.length = 0; prebreakoutState = "available";
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const path = new URL(r.url).pathname.replace(/^\/api\/(hsf|public)/, "");
    const body = r.method === "GET" ? null : JSON.parse((await r.text()) || "null");
    calls.push({ path, method: r.method, body });
    if (path === "/v1/admin/ml/production/prebreakout") return prebreakoutState === "error" ? jsonResponse({ detail: "Model evidence unavailable" }, 503) : jsonResponse({
      input_health: prebreakoutState === "missing" ? { ...inputHealth, status: "UNKNOWN", latest_real_inputs: null, latest_defaulted_inputs: null, features: [] } : inputHealth,
      configured_model: prebreakoutState === "missing" ? null : { model_version: "prebreakout-configured", trained_at: "2026-10-07T20:00:00Z", dataset_version: null },
      latest_capture: prebreakoutState === "missing" ? null : { model_version: "prebreakout-actual", signal_at: "2026-11-06T17:43:00Z", inferred_at: "2026-11-06T17:43:01Z", artifact_sha256: "pb-hash" },
    });
    if (path === "/v1/admin/ml/ai-confidence") return jsonResponse(aiConfidence);
    if (path === "/v1/admin/ml/ai-confidence/explain") return jsonResponse({ ticker: "NVDA", ai_confidence: 46.2, tier: "High", earnings_in_days: 2,
      features: { "Trend10D%": 8.1, VolRel20: 2.4 }, explanation: "Strong 10-day trend and heavy volume lift the estimate.", error: null });
    if (path === "/v1/admin/ml/production/health") return jsonResponse({
      generated_at: "2026-11-06T20:00:00Z", since: "2026-10-30T20:00:00Z",
      note: "Capture completeness and identity only, not predictive quality.",
      models: [
        { role: "ai_confidence", status: "WATCH", observations: 350, predictions: 350, missing_predictions: 0, identity_missing: 300, latest_prediction_at: "2026-11-06T17:43:00Z", behind_latest_observation: false },
        { role: "prebreakout", status: "HEALTHY", observations: 350, predictions: 350, missing_predictions: 0, identity_missing: 0, latest_prediction_at: "2026-11-06T17:43:00Z", behind_latest_observation: false },
      ],
    });
    if (path === "/v1/admin/ml/production") return jsonResponse({
      configured_model: { model_version: "production-xgb", trained_at: "2026-09-09T20:55:40Z", dataset_version: null },
      latest_capture: { model_version: "captured-local-xgb", signal_at: "2026-11-05T17:00:00Z", inferred_at: null, artifact_sha256: null },
    });
    if (path === "/v1/admin/ml/models") return jsonResponse({ models: [shadowModel, candidate], count: 2 });
    if (path === "/v1/admin/ml/models/1") return jsonResponse({ model: candidate, events: [{ id: 1, from_status: null, to_status: "candidate", actor: "ci", reason: "registered after training", created_at: "2026-11-13T20:00:00Z" }], shadow: { scored: 0, labeled: 0, entry_days: 0, last_scored_at: null }, promotion_gates: null });
    if (path === "/v1/admin/ml/models/1/transition") return jsonResponse({ id: 1, from: "candidate", to: "validated", superseded: null });
    if (path === "/v1/admin/ml/shadow") return jsonResponse({ models: [{ model: shadowModel, shadow, promotion_gates: gates }], days: 30,
      disagreement: [{ target_label: "F_clean_path_5d", horizon_days: 5, models: 2, rows: 30, mean_range: 0.06, share_range_over_0_10: 0.2, note: "agreement is not independent confirmation" }] });
    if (path === "/v1/admin/ml/monitoring") return jsonResponse({ status: "WATCH", generated_at: "2026-11-24T22:00:00Z", models: [{ model_id: 2, model_version: shadowModel.model_version, status: "WATCH",
      checks: [{ check: "probability_drift", status: "WATCH", value: 0.12, detail: "PSI of the last 10 days vs the first 200 shadow predictions" }] }] });
    return jsonResponse({});
  }));
});
afterEach(() => vi.unstubAllGlobals());

describe("Admin > Models", () => {
  it("restores the AI Confidence card: model status, sortable scores, calibration and an on-demand explainer", async () => {
    render(<ModelsTab />);
    const card = (await screen.findByRole("heading", { name: "AI Confidence" })).closest("section")!;
    expect(await within(card).findByText("ai-confidence-xgb-v7")).toBeInTheDocument();
    expect(within(card).getByText("Not calibrated")).toBeInTheDocument();
    expect(within(card).getByText(/no calibration map/)).toBeInTheDocument();
    const scores = within(card).getByRole("region", { name: "Latest scan scores" });
    const tickers = () => within(scores).getAllByRole("rowheader").map((c) => c.textContent);
    expect(tickers()).toEqual(["NVDA", "AMD"]);
    await userEvent.click(within(scores).getByRole("button", { name: "AI Confidence" }));
    expect(tickers()).toEqual(["AMD", "NVDA"]);
    expect(within(card).getByText("30.0% – 40.0%")).toBeInTheDocument();
    expect(calls.some((c) => c.path === "/v1/admin/ml/ai-confidence/explain")).toBe(false);
    await userEvent.click(within(scores).getAllByRole("button", { name: "Why this score?" })[1]);
    await userEvent.click(within(scores).getByRole("button", { name: "Explain NVDA with Claude" }));
    expect(await within(scores).findByText(/Strong 10-day trend/)).toBeInTheDocument();
    expect(calls.find((c) => c.path === "/v1/admin/ml/ai-confidence/explain")?.body).toEqual({ ticker: "NVDA" });
  });
  it("shows monitoring, shadow results against AI Confidence and the registry", async () => {
    render(<ModelsTab />);
    expect(screen.getByText(/AI Confidence keeps using the production model/)).toBeInTheDocument();
    expect(await screen.findByText(/PSI of the last 10 days/)).toBeInTheDocument();
    const shadowCard = (await screen.findByRole("heading", { name: "Shadow test" })).closest("section")!;
    expect(await within(shadowCard).findByText("+0.050")).toBeInTheDocument();
    expect(within(shadowCard).getByText("1/2")).toBeInTheDocument();
    expect(within(shadowCard).getByText(/not independent confirmation/)).toBeInTheDocument();
    const registry = screen.getByRole("heading", { name: "Model registry" }).closest("section")!;
    expect(await within(registry).findByText("2 models")).toBeInTheDocument();
    expect(registry).not.toHaveTextContent("github-release://");
  });

  it("distinguishes production configuration, captured identity and unregistered challengers", async () => {
    render(<ModelsTab />);
    const summary = screen.getByRole("heading", { name: "What is trained and running?" }).closest("section")!;
    expect(await within(summary).findByText("production-xgb")).toBeInTheDocument();
    expect(await within(summary).findByText(/captured-local-xgb/)).toBeInTheDocument();
    expect(await within(summary).findByText(/An unregistered workflow artifact may still exist/)).toBeInTheDocument();
    expect(within(summary).getAllByText(/Not connected to production AI Confidence/)).toHaveLength(2);
    expect(screen.getByText(/An older saved dataset keeps its original gaps/)).toBeInTheDocument();
    expect(calls.filter((c) => c.method !== "GET")).toHaveLength(0);
  });

  it("moves a model to its next stage with a reason", async () => {
    render(<ModelsTab />);
    await userEvent.click(await screen.findByRole("button", { name: `Open ${candidate.model_version}` }));
    const dialog = await screen.findByRole("dialog");
    expect(await within(dialog).findByText(/registered after training/)).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "Make champion" })).toBeNull();
    await userEvent.click(within(dialog).getByRole("button", { name: "Mark validated" }));
    await userEvent.type(within(dialog).getByLabelText(/Reason/), "walk-forward and holdout reviewed");
    await userEvent.click(within(dialog).getByRole("button", { name: "Confirm: mark validated" }));
    const post = calls.find((c) => c.method === "POST");
    expect(post?.path).toBe("/v1/admin/ml/models/1/transition");
    expect(post?.body).toEqual({ to_status: "validated", reason: "walk-forward and holdout reviewed", approve: false });
    expect(await within(dialog).findByRole("status")).toHaveTextContent("Moved to validated.");
  });
});


it("shows separate PreBreakout configuration and actual prediction evidence", async () => {
  render(<ModelsTab />);
  const card = screen.getByRole("region", { name: "Production PreBreakout" });
  expect(await within(card).findByText("prebreakout-configured")).toBeInTheDocument();
  expect(within(card).getAllByText(/prebreakout-actual/)).toHaveLength(2);
  expect(within(card).getByText("pb-hash")).toBeInTheDocument();
  expect(card).not.toHaveTextContent("production-xgb");
  expect(card).toHaveTextContent(/do not prove which artifact is currently cached/);
  expect(calls.filter((c) => c.method !== "GET")).toHaveLength(0);
});

it("labels missing PreBreakout evidence without claiming a running model", async () => {
  prebreakoutState = "missing";
  render(<ModelsTab />);
  const card = screen.getByRole("region", { name: "Production PreBreakout" });
  expect(await within(card).findByText("No active database model recorded")).toBeInTheDocument();
  expect(card).toHaveTextContent("No saved PreBreakout prediction with model evidence found");
});

it("keeps PreBreakout failures explicit while AI Confidence still loads", async () => {
  prebreakoutState = "error";
  render(<ModelsTab />);
  const card = screen.getByRole("region", { name: "Production PreBreakout" });
  expect(await within(card).findByText(/Model evidence unavailable/)).toBeInTheDocument();
  expect(await screen.findByText("production-xgb")).toBeInTheDocument();
  expect(card).not.toHaveTextContent("No active database model recorded");
});


it("separates missing predictions from missing identity without claiming model quality", async () => {
  render(<ModelsTab />);
  const card = screen.getByRole("heading", { name: "Production prediction capture" }).closest("section")!;
  expect(await within(card).findByText("300")).toBeInTheDocument();
  expect(within(card).getAllByText("350 / 350")).toHaveLength(2);
  expect(within(card).getByText("Healthy")).toBeInTheDocument();
  expect(within(card).getByText(/not predictive quality/)).toBeInTheDocument();
  expect(calls.filter((c) => c.method !== "GET")).toHaveLength(0);
});


it("shows input defaults separately from successfully captured predictions", async () => {
  render(<ModelsTab />);
  const health = await screen.findByRole("region", { name: "PreBreakout input health" });
  expect(health).toHaveTextContent("17 of 75 inputs have recorded values; 58 filled with defaults");
  expect(health).toHaveTextContent("public_snapshot_observations_v1");
  expect(health).toHaveTextContent("not accuracy or calibration");
  await userEvent.click(within(health).getByText("Inputs filled with defaults (1)"));
  expect(within(health).getByText("LowSlope10D")).toBeVisible();
  expect(within(health).getByText("100%")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "Refresh PreBreakout health" }));
  expect(calls.filter((c) => c.path === "/v1/admin/ml/production/prebreakout").length).toBeGreaterThan(1);
  expect(calls.filter((c) => c.method !== "GET")).toHaveLength(0);
});

it("shows unknown input evidence without inventing complete coverage", async () => {
  prebreakoutState = "missing";
  render(<ModelsTab />);
  const health = await screen.findByRole("region", { name: "PreBreakout input health" });
  expect(health).toHaveTextContent("Latest input evidence is unavailable or incomplete");
  expect(health).not.toHaveTextContent("inputs have recorded values");
});
