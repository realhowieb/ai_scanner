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

const calls: { path: string; method: string; body: unknown }[] = [];
beforeEach(() => {
  calls.length = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: Request | string, init?: RequestInit) => {
    const r = typeof input === "string" ? new Request(new URL(input, "http://localhost"), init) : input;
    const path = new URL(r.url).pathname.replace(/^\/api\/(hsf|public)/, "");
    const body = r.method === "GET" ? null : JSON.parse((await r.text()) || "null");
    calls.push({ path, method: r.method, body });
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
