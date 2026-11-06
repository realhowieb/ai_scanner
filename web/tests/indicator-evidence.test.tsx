import { render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { Schemas } from "@/api/client";
import { StockIndicatorEvidence } from "@/features/IndicatorEvidence";
import { clearApiCache } from "@/hooks/useApi";
import { stockDetail } from "./fixtures";
import { jsonResponse } from "./helpers";

let params = new URLSearchParams();
vi.mock("next/navigation", () => ({ useSearchParams: () => params }));
const fetchMock = vi.fn();
const id = "a".repeat(32);
const evidence: Schemas["IndicatorEvidence"] = {
  rule_version: "daily-indicators-v1", bar_date: "2026-11-05", rsi_14: 39.4, rsi_previous: 29,
  prior_5_rsi_low: 27.1, prior_5_rsi_low_date: "2026-11-03", close: 98.55, previous_close: 97.04,
  change_pct: 1.56, relative_volume: 5.6, sma50: 102,
  rebound_candidate: true, momentum_recovery: false, uptrend_pullback: false,
  macd_line: -1.1, macd_signal: -1.2, macd_histogram: .1, macd_previous_histogram: -.1,
  macd_bullish_cross: true, macd_above_signal: true,
};
function saved(ticker = "BG", e: Schemas["IndicatorEvidence"] | null = evidence) {
  return { scan_id: id, status: "complete", params: { rebound: true, macd_confirmation: "bullish_cross" }, started_at: "2026-11-05T21:00:00Z",
    result: { setups: [{ ticker, indicator_evidence: e }] } };
}
beforeEach(() => { clearApiCache(); params = new URLSearchParams(); fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

it("shows the exact saved scan separately from newer chart readings", async () => {
  params.set("scan", id);
  fetchMock.mockResolvedValue(jsonResponse(saved()));
  render(<StockIndicatorEvidence s={stockDetail({ ticker: "BG", indicator_evidence: { ...evidence, bar_date: "2026-11-06", rsi_14: 54, rebound_candidate: false } })} />);
  expect(await screen.findByText("Rebound candidate")).toBeInTheDocument();
  const region = screen.getByRole("region", { name: "Selected scan RSI and MACD evidence" });
  expect(within(region).getByText("27.1 on 2026-11-03")).toBeInTheDocument();
  expect(within(region).getByText("39.4")).toBeInTheDocument();
  expect(screen.getByText(/These readings are saved/)).toBeInTheDocument();
  expect(screen.getByText(/Latest chart-bar checks · 2026-11-06/)).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(new URL(fetchMock.mock.calls[0]![0].url).pathname).toContain(`/v1/scans/${id}`);
});

it("does not mistake another ticker's scan evidence for this stock", async () => {
  params.set("scan", id); fetchMock.mockResolvedValue(jsonResponse(saved("TAL")));
  render(<StockIndicatorEvidence s={stockDetail({ ticker: "BG" })} />);
  expect(await screen.findByText(/Saved evidence is unavailable/)).toBeInTheDocument();
  expect(screen.queryByText("Rebound candidate")).not.toBeInTheDocument();
});

it("handles an expired or unowned scan without inventing qualification", async () => {
  params.set("scan", id); fetchMock.mockResolvedValue(jsonResponse({ detail: "Scan not found" }, 404));
  render(<StockIndicatorEvidence s={stockDetail({ ticker: "BG" })} />);
  await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
  expect(screen.queryByText("Rebound candidate")).not.toBeInTheDocument();
});

it("does not request invalid scan identifiers and explains older evidence", () => {
  params.set("scan", "invalid");
  render(<StockIndicatorEvidence s={stockDetail({ ticker: "BG" })} />);
  expect(screen.getByText(/Saved evidence is unavailable/)).toBeInTheDocument();
  expect(fetchMock).not.toHaveBeenCalled();
});

it("shows missing history as needs data and makes no scan request for a direct visit", () => {
  render(<StockIndicatorEvidence s={stockDetail({ ticker: "BG", indicator_evidence: { ...evidence, rsi_14: null,
    rsi_previous: null, prior_5_rsi_low: null, sma50: null, relative_volume: null, macd_histogram: null,
    macd_previous_histogram: null, rebound_candidate: false, macd_bullish_cross: false, macd_above_signal: false } })} />);
  expect(screen.getAllByText("Needs data").length).toBeGreaterThan(0);
  expect(screen.getByText("Rebound not confirmed")).toBeInTheDocument();
  expect(fetchMock).not.toHaveBeenCalled();
});
