import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import { ApiError } from "@/api/client";
import type { Schemas } from "@/api/client";
import { ScannerDataStatus } from "@/features/ScannerDataStatus";

const info: Schemas["DataFreshness"] = {
  state: "fresh", scan_state: "fresh", market_data_state: "fresh",
  last_successful_scan_at: "2026-10-16T16:44:00Z", scan_completed_at: "2026-10-16T16:44:00Z",
  market_data_at: "2026-10-16T16:36:00Z", checked_at: "2026-10-16T19:16:00Z",
  expected_scan_at: "2026-10-16T16:35:00Z", calendar_covered: true, timestamp_basis: "saved_scan",
};
const base = { checkedAt: Date.parse(info.checked_at), refreshing: false, error: null, reload: () => {} };

it("distinguishes an on-schedule scan from live prices and collapses diagnostic timestamps", () => {
  render(<ScannerDataStatus {...base} info={info} />);
  expect(screen.getByText("On schedule")).toBeInTheDocument();
  expect(screen.getByText(/Snapshot prices as of/)).toHaveTextContent("12:36 PM ET · 2h 40m old");
  expect(screen.queryByText("Data freshness: Fresh")).not.toBeInTheDocument();
  const details = screen.getByText("Details").closest("details")!;
  expect(details).not.toHaveAttribute("open");
  expect(screen.getByText(/Last checked for a newer scan/)).not.toBeVisible();
  fireEvent.click(screen.getByText("Details"));
  expect(details).toHaveAttribute("open");
  expect(screen.getByText(/It does not mean the prices are live/)).toBeVisible();
});

it("keeps an overdue-scan warning visible even with details closed", () => {
  render(<ScannerDataStatus {...base} info={{ ...info, state: "stale" }} />);
  expect(screen.getByText("Scan delayed")).toBeVisible();
  expect(screen.getByRole("status")).toHaveTextContent("A scheduled scan is overdue");
});

it("does not invent a price timestamp for legacy snapshots", () => {
  render(<ScannerDataStatus {...base} scanAt={info.last_successful_scan_at} />);
  expect(screen.getByText("Saved snapshot")).toBeInTheDocument();
  expect(screen.getByText(/Price time unavailable/)).toBeInTheDocument();
  expect(screen.queryByText("On schedule")).not.toBeInTheDocument();
});

it("keeps failed refreshes actionable without relabeling the snapshot as live", () => {
  const reload = vi.fn();
  render(<ScannerDataStatus {...base} info={info} reload={reload}
    error={new ApiError(503, "Service unavailable", "request-123", null, null)} />);
  expect(screen.getByRole("alert")).toHaveTextContent("request-123");
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(reload).toHaveBeenCalledOnce();
});
