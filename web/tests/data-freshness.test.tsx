import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Schemas } from "@/api/client";
import { DataFreshness } from "@/features/DataFreshness";
import { useApi } from "@/hooks/useApi";

const info: Schemas["DataFreshness"] = {
  state: "partial", scan_state: "fresh", market_data_state: "unavailable",
  last_successful_scan_at: "2026-10-09T19:35:00+00:00", scan_completed_at: null,
  market_data_at: null, expected_scan_at: "2026-10-09T19:35:00+00:00",
  calendar_covered: true, checked_at: "2026-10-10T12:00:00+00:00", timestamp_basis: "saved_scan",
};

describe("data freshness", () => {
  it("does not mistake a successful saved scan for fresh underlying data or completion", () => {
    render(<DataFreshness info={info} />);
    expect(screen.getByLabelText("Data freshness")).toHaveTextContent("Partially available");
    expect(screen.getByText(/Last successful scan/)).toHaveTextContent("3:35 PM ET");
    expect(screen.getByText("Scan completed: Unavailable")).toBeInTheDocument();
    expect(screen.getByText("Market data as of: Unavailable")).toBeInTheDocument();
  });
  it("uses the server's session-aware state rather than clock age", () => {
    const { rerender } = render(<DataFreshness info={{ ...info, state: "fresh" }} />);
    expect(screen.getByText("Data freshness: Fresh")).toBeInTheDocument();
    rerender(<DataFreshness info={{ ...info, state: "stale" }} />);
    expect(screen.getByText("Data freshness: Stale")).toBeInTheDocument();
  });
  it("supports older API responses without freshness metadata", () => {
    render(<DataFreshness />);
    expect(screen.queryByLabelText("Data freshness")).not.toBeInTheDocument();
  });
  it("preserves usable results when a manual refresh fails without polling", async () => {
    const load = vi.fn().mockResolvedValueOnce({ freshness: info, setups: ["AAA"] })
      .mockRejectedValueOnce(new Error("temporary failure"));
    const { result } = renderHook(() => useApi<{ freshness: Schemas["DataFreshness"]; setups: string[] }>("scan", load));
    await waitFor(() => expect(result.current.data?.setups).toEqual(["AAA"]));
    act(() => result.current.reload());
    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.data?.setups).toEqual(["AAA"]);
    expect(load).toHaveBeenCalledTimes(2);
  });
});
