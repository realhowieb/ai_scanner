import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/api/client";
import { ErrorLine, ErrorState, RefreshStatus } from "@/components/ui";
import { PaperOrderTracker, usePaperTracking } from "@/features/PaperOrderTracker";
import { jsonResponse } from "./helpers";

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

it("shows field-specific validation and the support code", () => {
  render(<ErrorLine error={new ApiError(422, "Check the fields below and try again.", "request-123", null,
    { fields: { shares: "Enter a whole number." } })} />);
  expect(screen.getByRole("alert")).toHaveTextContent("shares: Enter a whole number.");
  expect(screen.getByRole("alert")).toHaveTextContent("request-123");
});

it("disables retries until Retry-After expires", async () => {
  vi.useFakeTimers();
  const retry = vi.fn();
  render(<ErrorState error={new ApiError(429, "Too many requests.", null, 2, null)} onRetry={retry} />);
  expect(screen.getByRole("button", { name: "Try again in 2s" })).toBeDisabled();
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(retry).toHaveBeenCalledOnce();
});

it("explains failed refreshes while keeping the last response time distinct from market data", () => {
  render(<RefreshStatus updatedAt={Date.now()} refreshing={false}
    error={new ApiError(503, "Service unavailable", "refresh-123", null, null)} onRetry={() => {}} />);
  expect(screen.getByText(/Market prices may be older/)).toBeInTheDocument();
  expect(screen.getByText(/Showing the last available data/)).toBeInTheDocument();
  expect(screen.getByRole("alert")).toHaveTextContent("refresh-123");
});

describe("paper confirmation tracking", () => {
  it("recovers after remount and checks fills without posting another order", async () => {
    let filled = false;
    const fetch = vi.fn<(request: Request) => Promise<Response>>(async () => jsonResponse({ state: filled ? "synced" : "submitted",
      status: filled ? "filled" : "accepted", ticker: "AAPL", qty: 2, order_id: "order-1",
      filled_qty: filled ? 2 : 0, journal_synced: filled, broker_available: true }));
    vi.stubGlobal("fetch", fetch);
    function Recover({ owner = "a@example.com" }: { owner?: string }) {
      const tracking = usePaperTracking(owner);
      return <><button onClick={() => tracking.save("confirmation-123")}>Track</button>
        {tracking.key && <PaperOrderTracker confirmationId={tracking.key} />}</>;
    }
    const first = render(<Recover />);
    fireEvent.click(screen.getByText("Track"));
    await screen.findByText(/Awaiting fill/);
    first.unmount();
    const next = render(<Recover />);
    await screen.findByText(/Awaiting fill/);
    filled = true;
    fireEvent.click(screen.getByText("Check order status"));
    await screen.findByText("Filled · Journal synced");
    expect(fetch.mock.calls.every((args) => (args[0] as Request).method === "GET")).toBe(true);
    next.rerender(<Recover owner="b@example.com" />);
    expect(screen.queryByLabelText("Latest paper order")).not.toBeInTheDocument();
  });

  it("labels a cached broker response as unconfirmed", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ state: "submitted", status: "accepted",
      ticker: "AAPL", qty: 2, order_id: "order-1", filled_qty: 0, journal_synced: false, broker_available: false })));
    render(<PaperOrderTracker confirmationId="confirmation-123" />);
    await waitFor(() => expect(screen.getByText(/Showing the saved status/)).toBeInTheDocument());
  });
});


it("does not describe a canceled partial fill as a fully filled order", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ state: "synced", status: "canceled",
    ticker: "AAPL", qty: 2, order_id: "order-1", filled_qty: 1, journal_synced: true })));
  render(<PaperOrderTracker confirmationId="confirmation-123" />);
  await screen.findByText("Order canceled · 1 of 2 shares filled · Journal synced");
});
