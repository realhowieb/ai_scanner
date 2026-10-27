import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import BillingReturn from "@/app/billing/return/page";
import { beginBilling, pendingBilling } from "@/lib/billingReturn";
import { SessionProvider, useSession } from "@/session/SessionProvider";

import { me } from "./fixtures";
import { jsonResponse } from "./helpers";

const fetchMock = vi.fn();
const original = me("pro");
const upgraded = { ...me("premium"), email: original.email };

function AccountPlan() {
  const { me, syncBilling, billingError } = useSession();
  return <><p>Current plan: {me?.plan}</p>
    <button onClick={() => void syncBilling()}>Refresh plan</button>
    {billingError && <p role="alert">{billingError.message}</p>}</>;
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

it("refreshes the original Mac app after Stripe even with an initial session", async () => {
  let fresh = original;
  const paths: string[] = [];
  fetchMock.mockImplementation((req: Request) => {
    const path = new URL(req.url).pathname;
    paths.push(path);
    if (path.endsWith("/billing/sync")) return Promise.resolve(jsonResponse({ message: "Checked Stripe" }));
    if (path.endsWith("/v1/me")) return Promise.resolve(jsonResponse(fresh));
    return Promise.resolve(jsonResponse({}, 404));
  });
  render(<SessionProvider initialMe={original}><AccountPlan /></SessionProvider>);
  expect(screen.getByText("Current plan: pro")).toBeInTheDocument();
  beginBilling("pro", "premium");
  await waitFor(() => expect(paths).toContain("/api/hsf/v1/billing/sync"));
  await waitFor(() => expect(paths).toContain("/api/hsf/v1/me"));
  fresh = upgraded;
  fireEvent(window, new Event("focus"));
  expect(await screen.findByText("Current plan: premium")).toBeInTheDocument();
  await waitFor(() => expect(pendingBilling()).toBeNull());
});

it("shows a configuration problem and preserves the server's current plan", async () => {
  fetchMock.mockImplementation((req: Request) => Promise.resolve(new URL(req.url).pathname.endsWith("/billing/sync")
    ? jsonResponse({ detail: "Stripe's subscription price is not configured in HSF." }, 409)
    : jsonResponse(original)));
  render(<SessionProvider initialMe={original}><AccountPlan /></SessionProvider>);
  await userEvent.setup().click(screen.getByRole("button", { name: "Refresh plan" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("not configured");
  expect(screen.getByText("Current plan: pro")).toBeInTheDocument();
  expect(screen.queryByText("Current plan: premium")).not.toBeInTheDocument();
});

it("does not poll Stripe when no billing flow is pending", () => {
  render(<SessionProvider initialMe={original}><AccountPlan /></SessionProvider>);
  fireEvent(window, new Event("focus"));
  expect(fetchMock).not.toHaveBeenCalled();
});

it("bounds polling when Stripe has not confirmed the requested upgrade", async () => {
  vi.useFakeTimers();
  fetchMock.mockImplementation((req: Request) => Promise.resolve(new URL(req.url).pathname.endsWith("/billing/sync")
    ? jsonResponse({ message: "Checked Stripe" }) : jsonResponse(original)));
  render(<SessionProvider initialMe={original}><AccountPlan /></SessionProvider>);
  await act(async () => {
    beginBilling("pro", "premium");
    await vi.advanceTimersByTimeAsync(15_000);
  });
  const syncCount = () => fetchMock.mock.calls.filter(([req]) => new URL(req.url).pathname.endsWith("/billing/sync")).length;
  expect(syncCount()).toBe(3);
  await act(async () => { await vi.advanceTimersByTimeAsync(60_000); });
  expect(syncCount()).toBe(3);
  expect(screen.getByText("Current plan: pro")).toBeInTheDocument();
});

it("gives a Mac web app user a return instruction and an account link", () => {
  fetchMock.mockImplementation(() => Promise.resolve(jsonResponse({ detail: "no session" }, 401)));
  render(<BillingReturn />);
  expect(screen.getByText(/switch back to its window/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Open account and refresh plan" })).toHaveAttribute("href", "/account?billing=return");
});

it("names the account this browser is signed in to, so a different account is obvious", async () => {
  fetchMock.mockImplementation(() => Promise.resolve(jsonResponse({ ...original, email: "other@example.com" })));
  render(<BillingReturn />);
  expect(await screen.findByRole("status")).toHaveTextContent("Signed in here as other@example.com");
  expect(screen.getByRole("button", { name: "sign out here" })).toBeInTheDocument();
});

it("shows no account note when this browser is signed out", async () => {
  fetchMock.mockImplementation(() => Promise.resolve(jsonResponse({ detail: "no session" }, 401)));
  render(<BillingReturn />);
  await waitFor(() => expect(fetchMock).toHaveBeenCalled());
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});
