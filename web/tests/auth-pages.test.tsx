import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ForgotPasswordForm, ResetPasswordForm, SignupForm, VerifyEmail, signupFailureKind } from "@/features/AuthForms";

import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
vi.mock("next/navigation", () => ({ useSearchParams: () => search, useRouter: () => ({ replace: vi.fn(), push: vi.fn() }), usePathname: () => "/" }));

const fetchMock = vi.fn();
const assign = vi.fn();
const bodies = () => fetchMock.mock.calls.map((c) => [c[0] as string, JSON.parse((c[1] as RequestInit).body as string)]);
beforeEach(() => {
  search = new URLSearchParams();
  fetchMock.mockReset();
  assign.mockReset();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("location", { ...window.location, assign });
});
afterEach(() => vi.unstubAllGlobals());

describe("sign-up and account recovery pages", () => {
  it("every auth page has a way back to the home page", () => {
    for (const page of [<SignupForm key="s" />, <ForgotPasswordForm key="f" />, <ResetPasswordForm key="r" />, <VerifyEmail key="v" />]) {
      const { unmount } = render(page);
      expect(screen.getByRole("link", { name: "HSFinest.com home" })).toHaveAttribute("href", "/");
      expect(screen.getByRole("link", { name: /Back to home/ })).toHaveAttribute("href", "/");
      unmount();
    }
  });

  it("sign-up: email, password and the agreement, shows the API's rule, then opens Today", async () => {
    const answers = [jsonResponse({ detail: "Password must be at least 10 characters." }, 400),
      jsonResponse({ ok: true, email: "a@b.co", verification_sent: true }, 201)];
    fetchMock.mockImplementation(async (url: string) => (url.endsWith("/v1/events") ? new Response(null, { status: 202 }) : answers.shift()!));
    localStorage.setItem("hsf-attribution", JSON.stringify({ at: Date.now(), attribution: { utm_source: "reddit" } }));
    const u = userEvent.setup();
    render(<SignupForm />);
    expect(screen.queryByLabelText("Display name")).toBeNull();
    expect(screen.queryByLabelText("Repeat password")).toBeNull();
    await u.type(screen.getByLabelText("Email"), "a@b.co");
    await u.type(screen.getByLabelText("Password"), "short");
    const go = screen.getByRole("button", { name: "Create account" });
    expect(go).toBeDisabled();
    await u.click(screen.getByRole("checkbox"));
    await u.click(go);
    expect(await screen.findByRole("alert")).toHaveTextContent("at least 10 characters");
    await u.click(go);
    await waitFor(() => expect(assign).toHaveBeenCalledWith("/today"));
    const signups = bodies().filter(([url]) => url === "/api/auth/signup");
    expect(signups[1]).toEqual(["/api/auth/signup", { email: "a@b.co", password: "short", accept_terms: true,
      attribution: { utm_source: "reddit" } }]);
    // Each attempt records "sign-up started"; the refused one also records why (no personal data).
    const events = bodies().filter(([url]) => url === "/api/public/v1/events").map(([, b]) => [b.event, b.surface]);
    expect(events).toEqual([["signup_started", "signup_form"], ["signup_failed", "password"], ["signup_started", "signup_form"]]);
    localStorage.clear();
  });

  it("keeps the plan picked on Pricing for checkout after the email is confirmed", async () => {
    search = new URLSearchParams("plan=premium&interval=year");
    render(<SignupForm />);
    expect(screen.getByText(/You picked Premium/)).toBeInTheDocument();
    await waitFor(() => expect(JSON.parse(localStorage.getItem("hsf-plan-intent") || "{}")).toMatchObject({ plan: "premium", interval: "year" }));
    localStorage.clear();
  });

  it("keeps password guidance available and reveals the password without submitting", async () => {
    const u = userEvent.setup();
    render(<SignupForm />);
    const password = screen.getByLabelText("Password");
    expect(password).toHaveAccessibleDescription(/At least 10 characters/);
    expect(screen.getByRole("link", { name: "HSFinest.com home" })).toHaveAttribute("href", "/");
    await u.type(password, "example-passphrase");
    await u.click(screen.getByRole("button", { name: "Show password" }));
    expect(password).toHaveAttribute("type", "text");
    expect(password).toHaveValue("example-passphrase");
    await u.click(screen.getByRole("button", { name: "Hide password" }));
    expect(password).toHaveAttribute("type", "password");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("names why a sign-up was refused, without personal data", () => {
    expect(signupFailureKind(409, "That email or username is already taken.")).toBe("taken");
    expect(signupFailureKind(400, "Password must be at least 10 characters.")).toBe("password");
    expect(signupFailureKind(400, "Please enter a valid email address.")).toBe("email");
    expect(signupFailureKind(0, "Couldn't reach HSF.")).toBe("network");
    expect(signupFailureKind(503, "Try again")).toBe("server");
  });

  it("forgot password: same answer either way", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ ok: true, message: "If that email is registered, a reset link has been sent." }));
    const u = userEvent.setup();
    render(<ForgotPasswordForm />);
    await u.type(screen.getByLabelText("Email"), "x@y.co");
    await u.click(screen.getByRole("button", { name: "Send reset link" }));
    expect(await screen.findByText(/a reset link has been sent/)).toBeInTheDocument();
  });

  it("reset password: uses the emailed token; a missing token offers a new link; a bad one says so", async () => {
    const { unmount } = render(<ResetPasswordForm />);
    expect(screen.getByRole("link", { name: "Send a new link" })).toHaveAttribute("href", "/forgot-password");
    unmount();
    search = new URLSearchParams("token=expired");
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "This reset link is invalid or has expired." }, 400))
      .mockResolvedValueOnce(jsonResponse({ ok: true, message: "Password updated." }));
    const u = userEvent.setup();
    render(<ResetPasswordForm />);
    await u.type(screen.getByLabelText("New password"), "a-long-password");
    await u.type(screen.getByLabelText("Repeat new password"), "a-long-password");
    await u.click(screen.getByRole("button", { name: "Set password" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("invalid or has expired");
    await u.click(screen.getByRole("button", { name: "Set password" }));
    expect(await screen.findByText(/Password updated/)).toBeInTheDocument();
    expect(bodies()[1]).toEqual(["/api/auth/password-reset-confirm", { token: "expired", new_password: "a-long-password" }]);
  });

  it("verify email: confirms the token once", async () => {
    search = new URLSearchParams("token=emailed");
    fetchMock.mockResolvedValue(jsonResponse({ ok: true, message: "Email verified." }));
    render(<VerifyEmail />);
    expect(await screen.findByText(/Email verified/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
