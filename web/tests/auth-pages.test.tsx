import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ForgotPasswordForm, ResetPasswordForm, SignupForm, VerifyEmail } from "@/features/AuthForms";

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
  it("sign-up: needs matching passwords and the agreement, shows the API's rule, then opens Today", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "Password must be at least 10 characters." }, 400))
      .mockResolvedValueOnce(jsonResponse({ ok: true, email: "a@b.co", verification_sent: true }, 201));
    const u = userEvent.setup();
    render(<SignupForm />);
    await u.type(screen.getByLabelText("Email"), "a@b.co");
    await u.type(screen.getByLabelText("Display name"), "ann");
    await u.type(screen.getByLabelText("Password"), "short");
    await u.type(screen.getByLabelText("Repeat password"), "shorx");
    expect(screen.getByText("The passwords don't match.")).toBeInTheDocument();
    const go = screen.getByRole("button", { name: "Create account" });
    expect(go).toBeDisabled();
    await u.clear(screen.getByLabelText("Repeat password"));
    await u.type(screen.getByLabelText("Repeat password"), "short");
    expect(go).toBeDisabled();
    await u.click(screen.getByRole("checkbox"));
    await u.click(go);
    expect(await screen.findByRole("alert")).toHaveTextContent("at least 10 characters");
    await u.click(go);
    await waitFor(() => expect(assign).toHaveBeenCalledWith("/today"));
    expect(bodies()[1]).toEqual(["/api/auth/signup", { email: "a@b.co", username: "ann", password: "short", accept_terms: true }]);
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
