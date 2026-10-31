import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { EmailLinkSignIn } from "@/app/login/link/EmailLinkSignIn";
import { LoginForm } from "@/app/login/LoginForm";
import { SignedOut } from "@/app/signed-out/SignedOut";
import { PasswordField } from "@/components/PasswordField";
import { forgetEmail, readLastEmail, rememberEmail } from "@/lib/lastEmail";

import { jsonResponse } from "./helpers";

let search = new URLSearchParams();
vi.mock("next/navigation", () => ({ useSearchParams: () => search, useRouter: () => ({ replace: vi.fn(), push: vi.fn() }), usePathname: () => "/" }));

const fetchMock = vi.fn();
const assign = vi.fn();
const posts = () => fetchMock.mock.calls.filter((c) => (c[1] as RequestInit | undefined)?.method === "POST")
  .map((c) => [c[0] as string, JSON.parse(((c[1] as RequestInit).body as string) || "null")]);
beforeEach(() => {
  search = new URLSearchParams();
  fetchMock.mockReset();
  assign.mockReset();
  forgetEmail();
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("location", { ...window.location, assign });
});
afterEach(() => vi.unstubAllGlobals());

describe("sign-in page", () => {
  it("welcomes back, links forgot password beside the field, and shows legal links", () => {
    render(<LoginForm />);
    expect(screen.getByRole("heading", { name: "Welcome back" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Forgot password?" })).toHaveAttribute("href", "/forgot-password");
    expect(screen.getByRole("link", { name: "Create a free account" })).toHaveAttribute("href", "/signup");
    expect(screen.getByRole("link", { name: "Terms" })).toHaveAttribute("href", "/terms");
    expect(screen.getByRole("link", { name: "Privacy" })).toHaveAttribute("href", "/privacy");
    expect(screen.getByText(/stay signed in for 30 days/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeDisabled();
  });

  it("remembers the email after signing in, and Not you? clears it", async () => {
    fetchMock.mockImplementation(async (url: string) => (url === "/api/auth/login" ? jsonResponse({ ok: true }) : jsonResponse({ email: "Ann@Example.com", username: "ann" })));
    const u = userEvent.setup();
    const first = render(<LoginForm />);
    await u.type(screen.getByLabelText("Email"), "ann@example.com");
    await u.type(screen.getByLabelText("Password"), "pw");
    await u.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() => expect(assign).toHaveBeenCalledWith("/today"));
    expect(readLastEmail()).toBe("ann@example.com");
    first.unmount();

    render(<LoginForm />);
    expect(screen.getByLabelText("Email")).toHaveValue("ann@example.com");
    expect(screen.getByLabelText("Password")).toHaveFocus();
    await u.click(screen.getByRole("button", { name: "Not you?" }));
    expect(screen.getByLabelText("Email")).toHaveValue("");
    expect(readLastEmail()).toBe("");
    expect(screen.queryByRole("button", { name: "Not you?" })).toBeNull();
  });

  it("emails a sign-in link instead of using the password", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ ok: true, message: "If that email has an HSF account, a sign-in link is on its way." }));
    const u = userEvent.setup();
    render(<LoginForm />);
    await u.type(screen.getByLabelText("Email"), "ann@example.com");
    await u.click(screen.getByRole("button", { name: "Email me a sign-in link instead" }));
    expect(screen.queryByLabelText("Password")).toBeNull();
    await u.click(screen.getByRole("button", { name: "Email me a link" }));
    expect(await screen.findByRole("status")).toHaveTextContent("sign-in link is on its way");
    expect(posts()).toEqual([["/api/auth/email-link", { email: "ann@example.com" }]]);
    await u.click(screen.getByRole("button", { name: "Back to sign in" }));
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  it("opens straight in link mode from ?mode=link", () => {
    search = new URLSearchParams("mode=link");
    render(<LoginForm />);
    expect(screen.getByRole("button", { name: "Email me a link" })).toBeInTheDocument();
  });
});

describe("Caps Lock warning", () => {
  it("shows while Caps Lock is on and hides on blur", () => {
    render(<PasswordField label="Password" />);
    const input = screen.getByLabelText("Password");
    fireEvent.keyUp(input, { key: "A", modifierCapsLock: true } as KeyboardEventInit);
    expect(screen.getByText("Caps Lock is on")).toBeInTheDocument();
    fireEvent.blur(input);
    expect(screen.queryByText("Caps Lock is on")).toBeNull();
  });
});

describe("emailed sign-in link page", () => {
  it("waits for a click, then signs in and opens Today", async () => {
    search = new URLSearchParams("token=tok-123456789");
    fetchMock.mockImplementation(async (url: string) => (url === "/api/auth/email-link-confirm" ? jsonResponse({ ok: true }) : jsonResponse({ email: "ann@example.com" })));
    const u = userEvent.setup();
    render(<EmailLinkSignIn />);
    expect(fetchMock).not.toHaveBeenCalled();
    await u.click(screen.getByRole("button", { name: "Continue to HSF" }));
    await waitFor(() => expect(assign).toHaveBeenCalledWith("/today"));
    expect(posts()[0]).toEqual(["/api/auth/email-link-confirm", { token: "tok-123456789" }]);
    expect(readLastEmail()).toBe("ann@example.com");
  });

  it("explains a used link and offers a new one", async () => {
    search = new URLSearchParams("token=old-123456789");
    fetchMock.mockResolvedValue(jsonResponse({ detail: "This sign-in link is invalid, expired or already used. Ask for a new one." }, 400));
    const u = userEvent.setup();
    render(<EmailLinkSignIn />);
    await u.click(screen.getByRole("button", { name: "Continue to HSF" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("already used");
    expect(screen.getByRole("link", { name: "Send a new link" })).toHaveAttribute("href", "/login?mode=link");
    expect(assign).not.toHaveBeenCalled();
  });
});

describe("signed-out page", () => {
  it("confirms sign-out, says what keeps arriving, and can forget the email", async () => {
    search = new URLSearchParams("push=off");
    rememberEmail("ann@example.com");
    const u = userEvent.setup();
    render(<SignedOut />);
    expect(screen.getByRole("heading", { name: "You're signed out" })).toBeInTheDocument();
    expect(screen.getByText(/daily emails you chose keep arriving/)).toHaveTextContent("Push alerts are now off on this browser");
    expect(screen.getByRole("link", { name: "Sign back in" })).toHaveAttribute("href", "/login");
    await u.click(screen.getByRole("button", { name: "Forget ann@example.com on this device" }));
    expect(readLastEmail()).toBe("");
    expect(screen.getByText(/won't fill in your email/)).toBeInTheDocument();
  });

  it("leaves out the push line when push wasn't on", () => {
    render(<SignedOut />);
    expect(screen.getByText(/keep arriving/)).not.toHaveTextContent("Push");
    expect(screen.queryByRole("button", { name: /Forget/ })).toBeNull();
  });
});
