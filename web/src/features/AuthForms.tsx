"use client";

// Sign-up, password reset and email verification, so none of them needs the classic
// app. They post to this site's /api/auth/* routes; tokens from sign-up go straight
// into HttpOnly cookies on the server. The reset and verify pages live at the same
// paths the emailed links use (/reset_password?token=, /verify_email?token=).
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useId, useRef, useState, useSyncExternalStore } from "react";
import type { FormEvent, ReactNode } from "react";

import { PasswordField } from "@/components/PasswordField";
import { messageFrom, newRequestId } from "@/api/client";
import { WAKING_UP, useSlow } from "@/components/ui";
import { captureAttribution, track } from "@/lib/funnel";

type Outcome = { ok: boolean; status: number; body: Record<string, unknown> | null; requestId: string | null };

export async function postAuth(path: string, body: unknown): Promise<Outcome> {
  try {
    const res = await fetch(`/api/auth/${path}`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json", "x-request-id": newRequestId() },
      body: JSON.stringify(body),
    });
    const parsed = (await res.json().catch(() => null)) as Record<string, unknown> | null;
    return { ok: res.ok, status: res.status, body: parsed, requestId: res.headers.get("x-request-id") };
  } catch {
    return { ok: false, status: 0, body: { detail: "Couldn't reach HSF. Check your connection and try again." }, requestId: null };
  }
}

const errorOf = (o: Outcome) => (o.status === 0 ? String(o.body?.detail) : messageFrom(o.body, o.status)) + (o.status >= 500 && o.requestId ? ` Support code: ${o.requestId}` : "");

export function AuthShell({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="card narrow login">
      <Link href="/" className="brand brand-lg" aria-label="HSFinest.AI home">HSFinest<span>.AI</span></Link>
      <h1 className="h1">{title}</h1>
      {children}
    </div>
  );
}

const noSubscribe = () => () => {};

export function SignupForm() {
  const fieldId = useId();
  const [showPasswords, setShowPasswords] = useState(false);
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [agree, setAgree] = useState(false);
  const invited = useSyncExternalStore(noSubscribe, () => !!captureAttribution().ref, () => false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const slow = useSlow(busy);
  const mismatch = !!pw2 && pw !== pw2;
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (pw !== pw2) return;
    setBusy(true);
    setError(null);
    const attribution = captureAttribution();
    track("signup_started", "signup_form");
    const o = await postAuth("signup", { email, username, password: pw, accept_terms: agree, attribution });
    if (o.ok) {
      // A full page load: the new session's cookies are already set.
      // eslint-disable-next-line @next/next/no-location-assign-relative-destination
      window.location.assign("/today");
      return;
    }
    setError(errorOf(o));
    setBusy(false);
  };
  return (
    <form onSubmit={submit} noValidate>
      <AuthShell title="Create your account">
        <p className="cap">Starter plan, free. Upgrade any time from Account.</p>
        {invited && <p className="notice" role="status">You were invited by a friend. Confirm your email after signing up and you both get free Pro time.</p>}
        <label className="field"><span>Email</span>
          <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label className="field"><span>Display name</span>
          <input autoComplete="nickname" aria-describedby={`${fieldId}-name-help`} required maxLength={40} value={username} onChange={(e) => setUsername(e.target.value)} />
        </label>
        <p className="cap" id={`${fieldId}-name-help`}>Shown in your account. You can also use it to sign in.</p>
        <div className="password-guidance">
          <p className="cap" id={`${fieldId}-password-help`}>Use at least 10 characters. Avoid common passwords, repeated patterns, and your name or email.</p>
          <button type="button" className="btn btn-sm" aria-pressed={showPasswords} aria-controls={`${fieldId}-password ${fieldId}-repeat`}
            onClick={() => setShowPasswords((v) => !v)}>{showPasswords ? "Hide passwords" : "Show passwords"}</button>
        </div>
        <label className="field"><span>Password</span>
          <input id={`${fieldId}-password`} type={showPasswords ? "text" : "password"} autoComplete="new-password" required value={pw}
            aria-describedby={`${fieldId}-password-help`} onChange={(e) => setPw(e.target.value)} />
        </label>
        <label className="field"><span>Repeat password</span>
          <input id={`${fieldId}-repeat`} type={showPasswords ? "text" : "password"} autoComplete="new-password" required value={pw2} onChange={(e) => setPw2(e.target.value)} aria-invalid={mismatch || undefined} aria-describedby={mismatch ? `${fieldId}-mismatch` : undefined} />
        </label>
        {mismatch && <p className="form-error" id={`${fieldId}-mismatch`} role="status">The passwords don&apos;t match.</p>}
        <label className="check"><input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
          <span>I agree to use this tool for educational/informational purposes only.</span></label>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button type="submit" className="btn btn-primary btn-block" disabled={busy || !email || !username || !pw || pw !== pw2 || !agree}>
          {busy ? "Creating account…" : "Create account"}
        </button>
        {slow && <p className="cap" role="status">{WAKING_UP}</p>}
        <p className="cap">Next, you&apos;ll open Today. We&apos;ll email a verification link so you can upgrade and receive email alerts.</p>
        <p className="cap login-links">Already have an account? <Link href="/login">Sign in</Link></p>
      </AuthShell>
    </form>
  );
}

export function ForgotPasswordForm() {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const o = await postAuth("password-reset", { email });
    if (o.ok) setSent(String(o.body?.message ?? "If that email is registered, a reset link has been sent."));
    else setError(errorOf(o));
    setBusy(false);
  };
  return (
    <form onSubmit={submit} noValidate>
      <AuthShell title="Reset your password">
        {sent ? <p className="notice" role="status">{sent}</p> : (
          <>
            <p className="cap">We&apos;ll email you a link to set a new password.</p>
            <label className="field"><span>Email</span>
              <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
            </label>
            {error && <p className="form-error" role="alert">{error}</p>}
            <button type="submit" className="btn btn-primary btn-block" disabled={busy || !email}>{busy ? "Sending…" : "Send reset link"}</button>
          </>
        )}
        <p className="cap login-links"><Link href="/login">Back to sign in</Link></p>
      </AuthShell>
    </form>
  );
}

export function ResetPasswordForm() {
  const token = useSearchParams().get("token") || "";
  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (pw !== pw2) return;
    setBusy(true);
    setError(null);
    const o = await postAuth("password-reset-confirm", { token, new_password: pw });
    if (o.ok) setDone(String(o.body?.message ?? "Password updated."));
    else setError(errorOf(o));
    setBusy(false);
  };
  if (!token) {
    return (
      <AuthShell title="Reset link missing">
        <p className="body-sm">Open the link from the reset email, or ask for a new one.</p>
        <Link href="/forgot-password" className="btn btn-primary btn-block">Send a new link</Link>
      </AuthShell>
    );
  }
  return (
    <form onSubmit={submit} noValidate>
      <AuthShell title="Set a new password">
        {done ? (
          <>
            <p className="notice" role="status">{done} You&apos;ve been signed out everywhere.</p>
            <Link href="/login" className="btn btn-primary btn-block">Sign in</Link>
          </>
        ) : (
          <>
            <p className="cap">Use at least 10 characters. Avoid common passwords, repeated patterns, and your name or email.</p>
            <PasswordField label="New password" autoComplete="new-password" required value={pw} onChange={(e) => setPw(e.target.value)} />
            <PasswordField label="Repeat new password" autoComplete="new-password" required value={pw2} onChange={(e) => setPw2(e.target.value)} aria-invalid={!!pw2 && pw !== pw2 || undefined} />
            {!!pw2 && pw !== pw2 && <p className="form-error">The passwords don&apos;t match.</p>}
            {error && <p className="form-error" role="alert">{error} <Link href="/forgot-password">Get a new link</Link></p>}
            <button type="submit" className="btn btn-primary btn-block" disabled={busy || !pw || pw !== pw2}>{busy ? "Saving…" : "Set password"}</button>
          </>
        )}
      </AuthShell>
    </form>
  );
}

export function VerifyEmail() {
  const token = useSearchParams().get("token") || "";
  const [state, setState] = useState<{ ok: boolean; text: string } | null>(null);
  const started = useRef(false);
  useEffect(() => {
    if (!token || started.current) return;
    started.current = true;
    void postAuth("verify-email", { token }).then((o) =>
      setState(o.ok ? { ok: true, text: String(o.body?.message ?? "Email verified.") } : { ok: false, text: errorOf(o) }));
  }, [token]);
  return (
    <AuthShell title="Verify your email">
      {!token ? <p className="body-sm">Open the link from the verification email. Signed in, you can send a new one from Account.</p>
        : !state ? <p className="cap" role="status">Checking your link…</p>
        : state.ok ? <p className="notice" role="status">{state.text} You can upgrade and get email alerts now.</p>
        : <p className="form-error" role="alert">{state.text} Signed in, you can send a new link from Account.</p>}
      <div className="row-actions"><Link href="/today" className="btn btn-primary">Open HSF</Link><Link href="/account">Account</Link></div>
    </AuthShell>
  );
}
