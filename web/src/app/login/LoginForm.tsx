"use client";

import Link from "@/components/Link";
import { AuthShell, postAuth } from "@/features/AuthForms";
import { PasswordField } from "@/components/PasswordField";
import { TURNSTILE_SITE_KEY, Turnstile } from "@/components/Turnstile";
import { LegalLinks } from "@/components/LegalLinks";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";

import { messageFrom, newRequestId } from "@/api/client";
import { WAKING_UP, useSlow } from "@/components/ui";
import { forgetEmail, rememberEmail, useLastEmail } from "@/lib/lastEmail";
import { safeNext, withNext } from "@/lib/nextPath";
import { writeCachedMe } from "@/session/meCache";
import { parseRetryAfter } from "@/lib/retryAfter";

/** After the API set the session cookies: confirm the browser kept them, remember the email,
 * then open `next`. Returns an error message when the cookie didn't stick. Shared with the
 * emailed sign-in link page. */
export async function finishSignIn(next: string): Promise<{ error: string; supportId: string | null } | null> {
  const check = await fetch("/api/hsf/v1/me", { credentials: "same-origin", headers: { "x-request-id": newRequestId() } });
  if (check.ok) {
    const me = (await check.json().catch(() => null)) as { email?: string } | null;
    writeCachedMe(me as Parameters<typeof writeCachedMe>[0]);
    rememberEmail(me?.email);
    window.location.assign(next);
    return null;
  }
  return {
    error: check.status === 401
      ? "You're signed in, but this browser didn't keep the sign-in cookie. Allow cookies for this site; when running locally, open it at http://localhost:3000 (not an IP address)."
      : messageFrom(await check.json().catch(() => null), check.status),
    supportId: check.headers.get("x-request-id"),
  };
}

export function LoginForm() {
  const params = useSearchParams();
  const next = safeNext(params.get("next"));
  const expired = params.get("expired") === "1";
  const [mode, setMode] = useState<"password" | "link">(params.get("mode") === "link" ? "link" : "password");
  // The last email used on this device fills the field until the visitor types their own.
  const last = useLastEmail();
  const [typed, setEmail] = useState<string | null>(null);
  const email = typed ?? last;
  const remembered = !!last && email === last;
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [supportId, setSupportId] = useState<string | null>(null);
  const [waitS, setWaitS] = useState<number | null>(null);
  const [linkSent, setLinkSent] = useState<string | null>(null);
  const [captcha, setCaptcha] = useState<string | null>(null);
  const [captchaReset, setCaptchaReset] = useState(0);
  const emailRef = useRef<HTMLInputElement>(null);
  const passwordRef = useRef<HTMLInputElement>(null);
  const slow = useSlow(busy);

  // With the email filled in, the password is the next thing to type.
  useEffect(() => {
    if (last) passwordRef.current?.focus();
  }, [last]);

  useEffect(() => {
    if (!waitS) return;
    const t = setTimeout(() => setWaitS((s) => (s && s > 1 ? s - 1 : null)), 1000);
    return () => clearTimeout(t);
  }, [waitS]);

  const notYou = () => {
    forgetEmail();
    setEmail("");
    emailRef.current?.focus();
  };

  const switchMode = (m: "password" | "link") => {
    setMode(m);
    setError(null);
    setSupportId(null);
    setLinkSent(null);
  };

  const submitPassword = async () => {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json", "x-request-id": newRequestId() },
      body: JSON.stringify({ email, password }),
    });
    if (res.ok) {
      // Confirm the browser kept the session cookie before leaving this page;
      // otherwise every page would bounce straight back here with no explanation.
      const failed = await finishSignIn(next);
      if (!failed) return true;
      setError(failed.error);
      setSupportId(failed.supportId);
      return false;
    }
    const body: unknown = await res.json().catch(() => null);
    setError(messageFrom(body, res.status));
    setSupportId(res.status >= 500 ? res.headers.get("x-request-id") : null);
    if (res.status === 429 || res.status === 503) setWaitS(parseRetryAfter(res.headers.get("retry-after")) ?? 60);
    return false;
  };

  const submitLink = async () => {
    const o = await postAuth("email-link", { email, ...(captcha ? { turnstile_token: captcha } : {}) });
    if (o.ok) {
      setLinkSent(String(o.body?.message ?? "If that email has an HSF account, a sign-in link is on its way."));
      return;
    }
    setError(o.status === 0 ? String(o.body?.detail) : messageFrom(o.body, o.status));
    if (o.status === 429) setWaitS(60);
    setCaptchaReset((n) => n + 1);
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setSupportId(null);
    try {
      if (mode === "password") {
        if (await submitPassword()) return;
      } else {
        await submitLink();
      }
    } catch {
      setError("Couldn't reach HSF. Check your connection and try again.");
    }
    setBusy(false);
  };

  const needsCaptcha = mode === "link" && !!TURNSTILE_SITE_KEY && !captcha;
  return (
    <form onSubmit={submit} noValidate>
      <AuthShell title="Welcome back" subtitle="Sign in to HSFinest.AI">
      <p className="auth-pitch">Today&apos;s setups, your watchlists and your alerts are waiting.</p>
      {expired && <p className="notice" role="status">Your session ended. Sign in again to continue.</p>}
      {linkSent ? (
        <>
          <p className="notice" role="status">{linkSent} Check your inbox (and spam folder) for <strong>{email}</strong>.</p>
          <button type="button" className="btn btn-block" onClick={() => switchMode("password")}>Back to sign in</button>
        </>
      ) : (
        <>
          <div className="field">
            <div className="field-label-row">
              <label htmlFor="login-email">Email</label>
              {remembered && <button type="button" className="link-btn" onClick={notYou}>Not you?</button>}
            </div>
            <input id="login-email" ref={emailRef} type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          {mode === "password" ? (
            <PasswordField ref={passwordRef} label="Password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)}
              aside={<Link href="/forgot-password">Forgot password?</Link>} />
          ) : (
            <>
              <p className="cap">We&apos;ll email you a link that signs you in. It works once and expires in 15 minutes.</p>
              <Turnstile action="email_link" onToken={setCaptcha} resetKey={captchaReset} />
            </>
          )}
          {error && (
            <p className="form-error" role="alert">
              {error}{waitS ? ` Try again in ${waitS}s.` : ""}{supportId ? ` Support code: ${supportId}` : ""}
            </p>
          )}
          {mode === "password" ? (
            <button type="submit" className="btn btn-primary btn-block" disabled={busy || !email || !password || !!waitS}>
              {busy ? "Signing in…" : "Sign in"}
            </button>
          ) : (
            <button type="submit" className="btn btn-primary btn-block" disabled={busy || !email || !!waitS || needsCaptcha}>
              {busy ? "Sending…" : "Email me a link"}
            </button>
          )}
          {slow && <p className="cap" role="status">{WAKING_UP}</p>}
          <div className="auth-or" aria-hidden="true">or</div>
          {mode === "password" ? (
            <button type="button" className="btn btn-block" onClick={() => switchMode("link")}>Email me a sign-in link instead</button>
          ) : (
            <button type="button" className="btn btn-block" onClick={() => switchMode("password")}>Use my password instead</button>
          )}
        </>
      )}
      <p className="cap login-links">
        New to HSF? <Link href={withNext("/signup", params.get("next"))}>Create a free account</Link>
      </p>
      <p className="cap auth-foot"><LegalLinks /> · You stay signed in for 30 days on this device</p>
      </AuthShell>
    </form>
  );
}
