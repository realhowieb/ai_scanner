"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { messageFrom, newRequestId } from "@/api/client";
import { safeNext } from "@/lib/nextPath";
import { parseRetryAfter } from "@/lib/retryAfter";

const CLASSIC = process.env.NEXT_PUBLIC_STREAMLIT_URL || "https://hsfinestai.streamlit.app";

export function LoginForm() {
  const params = useSearchParams();
  const next = safeNext(params.get("next"));
  const expired = params.get("expired") === "1";
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [supportId, setSupportId] = useState<string | null>(null);
  const [waitS, setWaitS] = useState<number | null>(null);

  useEffect(() => {
    if (!waitS) return;
    const t = setTimeout(() => setWaitS((s) => (s && s > 1 ? s - 1 : null)), 1000);
    return () => clearTimeout(t);
  }, [waitS]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setSupportId(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "content-type": "application/json", "x-request-id": newRequestId() },
        body: JSON.stringify({ email, password }),
      });
      if (res.ok) {
        // Confirm the browser kept the session cookie before leaving this page;
        // otherwise every page would bounce straight back here with no explanation.
        const check = await fetch("/api/hsf/v1/me", { credentials: "same-origin", headers: { "x-request-id": newRequestId() } });
        if (check.ok) {
          window.location.assign(next);
          return;
        }
        setError(check.status === 401
          ? "You're signed in, but this browser didn't keep the sign-in cookie. Allow cookies for this site; when running locally, open it at http://localhost:3000 (not an IP address)."
          : messageFrom(await check.json().catch(() => null), check.status));
        setSupportId(check.headers.get("x-request-id"));
        setBusy(false);
        return;
      }
      const body: unknown = await res.json().catch(() => null);
      setError(messageFrom(body, res.status));
      setSupportId(res.status >= 500 ? res.headers.get("x-request-id") : null);
      if (res.status === 429) setWaitS(parseRetryAfter(res.headers.get("retry-after")) ?? 60);
    } catch {
      setError("Couldn't reach HSF. Check your connection and try again.");
    }
    setBusy(false);
  };

  return (
    <form className="card narrow login" onSubmit={submit} noValidate>
      <p className="brand brand-lg">HSFinest<span>.AI</span></p>
      <h1 className="h1">Sign in</h1>
      {expired && <p className="notice" role="status">Your session ended. Sign in again to continue.</p>}
      <label className="field">
        <span>Email</span>
        <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
      </label>
      <label className="field">
        <span>Password</span>
        <input type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
      </label>
      {error && (
        <p className="form-error" role="alert">
          {error}{waitS ? ` Try again in ${waitS}s.` : ""}{supportId ? ` Support code: ${supportId}` : ""}
        </p>
      )}
      <button type="submit" className="btn btn-primary btn-block" disabled={busy || !email || !password || !!waitS}>
        {busy ? "Signing in…" : "Sign in"}
      </button>
      <p className="cap login-links">
        <a href={CLASSIC}>Create an account</a> · <a href={CLASSIC}>Forgot password?</a>
      </p>
      <p className="cap">Same account as the HSF web app.</p>
    </form>
  );
}
