"use client";

// Where the emailed sign-in link lands (/login/link?token=). It waits for a click before
// using the one-time token, so email scanners that open links can't use it up.
import Link from "@/components/Link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import type { FormEvent } from "react";

import { AuthShell, postAuth } from "@/features/AuthForms";
import { messageFrom } from "@/api/client";
import { WAKING_UP, useSlow } from "@/components/ui";

import { finishSignIn } from "../LoginForm";

export function EmailLinkSignIn() {
  const token = useSearchParams().get("token") || "";
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const slow = useSlow(busy);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const o = await postAuth("email-link-confirm", { token });
    if (o.ok) {
      const failed = await finishSignIn("/today").catch(() => ({ error: "Couldn't reach HSF. Check your connection and try again.", supportId: null }));
      if (!failed) return;
      setError(failed.error + (failed.supportId ? ` Support code: ${failed.supportId}` : ""));
    } else {
      setError(o.status === 0 ? String(o.body?.detail) : messageFrom(o.body, o.status));
    }
    setBusy(false);
  };
  return (
    <form onSubmit={submit} noValidate>
      <AuthShell title="Sign in to HSF" subtitle="You opened a sign-in link from your email.">
        {!token ? (
          <p className="form-error" role="alert">This link is incomplete. Ask for a new one from the sign-in page.</p>
        ) : (
          <>
            {error && <p className="form-error" role="alert">{error}</p>}
            <button type="submit" className="btn btn-primary btn-block" disabled={busy}>{busy ? "Signing in…" : "Continue to HSF"}</button>
            {slow && <p className="cap" role="status">{WAKING_UP}</p>}
          </>
        )}
        <p className="cap login-links">
          <Link href="/login?mode=link">Send a new link</Link> · <Link href="/login">Sign in with a password</Link>
        </p>
      </AuthShell>
    </form>
  );
}
