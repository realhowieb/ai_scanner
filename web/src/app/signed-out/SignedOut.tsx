"use client";

// Where "Sign out" lands: confirms it worked, says what keeps arriving while signed out,
// and lets a shared computer forget the remembered email.
import Link from "@/components/Link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";

import { AuthShell } from "@/features/AuthForms";
import { forgetEmail, useLastEmail } from "@/lib/lastEmail";

export function SignedOut() {
  const pushWasOn = useSearchParams().get("push") === "off";
  const email = useLastEmail();
  const [forgotten, setForgotten] = useState(false);
  const forget = () => {
    forgetEmail();
    setForgotten(true);
  };
  return (
    <AuthShell title="You're signed out" subtitle="Your watchlists, alerts and settings are saved for next time."
      mark={<div className="signed-out-mark" aria-hidden="true">✓</div>}>
      <p className="notice" role="status">
        Alert emails and the daily emails you chose keep arriving.
        {pushWasOn ? " Push alerts are now off on this browser; turn them back on from Alerts after you sign in." : ""}
      </p>
      <Link href="/login" className="btn btn-primary btn-block">Sign back in</Link>
      {email && (
        <p className="cap login-links">
          On a shared computer? <button type="button" className="link-btn" onClick={forget}>Forget {email} on this device</button>
        </p>
      )}
      {forgotten && <p className="cap login-links" role="status">Done. This device won&apos;t fill in your email next time.</p>}
      <p className="cap login-links">Signed in somewhere else too? Sign in, then use <strong>Sign out everywhere</strong> in Account.</p>
    </AuthShell>
  );
}
