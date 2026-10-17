"use client";

import { useEffect, useState } from "react";

import { signOut } from "@/session/SessionProvider";

// Browsers keep their own sign-in, so the account here can differ from the one that
// opened Stripe (for example a Mac web app). Plain fetch: a signed-out browser must
// stay on this page rather than be sent to the login screen.
export function SignedInAs() {
  const [email, setEmail] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    fetch("/api/hsf/v1/me", { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : null))
      .then((me: { email?: unknown } | null) => { if (live && typeof me?.email === "string") setEmail(me.email); })
      .catch(() => undefined);
    return () => { live = false; };
  }, []);
  if (!email) return null;
  return <p className="notice" role="status">
    Signed in here as <strong>{email}</strong>. If you paid from a different account, switch back to your HSF app
    or <button type="button" className="link-btn" onClick={() => void signOut()}>sign out here</button>.
  </p>;
}
