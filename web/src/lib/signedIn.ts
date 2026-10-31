"use client";

import { useEffect, useState } from "react";

let cached: Promise<boolean> | null = null;

function check(): Promise<boolean> {
  cached ??= Promise.resolve()
    .then(() => fetch("/api/auth/status", { cache: "no-store", credentials: "same-origin" }))
    .then((r: Response | undefined) => (r?.ok ? r.json() : null))
    .then((b: { signed_in?: boolean } | null) => !!b?.signed_in)
    .catch(() => false);
  return cached;
}

/** Whether this browser has an HSF session, for public pages that members also open.
 * False until the check answers, so signed-out visitors (and crawlers) see the default. */
export function useSignedIn(): boolean {
  const [signedIn, setSignedIn] = useState(false);
  useEffect(() => {
    let live = true;
    void check().then((v) => { if (live) setSignedIn(v); });
    return () => { live = false; };
  }, []);
  return signedIn;
}
