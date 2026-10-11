"use client";

import Link from "next/link";
import { useEffect } from "react";

import { reportError, type ErrorKind } from "@/lib/reportError";

/** What a page shows when it crashes while rendering: the rest of the app keeps working. */
export function PageError({ error, retry, kind = "boundary", home = "/today" }: {
  error: Error & { digest?: string };
  retry: () => void;
  kind?: ErrorKind;
  home?: string;
}) {
  useEffect(() => {
    reportError(error, kind);
  }, [error, kind]);
  return (
    <div className="card narrow" role="alert">
      <h1 className="h1">Something went wrong on this page</h1>
      <p className="cap">We&apos;ve been told about it. Trying again usually fixes it.</p>
      <div className="row-actions">
        <button type="button" className="btn btn-primary" onClick={() => retry()}>Try again</button>
        <Link className="btn" href={home}>Go to Today</Link>
      </div>
      {error.digest && <p className="cap mono">Support code: {error.digest}</p>}
    </div>
  );
}
