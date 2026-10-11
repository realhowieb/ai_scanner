"use client";

import { useEffect } from "react";

import { reportError } from "@/lib/reportError";

// Replaces the root layout when it fails, so it brings its own document and plain styles.
export default function GlobalError({ error, retry }: { error: Error & { digest?: string }; retry: () => void }) {
  useEffect(() => {
    reportError(error, "global");
  }, [error]);
  return (
    <html lang="en">
      <body style={{ margin: 0, minHeight: "100vh", display: "grid", placeItems: "center", background: "#0b0f14", color: "#e7edf5", fontFamily: "system-ui, sans-serif" }}>
        <main style={{ maxWidth: 420, padding: 24, textAlign: "center" }} role="alert">
          <title>Something went wrong · HSFinest.AI</title>
          <h1 style={{ fontSize: 22 }}>Something went wrong</h1>
          <p style={{ opacity: 0.75 }}>We&apos;ve been told about it. Trying again usually fixes it.</p>
          <button type="button" onClick={() => retry()}
            style={{ minHeight: 40, padding: "0 16px", borderRadius: 10, border: 0, background: "#4cc2ff", color: "#0b0f14", fontWeight: 600, cursor: "pointer" }}>
            Try again
          </button>
          {error.digest && <p style={{ opacity: 0.6, fontFamily: "monospace", fontSize: 12 }}>Support code: {error.digest}</p>}
        </main>
      </body>
    </html>
  );
}
