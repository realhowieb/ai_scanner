// Crash reports from the browser to the API (POST /v1/client-errors), which logs them and
// forwards them to Sentry when it's configured. Best effort and bounded: the same message
// is sent once and at most a few per page load. Sends no account, token or query string.
import { newRequestId } from "@/api/client";

export type ErrorKind = "boundary" | "global" | "window" | "promise";

const MAX_PER_LOAD = 5;
const sent = new Set<string>();

function text(v: unknown, max: number): string | undefined {
  if (typeof v !== "string" || !v) return undefined;
  return v.slice(0, max);
}

export function reportError(error: unknown, kind: ErrorKind = "boundary"): void {
  try {
    if (typeof window === "undefined" || sent.size >= MAX_PER_LOAD) return;
    const e = error as { message?: unknown; stack?: unknown; digest?: unknown } | null;
    const message = text(e?.message, 500) ?? text(typeof error === "string" ? error : undefined, 500) ?? "Unknown error";
    if (sent.has(message)) return;
    sent.add(message);
    void fetch("/api/public/v1/client-errors", {
      method: "POST",
      keepalive: true,
      credentials: "omit",
      headers: { "content-type": "application/json", "x-request-id": newRequestId() },
      body: JSON.stringify({
        message,
        kind,
        path: window.location.pathname.slice(0, 200),
        digest: text(e?.digest, 64),
        stack: text(e?.stack, 2000),
      }),
    }).catch(() => undefined);
  } catch {
    /* reporting never adds a second failure */
  }
}

/** For tests. */
export function resetReportedErrors(): void {
  sent.clear();
}
