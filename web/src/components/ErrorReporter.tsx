"use client";

import { useEffect } from "react";

import { reportError } from "@/lib/reportError";

/** Reports errors that never reach a page's error screen: ones thrown in click handlers,
 * timers and failed promises nobody waited on. Renders nothing. */
export function ErrorReporter() {
  useEffect(() => {
    const onError = (e: ErrorEvent) => reportError(e.error ?? e.message, "window");
    const onRejection = (e: PromiseRejectionEvent) => {
      const reason = e.reason as { name?: string } | null;
      if (reason?.name === "AbortError") return; // a page we left cancelled its own request
      reportError(e.reason, "promise");
    };
    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onRejection);
    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onRejection);
    };
  }, []);
  return null;
}
