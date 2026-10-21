import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

import { clearApiCache } from "@/hooks/useApi";

afterEach(() => {
  cleanup();
  clearApiCache(); // remembered page data must not leak between tests either
  if (typeof window !== "undefined") window.sessionStorage.clear(); // the cached /v1/me answer must not leak between tests
});

// jsdom has no ResizeObserver; Radix popovers measure their arrow with it.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;
}
