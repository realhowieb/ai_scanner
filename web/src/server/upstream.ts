import "server-only";

import type { Upstream } from "./bff";
import { UPSTREAM_TIMEOUT_MS, apiBaseUrl } from "./config";

export const upstream: Upstream = (path, init) =>
  fetch(`${apiBaseUrl()}${path}`, { ...init, cache: "no-store", redirect: "manual", signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
