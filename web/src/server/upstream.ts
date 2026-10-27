import "server-only";

import type { Upstream } from "./bff";
import { withClientAddress } from "./clientAddress";
import { UPSTREAM_TIMEOUT_MS, apiBaseUrl } from "./config";

export const upstream: Upstream = (path, init) =>
  fetch(`${apiBaseUrl()}${path}`, { ...init, cache: "no-store", redirect: "manual", signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });

/** upstream for one browser request: carries the visitor's address (src/server/clientAddress.ts). */
export const upstreamFor = (req: Request): Upstream => withClientAddress(req, upstream);
