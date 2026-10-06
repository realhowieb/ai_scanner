// Single-flight refresh. Refresh tokens are single use and the API treats a late
// replay as theft (it revokes every session), so concurrent requests holding the
// same refresh token must share ONE rotation. Requests that arrive just after it
// settled reuse its result for a few seconds instead of replaying the old token.
import { createHash } from "node:crypto";

import type { TokenPair } from "./cookies";

export type RefreshOutcome = { ok: true; pair: TokenPair } | { ok: false; status: number };
export type RefreshFn = (refreshToken: string) => Promise<RefreshOutcome>;

const REUSE_MS = 10_000;
const inflight = new Map<string, Promise<RefreshOutcome>>();
const recent = new Map<string, { at: number; outcome: RefreshOutcome }>();

function key(token: string): string {
  return createHash("sha256").update(token).digest("hex");
}

export async function refreshOnce(refreshToken: string, doRefresh: RefreshFn, nowMs = Date.now()): Promise<RefreshOutcome> {
  const k = key(refreshToken);
  const done = recent.get(k);
  if (done && nowMs - done.at < REUSE_MS) return done.outcome;
  const running = inflight.get(k);
  if (running) return running;
  const p = doRefresh(refreshToken)
    .catch((): RefreshOutcome => ({ ok: false, status: 502 }))
    .then((outcome) => {
      // Only a definite answer is remembered; a network failure may be retried.
      if (outcome.ok || outcome.status === 401) recent.set(k, { at: Date.now(), outcome });
      return outcome;
    })
    .finally(() => inflight.delete(k));
  inflight.set(k, p);
  prune(nowMs);
  return p;
}

function prune(nowMs: number): void {
  if (recent.size < 500) return;
  for (const [k, v] of recent) if (nowMs - v.at >= REUSE_MS) recent.delete(k);
}

export function _resetRefreshState(): void {
  inflight.clear();
  recent.clear();
}
