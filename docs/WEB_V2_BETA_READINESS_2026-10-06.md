# Web v2 beta readiness + live acceptance (2026-10-06)

## Verdict: NO-GO for an invited beta today. GO once the 4 steps below are done.

The app is ready. In an isolated environment the whole journey passes in a real browser
through the BFF, with the invite list on. Production isn't ready, for three reasons:

- production runs code without the endpoints the beta needs;
- there is no beta deployment yet;
- no signed-in check has run against production.

Each needs the owner (merge/promote, Render, test accounts). The steps to GO:

1. **Merge PR #8 into dev, then promote** (say "promote"). Production hsf-api runs main
   at `de341de` (its live OpenAPI is byte-identical to main's). It lacks
   `DELETE /v1/scans/{id}` (Cancel) and `GET /v1/alerts/types` (alert forms).
2. **Move hsf-api to Starter.** It is on the free plan: the first request today took
   **61.5 s** (sleep), and restarts lose queued scans.
3. **Create `hsf-web-beta` on Render** with `docs/WEB_V2_BETA_DEPLOY.md`: root `web`,
   Node 22, `npm ci && npm run build` / `npm start`, health `/api/healthz`, env
   `HSF_API_BASE_URL`, `NEXT_PUBLIC_STREAMLIT_URL` and `WEB_BETA_ALLOWED_EMAILS`.
4. **Live acceptance passes**:
   - `web/scripts/beta-smoke.mjs`;
   - `scripts/api_acceptance.py --allow-writes`;
   - `web/scripts/journey.mjs` against the beta.

   This needs dedicated test accounts set as environment variables here.

Not authorized in this run, and not done: merge, promote, hosting changes, creating the
Render service, switching traffic. Streamlit and production routing are unchanged.

## 1. Deployment parity (live, 2026-10-06 ~20:10 UTC)

| Check | Result | Evidence |
|---|---|---|
| hsf-api `/healthz` | PASS | 200. Cold 61.5 s (free-plan wake-up), then 0.89 s and 0.42 s. |
| hsf-api `/readyz` | PASS | 200, database ok, latest market scan 37.5 min old, 1.54 s |
| `/openapi.json` vs Web v2 contract | **FAIL (expected until promoted)** | Live: 61 operations; Web v2: 63. Missing on live: `DELETE /v1/scans/{scan_id}`, `GET /v1/alerts/types`. Schemas differing: `AlertType` and `AlertThreshold` (new), `ScanJob` (description only). |
| Live probes of the 2 endpoints | FAIL (not deployed) | Both answer 405 (no such method), not 401 |
| Which code is deployed | main `de341de` | The live OpenAPI equals the spec generated from `origin/main` byte for byte. |
| Missing from production | 73 commits | All of PR #8 (`origin/main..claude/latest-updates-l04ert`), including `b475bb9` (cancel and heartbeat) and `282939b` (alert types) |
| Endpoints the screens need (other than those 2) | PASS | Present live; protected ones answer 401 without a token |
| `scripts/api_acceptance.py` (read-only, no accounts) | 7 PASS / 1 FAIL / 8 BLOCKED | The FAIL is the 2 missing operations above, newly checked (`REQUIRED_OPERATIONS`). BLOCKED: every account journey and isolation (no `ACC_*`), CORS (no frontend origin), and the expired-token check (isolated only). |

## 2. Beta deployment prepared (not deployed)

| Item | Status |
|---|---|
| Render settings, env vars, order of operations, rollback | Written: `docs/WEB_V2_BETA_DEPLOY.md` |
| Node version | `web/.node-version` = 22; `engines` ≥ 20.9; CI uses 22 |
| Health check | New `/api/healthz`: liveness, never calls the API |
| HTTPS and cookies | Production mode sets `__Host-` cookies with `Secure`, `HttpOnly`, `SameSite=Lax`; Render provides TLS. `HSF_API_BASE_URL` is server-only and must be https. |
| Invite-only beta | New `WEB_BETA_ALLOWED_EMAILS`: only listed accounts can sign in. Anyone else gets "This preview… is invite-only…", no cookies, and their new API session is revoked. It is checked against the account's email from `/v1/me`, so signing in with a username works too. |
| Cold starts | Sign-in and every loading state now say "The HSF service may be waking up, which can take up to a minute" after 8 s. The BFF waits up to 75 s for the API. |
| Post-deploy smoke script | New `web/scripts/beta-smoke.mjs`. It checks HTTPS, health, security headers, the sign-in redirect, the BFF refusals, cross-site sign-in, and, with a test account, cookie flags and per-screen timings. No credentials or bodies are printed. |
| Rollback | Render Rollback or Suspend for the beta; Rollback plus a revert for hsf-api. Streamlit is never involved. |

## 3. Acceptance results

### Live (production)

| # | Check | Result | Dependency |
|---|---|---|---|
| 1–12 | Login and session; Today; Scanner; Stock Intelligence; custom scan complete, resume and cancel; save to watchlist; note; non-firing alert create, disable and delete; logout and login persistence; second-account isolation | **BLOCKED** | No deployed beta (step 3), no test accounts (`HSF_TEST_EMAIL`, `HSF_TEST_PASSWORD_FILE`, `HSF_TEST_EMAIL_2`, `ACC_*`). Cancel and alert creation also need step 1. |
| — | Web v2 production build (local) → **live** hsf-api, unauthenticated | PASS 9/9, 1 BLOCKED | `beta-smoke.mjs`: health, sign-in page, security headers, no `x-powered-by`, redirect to sign-in, BFF 401 `session_expired` with no-store, auth routes not proxied, cross-site sign-in refused. Signed-in checks BLOCKED (no account). |

### Isolated (this run's build; real API code, local Postgres, synthetic data, Chromium; not production evidence)

| Check | Result |
|---|---|
| `journey.mjs` with `WEB_BETA_ALLOWED_EMAILS` set (Pro invited, Premium as the second account) | PASS 12/12: login and reload; Scanner → save to a new watchlist; stock page "Already in it"; non-firing price alert; duplicate refused; note; invalid and duplicate tickers; alert off; logout → login persistence; 390 px layout; isolation; confirmed cleanup (0 test records left) |
| Uninvited account (Free) at 1440 px and 390 px | PASS: invite-only message, no session cookies, API log shows login → `/v1/me` → logout 204 |
| `beta-smoke.mjs` against the build | PASS 9/9 (signed-in part BLOCKED by design) |
| `api_acceptance.py` endpoint check against this branch's API | PASS (49 paths, both new operations present) |
| Custom scans (earlier today, same code): complete, resume, cancel then another, interrupted recovery | PASS 4/4 |

### Local

| Check | Result |
|---|---|
| Vitest | PASS: 79 tests in 7 files. New `tests/beta.test.tsx`: allowlist parsing; invited sign-in, with the email taken from the API; uninvited refused, revoked and without cookies; no `/v1/me` call when there is no list; health route; the 8-second wake-up hint. |
| Python | PASS: 2672 |
| Typecheck, lint, production build | PASS |
| OpenAPI contract (`web/openapi.json` and `schema.d.ts` match the API) | PASS |

No failures were found in the journey this run, so there were no fixes to cookies,
refresh, contract, scans or mutations. The changes are beta readiness: health, invite
list, cold-start messaging, the smoke script, and the acceptance script's operation check.

## 4. Response timings

| Request | Cold | Warm | Where |
|---|---|---|---|
| hsf-api `GET /healthz` | **61.5 s** (free-plan wake-up; 42.8 s earlier today) | 0.89 s, 0.42 s; 0.55 s in acceptance | live |
| hsf-api `GET /readyz` | — | 1.54 s | live |
| hsf-api `GET /openapi.json` | — | 2.43 s | live |
| Web v2 `/api/healthz` | 143 ms | 6–10 ms | local production build |
| Web v2 `/login` | — | 42 ms | local production build |
| BFF `GET /api/hsf/v1/me` without a session | — | 13 ms | local production build |
| Signed-in screen calls through the BFF (`/v1/me`, `/today`, `/scans/latest`, `/watchlists`, `/alerts`, `/alerts/types`) | BLOCKED | BLOCKED | needs a test account (`beta-smoke.mjs` measures them) |

On Starter there are no sleeps, so the 61.5 s cold start goes away. On a free web beta,
expect the first page after 15 idle minutes to take 30–60 s as well.

## 5. Screenshots

New: `docs/screenshots/web-v2/beta-not-invited-desktop.webp` and
`beta-not-invited-mobile.webp`. The screens themselves are unchanged since the
daily-workflow run; see that run's screenshots, listed in
`docs/WEB_V2_FOUNDATION_2026-10-06.md`.
