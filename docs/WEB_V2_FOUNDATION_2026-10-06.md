# Web v2 Foundation, Daily Workflow + Production API Acceptance Gate (2026-10-06)

## Current verdict (latest; supersedes earlier statements below)

> **Newer:** the beta readiness run (`docs/WEB_V2_BETA_READINESS_2026-10-06.md`) has the
> current deployment parity and the GO / NO-GO for an invited beta (NO-GO until PR #8
> is promoted, hsf-api is on Starter, the beta is deployed and live acceptance passes).

**Web v2 covers the daily customer journey: sign in → find a setup → inspect the stock →
save it to a watchlist → create an alert → return later.** It passes in an isolated
environment (real API code, local Postgres, synthetic data, the production build in
Chromium: 12/12 journey steps, 4/4 custom-scan checks) and in 73 frontend and 2672 Python
tests.

**Production: unauthenticated checks PASS; every signed-in check is BLOCKED** on test
accounts (no `ACC_*` credentials in this environment). The owner has signed in to Web v2
locally against production by hand; that isn't recorded evidence.

**Not live yet:** this branch (PR #8) isn't merged or promoted. Two API additions the new
screens use, `DELETE /v1/scans/{id}` (cancel) and `GET /v1/alerts/types`, exist only
here. Until it's promoted, Web v2 against production can't cancel a scan, and its alert
forms show a "needs the latest API" note instead of breaking. Streamlit is unchanged and
still serves all production traffic.

Evidence levels used below:

- **Local**: unit tests, typecheck, lint and build on this machine.
- **Isolated**: the real API code (`api.main:create_app`) on a local Postgres with
  synthetic data (tickers `T000`–`T299`, made-up prices), price downloads mocked, driven
  through the real Web v2 production build in Chromium. Not production evidence.
- **Live (production)**: requests to `https://hsf-api.onrender.com` from this
  environment, unauthenticated only.

## 1. P0: production API readiness (as of the latest check)

| Check | Result | Evidence / dependency |
|---|---|---|
| `GET /healthz`, `GET /readyz` | PASS | 200; database ok. The latest check found the newest market scan 178 minutes old (16:40 UTC). The first call of the day took 42.8 s (free-plan wake-up). |
| `GET /openapi.json` vs this branch | PASS with 2 expected gaps | Every live operation is in `web/openapi.json`. Missing on production, because they ship in this PR: `DELETE /v1/scans/{scan_id}`, `GET /v1/alerts/types`. |
| Data routes without a token | PASS | `/v1/me`, `/v1/today`, `/v1/scans/latest`, `/v1/stocks/AAPL`, `/v1/watchlists`, `/v1/alerts`, `/v1/alerts/events` → 401, with `X-Request-ID`. |
| Signed-in journey (sign-in, Today, Scanner, Stock, Custom scan, watchlists, alerts) | **BLOCKED** (live); PASS (isolated) | Needs dedicated test accounts as environment variables (`HSF_TEST_EMAIL`, `HSF_TEST_PASSWORD_FILE`, optional `HSF_TEST_EMAIL_2`) for `web/scripts/journey.mjs`, and the five `ACC_*` accounts for `scripts/api_acceptance.py`. |
| CORS | N/A by design; PASS | Web v2 calls the API from its server (BFF). An unknown origin's preflight gets no `Access-Control-Allow-Origin`. The owner deleted a misspelled `API_CORS_ORGINS`; `API_CORS_ORIGINS` isn't needed. |
| hsf-api configuration | PASS (owner screenshot) | `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `ANTHROPIC_API_KEY`, `APP_ENCRYPTION_KEY` are set, and `API_SCAN_WORKERS=1`. Values weren't viewed. |
| Hosting | FAIL (known) | hsf-api is on Render's free plan (owner confirmed): it sleeps after 15 min and has 512 MB. Restarts lose queued scans, and big lists can run out of memory. Move to Starter (P1-73). Not changed: no authorization to change hosting. |

## 2. What was built (`web/`)

- **Stack**: Next.js 16 (App Router), React 19, TypeScript 5.9 (strict); no UI framework.
  Design tokens from the design canvas (dark terminal look, cyan accent, Sora / IBM Plex
  Sans / IBM Plex Mono, self-hosted by `next/font`).
- **Typed API client** generated from `web/openapi.json` (`openapi-typescript` →
  `src/api/schema.d.ts`, used through `openapi-fetch`). `scripts/export_openapi.py`
  regenerates the contract; CI fails when either file is stale.
- **BFF session** (`src/server/`): sign-in and sign-out routes; HttpOnly + Secure +
  SameSite=Lax `__Host-` cookies; server-side refresh before expiry and one retry after
  a 401; single-flight refresh; `session_expired` → sign in again with `?next=`;
  same-origin check on writes; only `/v1/` data routes proxied (not `/v1/auth/*`);
  request IDs end to end, shown as "Support code" on errors. Security headers: CSP
  with `connect-src 'self'`, `frame-ancestors 'none'`, HSTS, nosniff and others.
- **App shell**: responsive header with nav (Today, Scanner, Custom scan), ticker
  search, account menu (plan label from `/v1/me`, "Open classic app", sign out),
  disclaimer footer.
- **Today** (`/v1/today`): market phase, Before the open, Top setups, After the close,
  last-session recap; scan times and freshness; stale-scan banner (older than 6 h,
  the web's rule; not applied to session scans); per-section failure notices (other
  sections still render); empty-scan and no-qualifying states; Pro lock with upgrade
  when the API says `locked`.
- **Scanner** (`/v1/scans/latest`): ranked table on wide screens, cards on phones;
  setup-type chips, minimum score, page size and pagination in the URL; plan cap note
  with upgrade when `limited`; PreBreakout locked below Premium (no request made);
  PreBreakout probability shown only when the plan includes it.
- **Custom scan** (`POST /v1/scans`, `GET /v1/scans/{id}`, `GET /v1/scans`): universe,
  session and filter controls from `/v1/me` entitlements; result count capped by the
  plan's `max_results` from the API; queued/running/complete/failed states with the
  API's phase, symbol count and elapsed time; polling every 3 s (clamped 2–5 s);
  resumes the account's active scan after navigation or reload (from the API, nothing
  stored in the browser); stops on completion, failure, unmount or sign-out; 403 shows
  the server's reason with an upgrade button, 409 follows the running scan, 429 and
  503 respect `Retry-After` with a countdown.
- **Stock Intelligence** (`/v1/stocks/{ticker}`): HSF Score with movement, components,
  signals, breakout score, PreBreakout (Premium), earnings, reasons, risks, what to
  watch, historical context (locked below Pro when `historical_locked`), your lists and
  alerts, recent history, a daily candlestick and volume chart drawn from the
  returned bars. The price reads "at the HH:MM ET scan, not a live quote". Banners
  cover "not in the latest scan" (with the historical fallback), "scanned but not a
  setup", "no recorded score" and "no cached bars".

Nothing in `web/` reimplements scanner logic: scores, ranking, caps and redaction all
come from the API.

## 3. Validation

### Local

| Check | Result |
|---|---|
| `npm run typecheck` (strict, `noUncheckedIndexedAccess`) | PASS |
| `npm run lint` (eslint-config-next core-web-vitals + TypeScript) | PASS, 0 warnings |
| `npm test` (Vitest) | PASS, 48 tests in 5 files |
| `npm run build` (production) | PASS |
| `npm audit --omit=dev` | PASS, 0 vulnerabilities |
| `npm audit` (all) | 5 high, dev-only: `braces` via `eslint-config-next` → `fast-glob` → `micromatch`. No fixed version exists (advisory range `*`); lint tool only, never shipped. CI audits runtime dependencies. |
| Python suite (with Postgres) | PASS, 2668 passed (guards accept the new tree) |
| actionlint + zizmor on `.github/workflows/web.yml` | PASS |

The Vitest suites cover, as the run asked:

- **Login, refresh, logout, expired sessions** (`tests/bff.test.ts`): tokens only in
  HttpOnly cookies; refresh before forwarding; one retry after a 401; failed refresh →
  401 `session_expired` and cleared cookies; no retry loop; logout revokes and clears
  even when the API is down; wrong password, lockout `Retry-After` and cross-site
  sign-in refused.
- **Concurrent refresh** (`tests/bff.test.ts`): 6 parallel requests produce one
  rotation and the same cookie; a request right after reuses it.
- **Server-driven entitlements** (`tests/screens.test.tsx`): Free, Pro and Premium
  custom-scan controls; PreBreakout lock; probability visibility; plan cap.
- **Today partial failures** (`tests/today.test.tsx`): one section failing, Pro lock,
  empty or no-qualifying scans, stale banner, session scans not marked stale.
- **Scanner filters and caps** (`tests/screens.test.tsx`): exact query sent, pagination
  within the cap, upgrade note, outage with support code.
- **Custom-scan lifecycle and polling cleanup** (`tests/scanJob.test.tsx`): queued →
  running → complete then no more polls; failure stops; resume after reload; 429 while
  polling waits `Retry-After`; unmount and 401 stop polling; 409 follows; 503/429
  countdown; interval clamp.
- **Stock fallback and redaction** (`tests/screens.test.tsx`): not a live quote,
  missing bars, historical fallback, no score, Pro and Premium locks, chart from bars.
- **Retry-After parsing** (`tests/client.test.ts`): seconds, HTTP date, cap, junk;
  request IDs; one session-expired redirect for many failures.

### Isolated (local API + Postgres, synthetic data, real Web v2 build in Chromium)

| Check | Result |
|---|---|
| Sign-in through `/api/auth/login` sets two `__Host-` cookies (HttpOnly, Secure, SameSite=Lax); the body has no token | PASS |
| Cross-site sign-in (`Origin: https://evil.example`) | PASS (403) |
| `/v1/me`, `/v1/scans/latest`, `/v1/today`, `/v1/stocks/T003` through the BFF; `X-Request-ID` returned and logged by the API | PASS |
| `/api/hsf/v1/auth/refresh` (auth route through the proxy) | PASS (404) |
| 5 concurrent requests with only a refresh cookie | PASS: all 200, one rotation, identical new token |
| Old refresh token replayed after the API's 30 s grace | PASS: API treats it as theft and revokes every session; BFF answers 401 `session_expired` and clears cookies; the rotated token stops working too |
| Browser journey for Pro and Free (sign-in → Today → Scanner → Stock → Custom scan S&P 500 to completion) at 1440×1000 and 390×844 | PASS, no console errors |
| Cookies not readable by page scripts; no token in `document.cookie`, localStorage or sessionStorage | PASS |
| Pre-market Today (API clock fixed at 8:50 AM ET) for Pro (movers) and Free (locked) | PASS (see caveat) |
| Custom scan completed (120 synthetic S&P 500 tickers, mocked downloads) | PASS, about 12 s |

Caveats: the data is synthetic, so the scores and prices in screenshots mean nothing.
The fixed clock moves only the API's Today builder; browser freshness labels use the
real time, so pre-market screenshots show the 11:51 AM test scan as "51m ago". The
upgrade button calls `POST /v1/billing/checkout` but wasn't followed to Stripe.

### Production

Unauthenticated checks PASS (section 1); everything signed in is BLOCKED on test
accounts. Nothing was written to production, and no production account was used.

## 4. Screenshots

`docs/screenshots/web-v2/` (synthetic isolated data; mobile images downscaled to 1×
and cut at 2,600 px):

| Screen | Desktop | Mobile |
|---|---|---|
| Today (Pro, market open) | `today-desktop.webp` | `today-mobile.webp` |
| Today (Pro, pre-market) | `premarket-pro-today-desktop.webp` | `premarket-pro-today-mobile.webp` |
| Today (Free, pre-market, Pro locks) | `premarket-free-today-desktop.webp` | `premarket-free-today-mobile.webp` |
| Scanner (Pro, cap note) | `scanner-desktop.webp` | `scanner-mobile.webp` |
| Custom scan (Pro, results) | `custom-desktop.webp` | `custom-mobile.webp` |
| Stock Intelligence (Pro, chart) | `stock-desktop.webp` | `stock-mobile.webp` |
| Stock Intelligence (Free, historical locked) | `free-stock-desktop.webp` | `free-stock-mobile.webp` |

## 5. Deployment and configuration requirements (foundation run; current list in "Production blockers" below)

Web v2 isn't deployed; that needs the owner. Steps are in `web/README.md` (Render web
service, root `web`, `npm ci && npm run build`, `npm start`, health `/login`, env
`HSF_API_BASE_URL`, `NEXT_PUBLIC_STREAMLIT_URL`, `NODE_VERSION=22`; no secrets).

Before Web v2 is used against production:

1. **P1-73** (owner): on `hsf-api` set `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`,
   `ANTHROPIC_API_KEY`, `APP_ENCRYPTION_KEY` (same value as Streamlit); don't set
   `ALPACA_BASE_URL`; move to Starter.
2. **P1-63** (owner + run): network access is done; provide the five `ACC_*` test
   accounts as environment variables, then run
   `scripts/api_acceptance.py --allow-writes`, and this frontend's
   `npm run screenshots` against a deployed Web v2 beta.
3. Create the Render service for Web v2 (beta first, from `dev`). Keep the Streamlit
   domain and routing as they are.
4. `API_CORS_ORIGINS`: leave empty (the BFF doesn't need it).

## 6. Remaining gaps (as of the foundation run; see the daily-workflow run for updates)

- Screens still only in Streamlit: Day Trader, Market Brief, scan history and track
  record, earnings, AI notes and chat, journal and paper trading, settings and billing,
  sign-up, password reset, email verification. Their APIs exist. (Watchlists and Alerts
  were added in the daily-workflow run below.)
- CSV export (client-side from rows).
- After Stripe checkout, customers return to the Streamlit app (`APP_SUCCESS_URL`).

## 7. Recommended next run (superseded by the daily-workflow run's recommendation)

**"Web v2 beta deploy + production acceptance"**, once P1-73 and P1-63 are done:
deploy `hsf-web-beta` from `dev`, run `scripts/api_acceptance.py` and
`npm run screenshots` against production with the test accounts, and fix what they
find. In parallel or right after: **"Web v2 screens, part 2"**: watchlists and alerts
(including add-to-watchlist and price alert from the Stock and Scanner pages), then
Market Brief and Day Trader. These cover most of what customers use daily and have
APIs already.

## Addendum: stuck "Queued" custom scans (2026-10-06, later)

The owner's first production custom scan stayed at "Queued, waiting for a scanner".
The API had accepted it, but its single scan worker never started it: either the worker
was busy, or (most likely, since the Alpaca keys turned out to be set) hsf-api restarted
on the free plan and lost its in-process queue. Such a job used to block the account
for 60 minutes. Three fixes:

| Fix | What changed | Evidence |
|---|---|---|
| Cancel | `DELETE /v1/scans/{scan_id}` (owner only): the job reads `failed` / "Cancelled." and the account can start another at once. A queued job never starts; a running one stops at its next progress step, so its results aren't saved and the worker is freed. | `tests/test_api_scans.py` (route: owner only, slot freed, 401/422; Postgres: queued and running cancel, nothing overwrites it, worker released). Isolated end to end: queued cancel at once, another user's scan → 404, running cancel freed the worker for the next scan. |
| Faster recovery after a restart | Each API process marks the jobs it holds (queued or running) as alive every 30 s. A job left by a dead process expires after 3 minutes (was 60 queued / 20 running) with "The scan was interrupted because the service restarted. Start it again."; a job waiting in line keeps its heartbeat and never expires. | Postgres tests (heartbeat keeps a 30-min wait queued; no heartbeat → failed). Isolated end to end: API process killed mid-queue → job cleared 170 s after the restart, new scan accepted. |
| Clearer waiting in Web v2 | The progress card shows "Waiting for 1m 05s" / "Running for …", a **Cancel scan** button, and after 60 s queued a note saying why it may be slow and suggesting a smaller list. A cancelled scan shows "Scan cancelled" with a fresh start. The page scrolls to the status when a scan starts (it was off-screen below the Start button). | `tests/scanJob.test.tsx`, `tests/screens.test.tsx` (54 Vitest tests in all). Chromium check at 1440 px and 390 px. |

Still needed for large lists in production: Starter for hsf-api (P1-73; the keys are set).


## Daily workflow run: Watchlists, Alerts and live journey validation (2026-10-06, later)

### Built

| Area | What |
|---|---|
| Watchlists screen (`/watchlists`) | List, create (optionally as default), rename, make default and delete. Delete is confirmed and names what goes. Each list shows its tickers with notes: add one or many (the server reports added, already present and invalid), edit notes (500 characters), and remove a ticker (confirmed). Tickers open Stock Intelligence. The HSF Score and price come from **one** latest-scan request, labelled with the scan time, and appear only for tickers among the plan's ranked rows; others say "Not ranked in the latest scan". No per-ticker requests and no live quotes. "N of 50" shows the limit. |
| Save to watchlist | A "+" on every Scanner row (table and phone cards) and a button on Stock Intelligence. The dialog lists your watchlists and marks the ones that already hold the ticker (from `/v1/stocks`). It can also create a new list and save in one step, and it reports "Saved" or "Already in it" from the server's answer. |
| Alerts screen (`/alerts`) | Your alerts with "Using N of M on your plan", each with a description, last fired and created times, Turn on/off and Delete (confirmed). Recently fired events show, with an empty state. The New alert form is built from the API's rules. At the limit it is replaced by a clear next step (Upgrade from Free or Pro; "delete one" on Premium). Delivery: an email toggle (`/v1/me/email-preferences`) for Pro+, Upgrade for Free, and the note "Phone push notifications aren't available yet." |
| Price alert from Stock Intelligence | A dialog prefilled with the ticker, direction "Rises above" and the last scan price, labelled with the scan time, "not a live quote". It checks the alert quota first. |
| API: `GET /v1/alerts/types` | Read-only. Each alert type with its label, description, whether it needs a ticker, the threshold minimum (exclusive or not), maximum and default, and the allowed directions. These are the same `ALERT_RULES` that `POST /v1/alerts` enforces, so forms never copy the rules. Before this API is deployed, the forms show a note instead of breaking. |
| Shared | An accessible dialog (`role="dialog"`, `aria-modal`, focus moves in and back to the opener, Tab stays inside, Escape and backdrop close). Mutations update the UI only after the server confirms, then reload the affected data. Failures keep the previous state and show the message with its support code. Nav adds Watchlists and Alerts and wraps on phones. Plan labels still come from `/v1/me`. |

### Validation

**Local**
| Check | Result |
|---|---|
| Vitest | PASS: 73 tests in 6 files. `tests/workflows.test.tsx` (19) runs against an in-memory fake that follows the API's rules. It covers watchlist CRUD; duplicate name inside the dialog; rename and default; delete with Keep it and Delete; failed delete keeping the list and showing the support code; add with already present and invalid; notes; remove; one scan request for scores. Save from a Scanner row and from Stock Intelligence (already present, create and save). Alert validation before sending; duplicate (409); server 422; web defaults; on/off; delete; failed toggle; limit; email toggle; no push promise; empty history; session expiry handing over once; the not-deployed-yet fallback; Escape and focus return. |
| Python | PASS: 2672 passed. New: `/v1/alerts/types` matches the rules POST enforces (every advertised minimum, and anything below it refused); the web test fixture equals the API's output. |
| Typecheck, lint, production build, OpenAPI contract (`web/openapi.json` and `src/api/schema.d.ts` regenerated, 49 paths) | PASS |

**Isolated** (`web/scripts/journey.mjs` through the BFF; Pro account plus a Premium account for isolation)
| Step | Result |
|---|---|
| Sign in; session survives reload and navigation | PASS |
| Find a setup on the Scanner, save it to a new watchlist from the row | PASS |
| Stock Intelligence shows the list; saving again → "Already in it" | PASS |
| Price alert from the stock page at a non-firing $999,999 | PASS |
| Duplicate alert refused with the server's message and support code | PASS |
| Watchlist: note saved; duplicate and invalid tickers reported | PASS |
| Alerts: capacity shown; turned off | PASS |
| Return later: sign out (protected pages go to sign-in), sign back in, list, note and alert state all still there | PASS |
| Phone layout (390 px): watchlist, alerts, save dialog, no sideways scroll | PASS |
| A second account sees neither the list nor the alert | PASS |
| Clean up through the UI (confirmed deletes); database check: 0 test lists, 0 test alerts left | PASS |
| Custom scan: progresses and completes with results | PASS |
| Reload or second tab resumes the running scan instead of starting another | PASS |
| Cancel frees the account to start another scan | PASS |
| API process killed mid-scan → the page shows "The scan was interrupted because the service restarted. Start it again." (177 s after the API came back) | PASS |

403, 409, 429 and 503 UI states are covered by unit tests (`tests/scanJob.test.tsx`,
`tests/screens.test.tsx`, `tests/workflows.test.tsx`), not forced in the isolated run.

**Live (production)**: section 1 at the top. Unauthenticated checks PASS; the signed-in
journey is BLOCKED on test accounts.

### Screenshots (`docs/screenshots/web-v2/`, synthetic isolated data)

| Screen | Desktop | Mobile |
|---|---|---|
| Watchlists (ticker, note, score from the scan) | `watchlists-desktop.webp` | `watchlists-mobile.webp` |
| Alerts (capacity, off, empty history, email toggle) | `alerts-desktop.webp` | `alerts-mobile.webp` |
| Save to watchlist from the Scanner | `scanner-save-desktop.webp` | `save-dialog-mobile.webp` |
| Price alert from Stock Intelligence | `stock-price-alert-desktop.webp` | |
| Stock Intelligence with its list and alert | `stock-actions-desktop.webp` | |
| Custom scan interrupted by a restart | `custom-scan-interrupted-desktop.webp` | |

### Production blockers (exact, current)

1. **Merge PR #8 into dev and promote to main.** Render deploys hsf-api from main, and it
   needs cancel and `/v1/alerts/types`. Promotion happens only when the owner says "promote".
2. **Move hsf-api to Starter** (P1-73). The keys are already set.
3. **Test accounts for live acceptance** (P1-63): `ACC_FREE`, `ACC_PRO`, `ACC_PRO2`,
   `ACC_PREMIUM`, `ACC_ADMIN` (`_EMAIL` / `_PASSWORD`) for `scripts/api_acceptance.py`,
   and one or two of them for `web/scripts/journey.mjs` (`HSF_TEST_EMAIL`,
   `HSF_TEST_PASSWORD_FILE`, `HSF_TEST_EMAIL_2`), set as environment variables here.
4. **A deployed Web v2** (Render web service from `dev`, steps in `web/README.md`) to
   run the journey against, rather than a laptop.

### Recommended next run

**"Web v2 beta deploy + live acceptance"** once blockers 1–3 are done. Deploy
`hsf-web-beta`, then run `scripts/api_acceptance.py --allow-writes` and
`web/scripts/journey.mjs` against production with the test accounts, and fix what they
find. Next screens after that: Market Brief and Day Trader.
