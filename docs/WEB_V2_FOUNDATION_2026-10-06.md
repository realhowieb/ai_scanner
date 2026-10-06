# Web v2 Foundation + Production API Acceptance Gate (2026-10-06)

**Result: Web v2 foundation built and verified locally and in an isolated environment.
Production: unauthenticated checks PASS (after the owner allowed `hsf-api.onrender.com`
in this environment's network settings); signed-in acceptance BLOCKED (no test
credentials, no Render access). Streamlit is unchanged and still
serves all production traffic.

Evidence levels used below:

- **Local**: unit tests, typecheck, lint and build on this machine.
- **Isolated**: the real API code (`api.main:create_app`) on a local Postgres with
  synthetic data (tickers `T000`–`T299`, made-up prices), price downloads mocked, driven
  through the real Web v2 production build in Chromium. Not production evidence.
- **Production**: none. Every production check is BLOCKED.

## 1. P0: production API readiness

| Check | Result | Evidence / dependency |
|---|---|---|
| `GET /healthz` | PASS | 200 (first call 42.8 s: the free plan was asleep and woke up; see Hosting). |
| `GET /readyz` | PASS | 200 `{"ok":true,"database":"ok"}`, latest market scan 19.8 minutes old. |
| `GET /openapi.json` | PASS | Live spec is **identical** to `web/openapi.json` (48 paths, every schema equal), so the generated client matches production. |
| Data routes without a token | PASS | `/v1/me`, `/v1/today`, `/v1/scans/latest`, `/v1/stocks/AAPL` → 401. Responses carry `X-Request-ID`. |
| Endpoints for the first three screens | Signed-in: BLOCKED (no test accounts); PASS (isolated) | `/v1/auth/login`, `/refresh`, `/logout`, `/v1/me`, `/v1/today`, `/v1/scans/latest`, `POST /v1/scans`, `GET /v1/scans/{id}`, `GET /v1/scans`, `/v1/stocks/{ticker}`, `/v1/watchlists`, `/v1/billing/checkout`: all exercised through the BFF against the isolated API. |
| CORS for the frontend origin | N/A by design; PASS (no CORS for an unknown origin: preflight 405, no `Access-Control-Allow-Origin`) | Web v2 calls the API from its own server (BFF), never from the browser, so `API_CORS_ORIGINS` is **not needed** and should stay empty. |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | BLOCKED | No Render access from here. Owner item P1-73 (still open per the backlog): needed for custom scans, the US market list, Day Trader and quotes. |
| `ANTHROPIC_API_KEY` | BLOCKED | P1-73. Not used by the three screens; `/v1/ai/*` answers 503 without it. |
| `APP_ENCRYPTION_KEY` | BLOCKED | P1-73 (same value as Streamlit). Not used by the three screens (paper trading). |
| `API_SCAN_WORKERS` | BLOCKED | Optional; default 1 is right for the current instance size. |
| `API_CORS_ORIGINS` | N/A | See CORS above. |
| Hosting | FAIL (likely) | The 42.8 s first response on 2026-10-06 looks like a free-plan wake-up, so `hsf-api` is probably still on Render's free plan (sleeps after 15 min, 512 MB). A US market custom scan needs Starter or more (P1-73). Not changed: no authorization to change hosting. |
| `scripts/api_acceptance.py` against production | BLOCKED | Network access now works; still needs the five `ACC_*` test accounts as environment variables (P1-63). Not run. |

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

## 5. Deployment and configuration requirements

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

## 6. Remaining gaps (not in this run)

- Screens still only in Streamlit: watchlists, alerts, Day Trader, Market Brief, scan
  history and track record, earnings, AI notes and chat, journal and paper trading,
  settings and billing, sign-up, password reset, email verification. Their APIs exist.
- Watchlist add and price alert from the Stock and Scanner pages (APIs exist; buttons
  not built yet).
- CSV export (client-side from rows).
- After Stripe checkout, customers return to the Streamlit app (`APP_SUCCESS_URL`).

## 7. Recommended next run

**"Web v2 beta deploy + production acceptance"**, once P1-73 and P1-63 are done:
deploy `hsf-web-beta` from `dev`, run `scripts/api_acceptance.py` and
`npm run screenshots` against production with the test accounts, and fix what they
find. In parallel or right after: **"Web v2 screens, part 2"**: watchlists and alerts
(including add-to-watchlist and price alert from the Stock and Scanner pages), then
Market Brief and Day Trader. These cover most of what customers use daily and have
APIs already.
