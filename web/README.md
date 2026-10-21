# HSF Web v2 (P1-58)

The new web frontend: Next.js (App Router) + TypeScript, built entirely on the HSF
API (`hsf-api`). The Next.js app is the current UI at `https://app.hsfinest.com`;
the Streamlit app was retired on 2026-10-21.

The web app includes public landing, interactive demo, methodology and pricing;
signup, sign-in, email verification and password reset; Today, Market Brief,
Scanner (including saved views, custom scans and history), Stock Intelligence and
comparison, Watchlists, Alerts, Track record, Day Trader, Journal, Paper trading,
Account and billing, and Admin. Availability follows the API's plan entitlements.


Research presentation:

- Scanner keeps setup chips and freshness visible; saved views, filters and table
  preferences share a compact toolbar. Display preferences still apply to mobile cards.
- Stock history distinguishes grouped ticker/day signals, individual stored scan
  observations, score-range price outcomes, and HSF-state persistence. Persistence
  horizons such as H24 mean the first comparable observation at least 24 elapsed
  hours later, including weekends; they are not trading-day return windows.
- Alerts start with one stock, a watchlist, or market-wide monitoring. Price and
  technical alerts use `/v1/alerts`; HSF conditions use `/v1/alerts/rules` and require
  a ticker or watchlist for stock conditions. Market-wide HSF Score, setup and earnings
  filters use scan alerts saved from the Scanner. Available conditions and validation
  come from the server catalogs.
- Day Trader describes closed-session rankings separately from live quotes.
  Its API response has a request timestamp but no session date or individual trade
  timestamps; the frontend discloses this rather than inventing dates.

The repository root is the active project. A separate older checkout may exist at
`ai_scanner/`; do not run or edit that copy by accident, or remove it without
confirming its purpose with the owner.

## How it talks to the API

```
browser ──(same origin, HttpOnly cookies)──> Next.js server (BFF) ──(Bearer)──> hsf-api
          /api/auth/login, /api/auth/logout      src/server/*
          /api/hsf/v1/...  (proxy)
```

- **No token ever reaches browser JavaScript.** Sign-in posts to `/api/auth/login`;
  the server stores the access and refresh tokens in `__Host-` cookies that are
  `Secure`, `HttpOnly`, `SameSite=Lax`. Nothing goes to localStorage or sessionStorage.
- **Refresh** happens on the server (`src/server/bff.ts`): before forwarding when the
  access token is missing or about to expire, and once more after a 401. Refresh is
  **single-flight** (`src/server/refresh.ts`): concurrent requests with the same
  refresh token share one rotation, and requests arriving within 10 s reuse its
  result. Across several server instances, the API's 30-second rotation grace covers
  the same race. A rejected refresh token clears the cookies and answers
  `401 {"code": "session_expired"}`; the client then sends the user to sign in again
  (`?next=` brings them back). Temporary service errors, rate limits and timeouts
  preserve the cookies and return a retryable error with `Retry-After`.
- **CSRF**: mutating calls must carry this site's `Origin` (or `Sec-Fetch-Site:
  same-origin`); cookies are `SameSite=Lax`.
- **Only `/v1/...` data routes** are proxied (`/v1/auth/*` is refused), bodies are
  capped at 256 KB, and upstream calls time out after 75 s (an idle free-plan API can take ~60 s to wake). Sign-in retries once on a gateway error (502/503/504) and otherwise says the service is starting up.
- **Request IDs**: the client sends `X-Request-ID: web-<uuid>` on every call; the BFF
  forwards it and the API logs it. Error screens show it as a "Support code".
- **`API_CORS_ORIGINS` isn't needed**: the browser never calls the API directly. Leave it
  empty unless another browser client needs it.
- Plans and features come only from `GET /v1/me` (`entitlements`) and from the API's
  own answers (`max_results`, `locked`, `historical_locked`, 403s). The frontend
  doesn't decide what a plan includes; it only hides or labels controls. The server
  enforces every rule.

The API client is generated from the API's OpenAPI contract: `openapi.json` (exported
by `python scripts/export_openapi.py` at the repo root) → `src/api/schema.d.ts`
(`npm run api:types`), used through `openapi-fetch`. CI fails if either is stale.

## Run locally

```
cd web
npm ci
cp .env.example .env.local      # point HSF_API_BASE_URL at an API (http allowed for localhost only)
npm run dev                      # http://localhost:3000
```

`npm run dev` uses plain (non-`Secure`) session cookies so sign-in works over http in
every browser, Safari included; production builds (`npm run build && npm start`) always
use `Secure` `__Host-` cookies and need https (or Chrome/Firefox on `localhost`). Dev
mode accepts `localhost` and `127.0.0.1`; to open it from a phone on the same Wi-Fi,
start it with `HSF_DEV_ORIGINS=<your computer's LAN IP>`. The first sign-in after the
API has been idle can take ~45 s while Render wakes it up.

Checks (all run in CI, `.github/workflows/web.yml`):

```
npm run typecheck && npm run lint && npm test && npm run build
```

Screenshots at desktop and phone sizes against a running server, with a test account
(credentials from the environment, never printed):

```
BASE_URL=http://localhost:3000 HSF_TEST_EMAIL=<test account> HSF_TEST_PASSWORD_FILE=<file> \
  STOCK=AAPL OUT_DIR=screenshots npm run screenshots
```

The whole daily journey (sign in → Scanner → stock → save to a new watchlist → price
alert → sign out and back in → isolation → cleanup) as a browser check, against any
deployment, with dedicated test accounts. It names everything `zz-e2e-<run>`, uses a
price alert that can't fire, and deletes only what it created:

```
BASE_URL=https://<web v2> HSF_TEST_EMAIL=<test account> HSF_TEST_PASSWORD_FILE=<file> \
  [HSF_TEST_EMAIL_2=<second test account>] OUT_DIR=screenshots node scripts/journey.mjs
```

An accessibility check (axe-core, WCAG 2.1 A and AA) signs in and scans Today, Scanner,
a stock page, Watchlists, Alerts, Day Trader and Account at desktop and phone sizes. It
fails on any critical or serious problem and lists the rest; Live acceptance runs it daily:

```
BASE_URL=https://<web v2> HSF_TEST_EMAIL=<test account> HSF_TEST_PASSWORD_FILE=<file> npm run a11y
```

After a deploy, `BASE_URL=https://<web v2> node scripts/beta-smoke.mjs` checks health,
security headers, redirects and the BFF without an account (add `HSF_TEST_EMAIL` and
`HSF_TEST_PASSWORD_FILE` for cookie flags and signed-in timings).

## Render deployment configuration

Deployment settings below describe configuration, not proof of the current live
release. The beta guide (order of operations, invite list, verification, rollback) is
[docs/WEB_V2_BETA_DEPLOY.md](../docs/WEB_V2_BETA_DEPLOY.md). Summary:

New → Web Service → this repository.

| Setting | Value |
|---|---|
| Branch | `main` for the live service (`hsf-web-v1`); `dev` for the preview (`hsf-web-beta`) |
| Root directory | `web` |
| Runtime | Node 22 (`web/.node-version`) |
| Build command | `npm ci && npm run build` |
| Start command | `npm start` (Next.js reads `$PORT`) |
| Health check path | `/api/healthz` (liveness; doesn't call the API) |

Environment variables:

| Name | Value |
|---|---|
| `HSF_API_BASE_URL` | `https://hsf-api-v1.onrender.com` (server-only; https required) |
| `WEB_BETA_ALLOWED_EMAILS` | Optional, comma-separated: only these accounts can sign in (invite-only beta). Unset = every HSF account. |
| `NODE_VERSION` | `22` (optional; `.node-version` already says 22) |

Optional analytics uses a public project key; private service credentials stay on
the API. Session cookies are
set per user. It must be served over https (Render does this), because the
`__Host-` cookies require it. The free plan sleeps after 15 idle minutes; use Starter
before real users depend on it. Confirm the domain target, deployed commit, and API configuration before each
release; local verification does not prove which version production is serving.

## Interactive stock charts

Stock pages enhance the existing daily-price chart with TradingView Lightweight
Charts (loaded in the browser). Candlesticks, volume, EMA 9/21, SMA 20/50, HSF
signal markers, cross markers, range selection and keyboard session readouts use
only the existing API bars and lifecycle events. Drag pans, pinch/wheel zooms, and
vertical touch gestures can still scroll the page. Upcoming earnings are a date
marker in empty chart space, not a projected candle or price. These remain saved
daily bars, not streaming quotes. The SVG chart stays available if the interactive
renderer cannot load. No new price feed or market-data subscription is needed.

TradingView attribution stays visible. The shipped notice and Apache license are
available at `/licenses/lightweight-charts.txt`.

## Optional PostHog usage analytics

Analytics is **disabled when `NEXT_PUBLIC_POSTHOG_KEY` is blank**. To enable it,
set the public **project key** (not a personal API key) and ingestion region before
building the frontend:

```dotenv
NEXT_PUBLIC_POSTHOG_KEY=<your public project key>
NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com
```

For an EU project use `https://eu.i.posthog.com`. These are build-time browser
settings; rebuild after changing them. Only the selected cloud ingestion origin
is added to `connect-src`. There is no analytics proxy or API credential change.

We send these explicit events only, after successful API actions where relevant:

| Event | Properties | Meaning |
| --- | --- | --- |
| `page_viewed` | `route` | Visit to an allowed page category |
| `scanner_viewed` | `route` | Scanner visit |
| `stock_viewed` | `route=/stocks/:ticker` | Stock research visit, without the symbol |
| `watchlist_created` | none | Watchlist creation succeeded |
| `watchlist_stock_added` | none | Adding tickers to a watchlist succeeded |
| `alert_created` | `kind` | A price, technical, HSF or scan alert was created |
| `signup_completed` | none | Signup returned a successful session |
| `watchlist_deleted`, `watchlist_stock_removed` | none | Removal succeeded |
| `alert_enabled_changed` | `enabled` | Alert or rule enable/disable succeeded |
| `alert_deleted` | none | Alert or rule removal succeeded |
| `custom_scan_started` | `universe`, `session` | API accepted a scan request |
| `journal_trade_logged`, `journal_trade_closed` | none | Journal action succeeded |
| `checkout_started` | `plan`, `interval` | Checkout URL returned successfully |

Suggested PostHog funnel: `scanner_viewed` → `stock_viewed` →
`watchlist_stock_added` → `alert_created`. For retention, measure browsers returning
to `scanner_viewed` or `stock_viewed` in later weeks. Identity is an anonymous
browser identifier, reset at sign-out/session expiry; there is no account
identification or cross-device merging. Clearing browser storage also resets it.

Autocapture, session replay, heatmaps, exception capture, surveys, feature flags,
and external SDK extensions are disabled. An outbound allowlist removes URLs,
referrers, query strings, ticker symbols, account fields, person properties, and
free text. Authentication tokens stay in the existing HttpOnly cookies. Failed
analytics loading or requests cannot block product actions.

When configured, Account → Usage privacy lets users disable analytics in this
browser. Do Not Track and Global Privacy Control also disable collection. Test
with a dedicated development project and synthetic data before enabling in
production; browser/ad blocking may reduce observed event counts.


The Next.js browser instrumentation hook and action events share one lazy,
privacy-gated initializer. Missing keys are supported in development and production.
Tests use a blank key globally; analytics tests override it with a dummy key.

### Optional operational logs

`POSTHOG_LOGS_ENABLED=1` explicitly enables server-side API proxy lifecycle logs
using the configured public project key and ingestion region. The default is off.
This setting is independent of the browser usage preference: it sends only HTTP
method/status and fixed lifecycle messages with service name `hsf-web`, without
URLs, bodies, account details or credentials. There is no global logger exporter.
Logging/export failures do not block API responses. Browser error tracking remains
disabled; the existing error reporting behavior is preserved.

The setup wizard created the [usage dashboard](https://us.posthog.com/project/679220/dashboard/2241150)
and a [setup notebook](https://us.posthog.com/project/679220/notebooks/9c8820e560fe).
The notebook records the wizard's generated snapshot; the reviewed implementation
preserves anonymous tracking and makes operational logs optional as described here.
Production environment configuration and live delivery still need verification
after an authorized deployment. AI observability in the upstream API and source-map
uploads are outside this frontend integration.


For per-service PostHog keys, browser push, Sentry and AI monitoring setup, see
[the integration guide](../docs/INTEGRATIONS_2026-10-21.md).
