# HSF Web v2 (P1-58)

The new web frontend: Next.js (App Router) + TypeScript, built entirely on the HSF
API (`hsf-api`). It runs **beside** the Streamlit app; Streamlit, its landing page and
the production domain are unchanged.

Screens: **Sign in**, **Today**, **Scanner** (latest scan) with **Custom scan**,
**Stock Intelligence**, **Watchlists** and **Alerts**, plus "Save to watchlist" (Scanner
rows, stock page) and "Set price alert" (stock page). Everything else (Day Trader, Market
Brief, journal, billing pages, sign-up, password reset) still lives in the Streamlit app,
linked from the account menu ("Open classic app") and the sign-in page.

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
  the same race. A failed refresh clears the cookies and answers
  `401 {"code": "session_expired"}`; the client then sends the user to sign in again
  (`?next=` brings them back).
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

After a deploy, `BASE_URL=https://<web v2> node scripts/beta-smoke.mjs` checks health,
security headers, redirects and the BFF without an account (add `HSF_TEST_EMAIL` and
`HSF_TEST_PASSWORD_FILE` for cookie flags and signed-in timings).

## Deploy on Render (not done yet; needs the owner)

The full beta guide (order of operations, invite list, verification, rollback) is
[docs/WEB_V2_BETA_DEPLOY.md](../docs/WEB_V2_BETA_DEPLOY.md). Summary:

New → Web Service → this repository.

| Setting | Value |
|---|---|
| Branch | `dev` for a beta service first (e.g. `hsf-web-beta`); `main` later |
| Root directory | `web` |
| Runtime | Node 22 (`web/.node-version`) |
| Build command | `npm ci && npm run build` |
| Start command | `npm start` (Next.js reads `$PORT`) |
| Health check path | `/api/healthz` (liveness; doesn't call the API) |

Environment variables:

| Name | Value |
|---|---|
| `HSF_API_BASE_URL` | `https://hsf-api.onrender.com` (server-only; https required) |
| `NEXT_PUBLIC_STREAMLIT_URL` | The Streamlit app, for "Create account", "Forgot password", "Open classic app" (default `https://hsfinestai.streamlit.app`). Read at build time. |
| `WEB_BETA_ALLOWED_EMAILS` | Optional, comma-separated: only these accounts can sign in (invite-only beta). Unset = every HSF account. |
| `NODE_VERSION` | `22` (optional; `.node-version` already says 22) |

No secrets are needed: the frontend holds no API keys, and the session cookies are
set per user. It must be served over https (Render does this), because the
`__Host-` cookies require it. The free plan sleeps after 15 idle minutes; use Starter
before real users depend on it. Production traffic and the domain stay on Streamlit
until the owner decides to switch.
