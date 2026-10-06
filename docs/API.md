# HSF API (P1-59)

One backend for the new web frontend and the iOS/Android app. It runs as its
own service beside the Streamlit app and the billing service, reads the same
Neon database, and reuses the app's Python modules. The Streamlit app is not
changed by it.

Interactive docs (OpenAPI) are served at `/docs` once deployed.

## Endpoints (v1)

| Method | Path | Auth | What |
|---|---|---|---|
| GET | `/healthz` | none | Liveness only (no database call). Use as the Render health check. |
| GET | `/readyz` | none | Readiness: 200 when the database answers, with the latest market scan's time and age (`scan_age_minutes`) for freshness monitoring; 503 when the database is down. Don't use it as the Render health check (a database blip would restart a healthy service). |
| POST | `/v1/auth/login` | none | `{"email","password","client"?}` → `access_token` (15 min), `refresh_token` (30 days). Same accounts and passwords as the web app. 10 failed attempts in 10 minutes → 429. |
| POST | `/v1/auth/refresh` | none | `{"refresh_token"}` → a new pair. Each refresh token works once. A retry within 30 s of rotating (lost response, two refreshes at once) gets another pair; a later replay, or one after logout, is treated as theft and revokes every session of that account. |
| POST | `/v1/auth/logout` | none | `{"refresh_token","push_token"?}` → 204, token revoked; with `push_token`, that device stops getting this account's pushes. |
| POST | `/v1/auth/signup` | none | `{"email","password","username","accept_terms","client"?}` → 201 with a token pair (signed in) and `verification_sent`. Same rules as the web form (password rule, unique email and username, usage agreement); Free plan. 5 per hour per address. |
| POST | `/v1/auth/verify-email` | none | `{"token"}` from the verification link → verified. |
| POST | `/v1/auth/password-reset` | none | `{"email"}` → 202, same answer whether or not the account exists; emails a reset link (3 per hour per account, 5 per hour per address). |
| POST | `/v1/auth/password-reset/confirm` | none | `{"token","new_password"}` → new password; signs the account out of the web app and every app session. |
| GET | `/v1/me` | Bearer | Email, name, plan (`basic`/`pro`/`premium`/`admin`), plan label, alert limit, `email_verified`, entitlement flags. |
| POST | `/v1/me/verify-email` | Bearer | Resend the verification link to your own email (3 per hour). |
| POST | `/v1/me/password` | Bearer | `{"current_password","new_password"}` → a new token pair for this device; every other session (web and app) is signed out. |
| GET / PATCH | `/v1/me/email-preferences` | Bearer | `{"digest","evening","alerts"}` on/off. |
| POST | `/v1/me/devices` | Bearer | Register this phone for push: `{"push_token","platform":"ios"\|"android","provider"?:"apns"\|"fcm"\|"expo","device_name"?,"app_version"?}` → the device (`id`, never the token). Idempotent; a token already registered to another account moves to this one. Provider defaults: Expo tokens → `expo`, iOS → `apns`, Android → `fcm` (400 when the token doesn't fit). Up to 10 devices per account (least recently seen dropped). |
| GET | `/v1/me/devices` | Bearer | Your registered devices, most recently seen first. |
| DELETE | `/v1/me/devices/{id}` | Bearer | Remove a device (204; 404 when not yours). |
| POST | `/v1/billing/checkout` | Bearer | `{"plan":"pro"\|"premium","interval":"month"\|"year"}` → `{url, mode}`: Stripe checkout, or Stripe's plan-change screen for existing subscribers (`mode: portal`). Needs a verified email (403 otherwise), as on the web. |
| POST | `/v1/billing/portal` | Bearer | `{"flow"?: "cancel"}` → Stripe Customer Portal URL (404 when there's no subscription yet). |
| GET | `/v1/today` | Bearer | Market phase, Before the open (this morning's pre-market movers, 8:35-10:30 ET), Top setups, After the close, last session recap. Pre/after-hours movers are Pro+ (`locked: true` below Pro); Premium model fields are redacted below Premium. Each section fails on its own (`errors` lists it). |
| GET | `/v1/scans/latest` | Bearer | The latest market scan's HSF setups, ranked as in the Scanner. `limit` (≤200), `offset`, `min_score`, `signal`. A plan sees its Scanner row cap (Free 25, Pro 100, Premium 200; `limited: true` when the cap hides some). PreBreakout (`prob`, `signal=prebreakout`) is Premium. |
| POST | `/v1/scans` | Bearer | Start a **custom scan** (the web's Custom Scan page) → **202** with a job: `{"universe": "sp500"\|"nasdaq"\|"combo"\|"us_market"\|"watchlist"\|"ticker", "ticker"?, "watchlist_id"?, "score_all"?, "filters"?: {min_price, max_price, min_dollar_vol, min_gap, apply_gap_filter, unusual_volume, session, profile, top_n, max_nasdaq, max_combo}}`. Plan rules are enforced here (403): NASDAQ/Combo Pro+, US market Premium+, `top_n` ≤ plan rows, pre-market/after-hours/unusual volume/gap filter Pro+, Pro ticker caps ≤ 4,000 / 6,000, Premium scans the full lists. One scan per account at a time (409 with the running `scan_id`), 30 per hour (429), 503 + `Retry-After` when the service is busy. |
| GET | `/v1/scans/{scan_id}` | Bearer | Poll every 2–5 s: `status` `queued` → `running` (with `progress.phase` and `progress.symbols`) → `complete` (`result.setups`, same rows as `/v1/scans/latest`) or `failed` (`error`, safe to show). 404 when not yours. |
| GET | `/v1/scans` | Bearer | Your recent custom scans (newest first, no rows; kept 7 days). |
| GET | `/v1/runs`, `/v1/runs/{id}` | Bearer, **Pro** | Your saved scans (the web's Scan History) and one with its rows. 404 when not yours. |
| GET | `/v1/track-record`, `/v1/track-record/daily` | Bearer, **Pro** | Historical research: saved scan picks vs SPY by ranking and horizon (descriptive, with disclaimer); daily excess return. |
| GET | `/v1/earnings?days=7&tickers=…` | Bearer, **Pro** | Upcoming earnings (0–30 days), soonest first, optionally only for given tickers. |
| GET | `/v1/brief` | Bearer | The Market Brief (same builder as the morning email): backdrop, breadth, sectors, top opportunities with movement, gappers, movers, setups, earnings today. PreBreakout picks Premium. Cached 5 min. |
| GET | `/v1/day-trader?source=…` | Bearer, **Pro** | Live Day Trader table for `watchlist` (default list or `watchlist_id`), `movers`, `movers_sp500`, `movers_nasdaq`, `premarket`, `postmarket`, `scan_picks`, `megacaps` or `custom` (`symbols=`). Quotes shared 30 s; poll every 30–60 s. |
| GET | `/v1/day-trader/stair-steppers?symbols=…` | Bearer, **Pro** | Smooth 1-minute trends (R², trend/hour, pullback filters). |
| POST | `/v1/ai/summary`, `/v1/ai/chat`, `/v1/ai/notes/{ticker}`; GET `/v1/ai/brief-narrative` | Bearer, **Premium** | Claude explains a scan (latest market scan or one of your `run_id`s), answers questions about it (client sends the conversation; 8 turns kept), writes a setup note for a ticker, or the brief's narrative. Same guardrails and daily limit as the web; 429 at the limit, 503 when AI is off. Shared answers are cached 30 min per scan. |
| GET / POST | `/v1/journal` | Bearer (POST **Pro**) | Your trades marked to live quotes with P&L and stats; log a trade `{ticker, entry_price, shares}`. |
| POST / DELETE | `/v1/journal/{id}/close`, `/v1/journal/{id}` | Bearer, **Pro** | Close `{exit_price}` (409 if already closed) or delete a trade. |
| GET | `/v1/stocks/{ticker}/plan?account_size&risk_pct` | Bearer, **Pro** | Trade plan for a latest-scan result: entry, stop, 1.5R/3R targets, size. |
| GET / POST / DELETE | `/v1/paper/account` | Bearer, **Premium** | Paper account status; connect your own Alpaca **paper** keys (validated, stored encrypted, never returned); disconnect. |
| GET | `/v1/paper/activity` | Bearer, **Premium** | Live positions and the order feed. |
| POST | `/v1/paper/orders` | Bearer, **Premium** | `{ticker, qty, "confirm": true}` → whole-share market buy in your paper account, imported into the journal. Refused unless the trading endpoint is Alpaca's paper host. |
| DELETE | `/v1/me` | Bearer | `{password, "confirm": "DELETE"}` → deletes your account and its data. 409 while a paid subscription is active (cancel via the portal first) and for admins. |
| GET | `/v1/stocks/{ticker}` | Bearer | Stock Intelligence (same builder as the web page): score, components, signals, reasons, risks, what to watch, lifecycle, historical context, daily bars the scans cached (up to 120), and your watchlists and alerts on it. |
| GET / POST | `/v1/watchlists` | Bearer | List; create `{"name","make_default"?}` (201; 409 duplicate name; 403 past 50 lists). |
| GET / PATCH / DELETE | `/v1/watchlists/{id}` | Bearer | Items with notes; rename / make default `{"name"?,"make_default"?}`; delete (204). 404 when not yours. |
| POST | `/v1/watchlists/{id}/tickers` | Bearer | `{"tickers":[...]}` (≤200) → `added`, `already_present`, `invalid`. |
| PATCH / DELETE | `/v1/watchlists/{id}/tickers/{ticker}` | Bearer | Set the note `{"note"}`; remove the ticker (204). |
| GET / POST | `/v1/alerts` | Bearer | Your alerts with `limit`, `used`, `email_enabled`; create `{"type","ticker"?,"threshold"?,"direction"?,"watchlist_only"?}`. Same types and input rules as the web app; Free 1 alert, Pro 5, Premium 25 (403 at the limit, 409 duplicate, 422 bad input). |
| PATCH / DELETE | `/v1/alerts/{id}` | Bearer | `{"enabled"}`; delete (204). |
| GET | `/v1/alerts/events` | Bearer | Your recent fired alerts (`limit` ≤100), newest first. |

Every route declares a response model, so `/openapi.json` describes each payload
and clients can be generated from it. Web v2's client is generated from the committed
copy `web/openapi.json`; after an API change run `python scripts/export_openapi.py`
and `cd web && npm run api:types` (CI fails while they're stale). A database outage answers **503** with
`Retry-After: 30`.

Account flows call the same functions as the web app, so password rules, emails,
tokens and limits match. Emailed links (verification, password reset) open the web
app's pages (`APP_BASE_URL`), which already handle them; a new frontend can instead
post the same tokens to the confirm endpoints. After Stripe, customers return to the
web app (`APP_SUCCESS_URL` on the billing service). Sign-in, sign-up, password reset
and verification are also limited per client address (in-process; Render's
proxy-added `X-Forwarded-For` entry is the one counted).

Send the access token as `Authorization: Bearer <token>`. Plan and admin status
are read from the database on every request, so a plan change applies at once.

Logout revokes the refresh token only. Access tokens are stateless: one already
issued keeps working until it expires (at most 15 minutes). Deactivating an
account or changing its plan applies on the next request.

Push devices (P1-64): the app should call `POST /v1/me/devices` at every start
and whenever it gets new tokens (sign-in, sign-up, password change) or the OS gives
it a new push token. Signing out with `push_token` removes that device; a password
change or reset, or a refresh-token reuse sweep, removes every device of the
account, so the app registers again after its next sign-in. Nothing sends pushes
yet; the alert sender will read `api.devices.devices_for_user()`.

Custom scans run in the background because they take from a couple of seconds
(one ticker, S&P 500) to minutes (US market: 224.6 s on 2026-10-06). They use
the web's own code: `scan.universe_selection.resolve_scan_universe` (same lists
and liquidity pre-filter) and `scan.execution.run_manual_scan_execution` with
`scan.engine.run_breakout_scan`; results are shaped like `/v1/scans/latest`
and saved to the account's scan history like a web scan. Jobs live in the
API-owned `api_scan_jobs` table and run on `API_SCAN_WORKERS` threads (default
1, which also bounds memory). A job interrupted by a restart or deploy reads
`failed` / `interrupted` (after 20 minutes without progress while running, 60 while
queued); start it again.

Example (shape only):

```
POST /v1/scans  {"universe": "nasdaq", "filters": {"top_n": 50, "session": "regular"}}
202 {"scan_id": "9f37c3e6…", "status": "queued", "universe": "nasdaq",
     "params": {"universe": "nasdaq", "top_n": 50, "session": "regular", "full_lists": false,
                "max_nasdaq": 1200, "max_results": 100, …}, "progress": {"phase": "queued"}}
GET /v1/scans/9f37c3e6…
200 {"status": "complete", "progress": {"phase": "complete", "elapsed_s": <seconds>},
     "result": {"label": "NASDAQ", "symbols_scanned": <n>, "duration_s": <seconds>, "total": <≤ 50>,
                "setups": [{"ticker": "…", "score": <0-100>, "signals": ["breakout"], …}]}}
```

Errors are `{"detail": "<message>"}` (validation errors: `{"detail": [ … ]}`, FastAPI's
standard list). Times are ISO 8601 and always carry a timezone. Plans are
`basic` / `pro` / `premium` / `admin` in data; `plan_label` (Free / Pro / Premium /
Admin) is the name to show. Tickers are upper case.

## Postman

Import `https://hsf-api.onrender.com/openapi.json` (File → Import → Link; OpenAPI
3.1). Set a collection variable `base_url`, and after login paste the access token
into the collection's Bearer auth. Never save real passwords or tokens in a shared
collection. Smoke sequence:

1. `GET /healthz`, `GET /readyz`
2. `POST /v1/auth/login` (test account) → copy `access_token`, `refresh_token`
3. `GET /v1/me`
4. `GET /v1/today`
5. `GET /v1/scans/latest?limit=10`
6. `GET /v1/stocks/AAPL`
7. `POST /v1/scans` `{"universe": "ticker", "ticker": "AAPL"}` → poll `GET /v1/scans/{scan_id}` until `complete`
8. `POST /v1/watchlists` `{"name": "postman-test"}` → `POST …/tickers` → `DELETE /v1/watchlists/{id}`
9. `POST /v1/alerts` `{"type": "move", "ticker": "AAPL", "threshold": 999999}` → `DELETE /v1/alerts/{id}`
10. `POST /v1/auth/refresh` `{"refresh_token": …}`
11. `POST /v1/auth/logout` `{"refresh_token": <the new one>}`

Every response carries `X-Request-ID` (a client's own id is kept when it is
8-64 characters of letters, digits, `.`, `_` or `-`). Each request writes one
JSON log line: request id, method, route template, status, milliseconds. Logs
never include headers, bodies, query strings or tokens.

Watchlists and alerts use the same tables and functions as the web app, so a
change in one shows in the other. The web app caches watchlists for up to 2
minutes, so a change made in the app can take that long to appear on an open
web page.

## Security

- Passwords: bcrypt only; a missing account costs the same time as a wrong password.
- Access tokens: HS256 JWT signed with `API_JWT_SECRET`, `iss=hsf-api`, 15-minute expiry.
- Refresh tokens: random 48-byte values; only their SHA-256 is stored (`api_refresh_tokens`).
- Sign-in attempts go to the existing `login_attempts` table (shared rate limit with the web app).
- CORS: only the exact origins in `API_CORS_ORIGINS` (no wildcards). Web v2 (`web/`) doesn't need it: its server (BFF) calls the API and keeps tokens in HttpOnly cookies (`web/README.md`). Preflights are cached 10 minutes; `X-Request-ID` and `Retry-After` are readable by the browser. Native apps don't need CORS.

## Deploy on Render

New → Web Service → this repository, branch `main`.

| Setting | Value |
|---|---|
| Root directory | (repository root) |
| Build command | `pip install -r api/requirements.txt` |
| Start command | `python -m uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/healthz` |
| Python version | 3.13 (from `.python-version`) |

Environment variables:

| Name | Value |
|---|---|
| `DATABASE_URL` | Same Neon URL as the billing service |
| `API_JWT_SECRET` | New random secret, 32+ characters (e.g. `openssl rand -base64 48`). Only on this service. |
| `API_CORS_ORIGINS` | Comma-separated web origins allowed to call it from a browser, exactly `scheme://host[:port]`, e.g. `https://app.hsfinest.ai,https://hsf-web.onrender.com`. `https` only (`http://localhost:<port>` allowed for local development). Wildcards, paths and plain-http hosts are ignored with a warning in the log. Empty = no browser access. Native iOS/Android apps don't need it. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM` (`SMTP_FROM_NAME` optional) | Same Resend values as the web app. Needed for sign-up verification and password-reset emails; without them sign-up still works (`verification_sent: false`) and reset requests send nothing. |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` (`ALPACA_DATA_FEED` optional) | Same values as the web app. **Needed for custom scans** (`POST /v1/scans`): the US market list and the liquidity pre-filter come from Alpaca, and price downloads fall back to slow one-by-one Yahoo calls without it. |
| `ANTHROPIC_API_KEY` (`AI_ENABLED`, `AI_DAILY_LIMIT` optional) | Same as the web app. Needed for `/v1/ai/*` (Premium); without it those answer 503. |
| `APP_ENCRYPTION_KEY` | Same as the web app. Needed to store paper-trading keys (`POST /v1/paper/account`); without it paper trading answers 503. |
| `API_SCAN_WORKERS` | Optional, default 1 (max 4): custom scans that run at once. Raise only with more memory (a US market scan holds thousands of price histories). |
| `APP_BASE_URL`, `BILLING_API_BASE` | Optional; default to the production web app and billing service. |

The service won't start without `API_JWT_SECRET` (uvicorn logs
"HSF API not started: API_JWT_SECRET must be set…" and exits). Changing it signs every app
user out once (access tokens stop verifying; refresh tokens still work).

The free plan sleeps after 15 minutes idle; move to Starter before the app has
real users.

## Acceptance journey

`scripts/api_acceptance.py` runs the full user journey (sign in, scanner, stock
page, watchlist, note, alert, disable, delete, logout) for each plan, plus
two-account isolation, plan alert limits, refresh rotation and CORS, and
reports PASS / FAIL / BLOCKED with evidence and per-route timings. It reads
test-account credentials from `ACC_<ROLE>_EMAIL` / `ACC_<ROLE>_PASSWORD`
(ROLE = FREE, PRO, PRO2, PREMIUM, ADMIN), never prints them, writes only with
`--allow-writes`, names everything it creates `zz-acceptance-<run id>` and
deletes it at the end. Alerts use a threshold that cannot fire, so no email is
sent. Use dedicated test accounts only.

    API_BASE_URL=https://hsf-api.onrender.com python scripts/api_acceptance.py \
        --allow-writes --origin https://<frontend origin> --report acceptance.json

## Tests

`tests/test_api_v1.py` (sign-in, tokens, refresh rotation and reuse, /me),
`tests/test_api_v1_data.py` (scans, stock detail, watchlists, alerts; plan caps,
ownership, readiness, request ids), `tests/test_alerts_atomic_create.py` and
`tests/test_api_db_sessions.py` (concurrency limits, no idle-in-transaction
connections, no leaked connections; the Postgres parts need `HSF_TEST_PG_URL`)
and `tests/test_api_today_and_store.py` (Today builder on saved-run fixtures; refresh
storage on a real Postgres when `HSF_TEST_PG_URL` is set). All run in CI's
billing-contract job; the scan and stock tests need pandas, so they run in the
full suite.
