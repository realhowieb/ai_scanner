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
| POST | `/v1/auth/logout` | none | `{"refresh_token"}` → 204, token revoked. |
| GET | `/v1/me` | Bearer | Email, name, plan (`basic`/`pro`/`premium`/`admin`), plan label, alert limit, entitlement flags. |
| GET | `/v1/today` | Bearer | Market phase, Before the open (this morning's pre-market movers, 8:35-10:30 ET), Top setups, After the close, last session recap. Pre/after-hours movers are Pro+ (`locked: true` below Pro); Premium model fields are redacted below Premium. Each section fails on its own (`errors` lists it). |
| GET | `/v1/scans/latest` | Bearer | The latest market scan's HSF setups, ranked as in the Scanner. `limit` (≤200), `offset`, `min_score`, `signal`. A plan sees its Scanner row cap (Free 25, Pro 100, Premium 200; `limited: true` when the cap hides some). PreBreakout (`prob`, `signal=prebreakout`) is Premium. |
| GET | `/v1/stocks/{ticker}` | Bearer | Stock Intelligence (same builder as the web page): score, components, signals, reasons, risks, what to watch, lifecycle, historical context, daily bars the scans cached (up to 120), and your watchlists and alerts on it. |
| GET / POST | `/v1/watchlists` | Bearer | List; create `{"name","make_default"?}` (201; 409 duplicate name; 403 past 50 lists). |
| GET / PATCH / DELETE | `/v1/watchlists/{id}` | Bearer | Items with notes; rename / make default `{"name"?,"make_default"?}`; delete (204). 404 when not yours. |
| POST | `/v1/watchlists/{id}/tickers` | Bearer | `{"tickers":[...]}` (≤200) → `added`, `already_present`, `invalid`. |
| PATCH / DELETE | `/v1/watchlists/{id}/tickers/{ticker}` | Bearer | Set the note `{"note"}`; remove the ticker (204). |
| GET / POST | `/v1/alerts` | Bearer | Your alerts with `limit`, `used`, `email_enabled`; create `{"type","ticker"?,"threshold"?,"direction"?,"watchlist_only"?}`. Same types and input rules as the web app; Free 1 alert, Pro 5, Premium 25 (403 at the limit, 409 duplicate, 422 bad input). |
| PATCH / DELETE | `/v1/alerts/{id}` | Bearer | `{"enabled"}`; delete (204). |
| GET | `/v1/alerts/events` | Bearer | Your recent fired alerts (`limit` ≤100), newest first. |

Every route declares a response model, so `/openapi.json` describes each payload
and clients can be generated from it. A database outage answers **503** with
`Retry-After: 30`.

Send the access token as `Authorization: Bearer <token>`. Plan and admin status
are read from the database on every request, so a plan change applies at once.

Logout revokes the refresh token only. Access tokens are stateless: one already
issued keeps working until it expires (at most 15 minutes). Deactivating an
account or changing its plan applies on the next request.

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
- CORS: only the origins in `API_CORS_ORIGINS`. Native apps don't need CORS.

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
| `API_CORS_ORIGINS` | Comma-separated web origins allowed to call it, e.g. the new web app's URL. Empty = no browser access. |

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
