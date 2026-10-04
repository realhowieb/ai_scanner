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
| POST | `/v1/auth/login` | none | `{"email","password","client"?}` → `access_token` (15 min), `refresh_token` (30 days). Same accounts and passwords as the web app. 10 failed attempts in 10 minutes → 429. |
| POST | `/v1/auth/refresh` | none | `{"refresh_token"}` → a new pair. Each refresh token works once. A retry within 30 s of rotating (lost response, two refreshes at once) gets another pair; a later replay, or one after logout, is treated as theft and revokes every session of that account. |
| POST | `/v1/auth/logout` | none | `{"refresh_token"}` → 204, token revoked. |
| GET | `/v1/me` | Bearer | Email, name, plan (`basic`/`pro`/`premium`/`admin`), plan label, alert limit, entitlement flags. |
| GET | `/v1/today` | Bearer | Market phase, Before the open, Top setups, After the close, last session recap. Pre/after-hours movers are Pro+ (`locked: true` below Pro); Premium model fields are redacted below Premium. Each section fails on its own (`errors` lists it). |

Every route declares a response model, so `/openapi.json` describes each payload
and clients can be generated from it. A database outage answers **503** with
`Retry-After: 30`.

Send the access token as `Authorization: Bearer <token>`. Plan and admin status
are read from the database on every request, so a plan change applies at once.

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

## Tests

`tests/test_api_v1.py` (sign-in, tokens, refresh rotation and reuse, /me) and
`tests/test_api_today_and_store.py` (Today builder on saved-run fixtures; refresh
storage on a real Postgres when `HSF_TEST_PG_URL` is set). Both run in CI's
billing-contract job.
