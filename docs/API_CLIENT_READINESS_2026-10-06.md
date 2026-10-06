# API v1 client readiness / frontend migration gate (2026-10-06)

Question: could a completely separate frontend reproduce the core HSF AI customer
experience using only the API, without Streamlit code or session state?

**Verdict: READY_WITH_BLOCKERS.** Every core customer workflow (account, plans,
billing, Today, latest scan, Stock Intelligence, watchlists, alerts, push devices
and, as of this run, Custom Scan) can be done through the API with server-side plan
enforcement. Streamlit can't be retired yet: several paid features in the pricing
table still exist only in Streamlit (section 6), and custom scans need the Alpaca
keys on the API service before they work in production.

- Start: `a649170` (dev = main). End: see the commit that adds this file.
- Evidence levels: **code** (read against current code), **test** (automated
  tests in this repo), **isolated** (`scripts/api_acceptance.py` as an external
  HTTP client against a local server on local Postgres 16, seeded test accounts,
  market-data downloads mocked). Nothing here ran against production.

## 1. Client-readiness matrix

Status: READY = an external React/native client can do it through the API today.

| Capability | Status | API endpoint | Auth | Entitlement (server-side) | Streamlit dependency | Gap |
|---|---|---|---|---|---|---|
| Signup | READY | `POST /v1/auth/signup` | none, 5/h per IP | Free plan | none | Verification link opens the Streamlit `verify_email` page (`APP_BASE_URL`); Web v2 can own that page and post the token to the API |
| Login | READY | `POST /v1/auth/login` | none, 30/10 min per IP + per-account lockout | none | none | — |
| Access-token refresh | READY | `POST /v1/auth/refresh` | refresh token | none | none | Rotation, 30 s retry grace, reuse = theft sweep (test) |
| Logout | READY | `POST /v1/auth/logout` | refresh token | none | none | Access tokens stay valid ≤ 15 min (documented) |
| Email verification | READY | `POST /v1/auth/verify-email` | token | none | link target only | as signup |
| Verification resend | READY | `POST /v1/me/verify-email` | Bearer, 3/h | none | none | — |
| Password reset | READY | `POST /v1/auth/password-reset`, `/confirm` | none, 5/h per IP | none | link target only | Signs out web + app sessions and removes push devices (test) |
| Password change | READY | `POST /v1/me/password` | Bearer | none | none | New pair for this device; others signed out (test) |
| Account / profile | PARTIAL | `GET /v1/me` | Bearer | — | none | Read-only (the web has no profile edit either). No account deletion anywhere; required for the App Store, not for Web v2 |
| Email preferences | READY | `GET/PATCH /v1/me/email-preferences` | Bearer | none | none | — |
| Plan / entitlement discovery | READY | `GET /v1/me` (`plan`, `plan_label`, `entitlements`, `alert_limit`); row caps in scan responses | Bearer | read from DB every request | none | — |
| Billing checkout | READY | `POST /v1/billing/checkout` | Bearer, verified email (403) | — | none | After Stripe, customers return to the web app (`APP_SUCCESS_URL` on the billing service) |
| Billing portal / cancel | READY | `POST /v1/billing/portal` (`flow=cancel`) | Bearer | — | none | as checkout |
| Today | READY | `GET /v1/today` | Bearer | movers Pro+, model fields Premium (redacted) | none | — |
| Latest market scan | READY | `GET /v1/scans/latest` | Bearer | rows capped by plan; PreBreakout Premium | none | — |
| Scanner filtering | READY | `?min_score`, `?signal`, `limit/offset`; rows carry signals, rvol, gap_pct, chg_pct | Bearer | as above | none | Lenses ("volume", "new since last visit"), card/table view and saved screens are client-side, as on the web |
| Stock Intelligence | READY | `GET /v1/stocks/{ticker}` | Bearer | Premium model output redacted | none | Bars come from the scans' cache (no live quote) |
| Custom Scan (S&P 500) | READY (new) | `POST /v1/scans` + `GET /v1/scans/{id}` | Bearer, 1 at a time, 30/h | every plan | none | Production needs Alpaca keys on hsf-api (section 5) |
| Single-ticker scan | READY (new) | `POST /v1/scans` `universe=ticker` | Bearer | every plan | none | as above |
| Watchlist scan | READY (new) | `universe=watchlist`, `score_all` | Bearer, own watchlist only (404) | every plan | none | as above |
| Full US market scan | READY (new) | `universe=us_market` | Bearer | Premium/admin (403 otherwise) | none | as above; takes minutes |
| Full NASDAQ scan | READY (new) | `universe=nasdaq` | Bearer | Pro (cap ≤ 4,000), Premium/admin full list | none | as above |
| Full Combo scan | READY (new) | `universe=combo` | Bearer | Pro (cap ≤ 6,000), Premium/admin full list | none | as above |
| Watchlists list/create/rename/default/delete | READY | `/v1/watchlists`, `/v1/watchlists/{id}` | Bearer | 50 lists | none | — |
| Watchlist tickers add/remove/note | READY | `/v1/watchlists/{id}/tickers[/{ticker}]` | Bearer | — | none | — |
| Alerts list/create/update/delete | READY | `/v1/alerts`, `/v1/alerts/{id}` | Bearer | Free 1, Pro 5, Premium 25 (advisory-locked) | none | — |
| Alert events | READY | `GET /v1/alerts/events` | Bearer | — | none | — |
| Push device register/remove | READY | `/v1/me/devices`, `DELETE …/{id}`, `logout.push_token` | Bearer | max 10 | none | Nothing sends pushes yet (separate item) |
| Health / readiness | READY | `/healthz`, `/readyz` | none | — | none | — |

Before this run the six Custom Scan rows were **STREAMLIT_DEPENDENT**: the scan ran
only inside `ui/scans.py` / `ui/three_step_scanner.py`, every plan rule (universe,
rows, sessions, filters, ticker caps) existed only as disabled Streamlit widgets,
and universe caps lived in `st.session_state`.

## 2. Custom Scan: how it executes, and why it's asynchronous

Web path: `pages/custom_scan.py` → `ui.custom_scan.render_custom_scan` (filters as
widgets) → `ui.scans.render_scan_controls` (buttons; `do_scan`) →
`scan.universe_selection.resolve_scan_universe` (lists, caps, liquidity pre-filter;
state in `st.session_state`) → `scan.execution.run_manual_scan_execution` →
`scan.engine.run_breakout_scan` (prices via `data.prices`, Alpaca first) →
post-processing (`score_prebreakout`, `score_ai_confidence`, ranking).

The engine and execution modules are not Streamlit-dependent: the scheduled scans
already run them headless, and this run verified `run_manual_scan_execution` in a
plain worker thread (900 symbols, mocked prices). Only the request layer (widgets,
session state, UI-only plan checks) was Streamlit-bound. The API now reuses the
same three functions; there is no second scanner.

Measured durations (production runs, `runs` table, 2026-09-28 to 2026-10-06):
S&P 500 2–10 s, Combo 31.9 s, NASDAQ 94.0 s, US market 224.6 s (before the
liquidity pre-filter, `a649170`). Minutes-long requests are fragile for browsers
and phones and would tie up API workers, so scans are jobs: `POST` returns 202
immediately and the client polls. No new infrastructure: a bounded in-process
thread pool (`API_SCAN_WORKERS`, default 1) and an API-owned table. A deploy or
crash interrupts running jobs; they read `failed` / `interrupted` after 20 minutes
without progress (60 minutes for a job still queued), and an expired job never starts later.

## 3. Changes made

| File | Why |
|---|---|
| `api/custom_scans.py` (new) | Server-side plan rules (`plan_scan`) mirroring the web's widget rules; `run_scan` reuses the web's universe resolution and scan execution |
| `api/scan_jobs.py` (new) | `api_scan_jobs` table, one active job per account, 10 across the service, stale-job expiry, worker pool |
| `api/main.py` | `POST /v1/scans`, `GET /v1/scans`, `GET /v1/scans/{scan_id}`; 403/409/503 handlers; datetimes always zoned |
| `api/models.py` | `ScanJob`, `ScanParams`, `ScanProgress`, `ScanResult` with enums (OpenAPI) |
| `api/ratelimit.py` | `scan` bucket: 30 per hour per account |
| `api/today.py`, `api/scans.py` | Timestamps always carry a timezone (naive `TIMESTAMP` values are UTC) |
| `scripts/api_acceptance.py` | Journey adds custom scans (plan refusals, ticker and watchlist scans polled to completion), device register/re-register/refresh/remove; OpenAPI path check covers all current endpoints |
| `tests/test_api_scans.py` (new) | Plan rules (adversarial), routes, real job store and a whole scan on Postgres, timestamps |
| `docs/API.md`, `GO_LIVE.md` | Contract, conventions, Postman sequence, hsf-api env (Alpaca keys, `API_SCAN_WORKERS`) |

No change to scanner scoring, models, signals, research, billing prices or plans.

## 4. Endpoints added (shapes; values in <> are placeholders)

```
POST /v1/scans
{"universe": "combo", "filters": {"top_n": 100, "session": "regular", "max_combo": 3000}}
→ 202 {"scan_id": "<32 hex>", "status": "queued", "universe": "combo",
       "params": {... "full_lists": false, "max_combo": 3000, "max_results": 100 ...},
       "progress": {"phase": "queued"}, "result": null}
→ 403 {"detail": "US market scans are part of Premium."}          (plan)
→ 409 {"detail": "You already have a scan running…", "scan_id": "<id>"}
→ 422 validation · 429 + Retry-After (30/h) · 503 + Retry-After (busy)

GET /v1/scans/{scan_id}
→ 200 {"status": "running", "progress": {"phase": "scanning", "symbols": <n>, "elapsed_s": <s>}}
→ 200 {"status": "complete", "result": {"label": "Combo", "symbols_scanned": …, "duration_s": …,
       "total": …, "setups": [<same rows as /v1/scans/latest>]}}
→ 200 {"status": "failed", "error": "The US market list isn't available right now…"}
→ 404 when it isn't yours

GET /v1/scans?limit=10   → your recent jobs, newest first, no rows
```

## 5. Tests

- `tests/test_api_scans.py`: 22 tests. Every universe × plan; row caps per plan;
  Pro features refused for Free; extended sessions outside their hours; Pro can't
  uncap, Premium is uncapped; bad input; routes (202, 403 for 8 bypass attempts,
  409, 404 for another account's scan and watchlist, 422, 429, 503, OpenAPI);
  on Postgres: store rules, stale expiry (queued and running; an expired job never
  starts or revives), a whole scan in the background (engine,
  shaping, redaction, scan history) and a failed scan's message.
- Full suite with Postgres: **2636 passed**, 0 skipped. Smoke-style environment
  (no Streamlit): 2258 passed. Ruff clean; Bandit: no findings in `api/` or `scan/`.
- Isolated acceptance (`scripts/api_acceptance.py`, external HTTP client, 5 plans):
  **195 PASS, 0 FAIL, 1 BLOCKED** (expired-token check needs the server secret;
  covered by unit tests). Includes ticker and watchlist custom scans completing,
  plan refusals, devices, isolation between two Pro accounts, CORS. A 120-symbol
  S&P 500 custom scan through the API completed with 100 ranked rows.

## 6. Remaining blockers (genuine, for replacing Streamlit)

**Must exist before Web v2 replaces Streamlit** (features customers pay for, per
the pricing table, with no API yet):

1. Live Day Trader monitor (Pro): intraday movers, VWAP, relative volume.
2. Scan history & historical research (Pro): the API lists only API-started
   scans; the web's saved runs (`runs` table) and track record aren't exposed.
3. Earnings calendar & filters (Pro): only `earnings_days` on a stock.
4. AI scan summaries, results chat and setup notes (Premium).
5. Alpaca paper-trading workflow and Journal (Premium).
6. Market Brief page (all plans).

**Deployment, before custom scans work in production:** add
`ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` to hsf-api; move hsf-api off the
free plan (sleeps; 512 MB). A US market scan holds thousands of price histories,
so keep `API_SCAN_WORKERS=1` unless the instance has more memory.

**Can migrate later** (don't block Web v2): admin console, Kalshi labs,
diagnostics, research dashboards, CSV export (client-side from rows), account
deletion (needed for the App Store, not the web).

## 7. Final verdict

**READY_WITH_BLOCKERS.** The boundary HSF core → FastAPI → clients now holds for
every core workflow, including Custom Scan, with plan rules enforced on the server.
A Web v2 foundation can start now; Streamlit can't be retired until the six paid
features above have APIs.

## 8. Recommended next run

Smallest API run to reach READY_FOR_WEB_V2: **"API v1 paid-feature parity"**:
`GET /v1/runs` + `GET /v1/runs/{id}` (scan history, Pro), `GET /v1/earnings`
(Pro), `GET /v1/brief` (Market Brief), `GET /v1/day-trader` (Pro, intraday movers
snapshot), then AI summaries (Premium) and paper trading (Premium) as their own
runs. In parallel, the owner adds the Alpaca keys to hsf-api and runs the live
acceptance (P1-63).
