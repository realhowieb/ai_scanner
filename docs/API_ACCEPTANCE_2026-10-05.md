# HSF API: live acceptance and frontend readiness (2026-10-05)

**Verdict: API_READY_WITH_GAPS.** In isolated testing the full journey passes for all
four plans; the defects it found are fixed and in CI. Nothing was verified live: this
session's network policy blocks `hsf-api.onrender.com`, and no test accounts exist.
Account flows (sign-up, password reset, billing) and push notifications are not in the API yet.

Evidence labels used below:

- **CI**: code review and GitHub Actions on the commit.
- **ISOLATED**: local Postgres 16 + uvicorn on this machine, seeded test data, never production.
- **LIVE**: the deployed service. **No live check ran in this session.**

## 1. Commit and deployment evidence

| Item | Value |
|---|---|
| Commit reviewed (start) | `f6118487292c0eddcd8bbbb2b22ae8354c5bf9c8`; dev == main (0 ahead, 0 behind) |
| Commit with fixes | `d6dc4b2` on dev; CI 9/9 green; **not on main**, not deployed |
| AGENTS.md | Not present in the repository (nor CLAUDE.md) |
| Deployed URL | `https://hsf-api.onrender.com` is **not** in any repo config. The owner reported it live on 2026-10-04 (/healthz, /docs, login, /v1/me, /v1/today). This session couldn't verify it: outbound requests got proxy 403 (`connect_rejected`). |
| What is deployed | Unknown from here. Render deploys hsf-api from `main` (docs/API.md), so main `f611848` is expected after its auto-deploy. Not verified, and the fixes in `d6dc4b2` are not live. |

## 2. Acceptance table

### P0: live deployment

| Check | Result | Evidence |
|---|---|---|
| /healthz, /docs, /openapi.json on the deployed URL | **BLOCKED** | proxy 403; needs the host allowed in the environment's network settings |
| Deployed schema has the current endpoints | **BLOCKED** (live) / PASS (ISOLATED) | isolated: all 15 required paths present |
| Cold vs warm latency, live | **BLOCKED** | Render free plan sleeps after 15 min; cold start untested |
| Browser CORS for the frontend origin | **BLOCKED** (live) / PASS (ISOLATED) | isolated: preflight from the listed origin allowed; an unlisted origin is refused. Live `API_CORS_ORIGINS` value unknown. |
| Liveness vs database readiness | PASS (ISOLATED, after fix) | new `/readyz`: 200 + scan age when the DB answers, 503 when down; `/healthz` stays 200 |

### P0: authentication and account isolation (ISOLATED unless marked)

| Check | Result | Evidence |
|---|---|---|
| Valid / invalid login | PASS | 200 with a token pair; 401 for unknown account |
| Refresh rotation + 30 s retry grace | PASS | new refresh token issued; immediate retry of the old one 200 |
| Logout | PASS | 204; refresh afterwards 401 |
| Access token after logout | Documented | stays valid until expiry, at most 15 min (stateless JWT) |
| Missing / invalid / expired token on protected routes | PASS | 401 on all 6 routes for each; expired token minted with the test secret 401 |
| Two accounts can't read or change each other's watchlists, notes, alerts | PASS | 8 cross-account attempts all 404; owner's data intact; not listed for the other account |
| Plan change affects existing sessions | PASS | downgrade: /me = basic and scan cap 25 on the same token; upgrade: premium. Deactivation: 401 |
| Premium model fields hidden below Premium | PASS | `prob` null and no `prebreakout` signal for Free/Pro on scan and stock page; `signal=prebreakout` 403. Shared caches never leak (Premium → Pro → Premium order checked). |
| All of the above against the live service | **BLOCKED** | no network access, no test accounts |

### P0: scanner and stock details (ISOLATED)

| Check | Result | Evidence |
|---|---|---|
| Matches the web app's saved-scan data and builders | PASS | 250 setups; ranking, score, status, signals, n_signals and redaction identical to `ui.market_scans.top_setups` on `normalize_results_to_df` (the web Scanner path) for Pro and Premium |
| Ranking, filters, pagination | PASS | sorted by score; `min_score`, `signal` work; page 1 + 2 = first 10 |
| Plan cap can't be bypassed with filters or offset | PASS | `offset=cap` returns nothing; filtered lists stay within the cap (Free 25, Pro 100, Premium 200, Admin unlimited) |
| Scan time, empty results, unknown / invalid tickers, missing chart data | PASS | `scan_at` ISO; empty scan → `scan_at: null`; unknown ticker → 200 `in_latest_scan: false`; invalid → 422; no bars → `[]`; NaN close dropped |
| Strict JSON (no NaN/Infinity) | PASS | every scan and stock payload parsed with NaN/Infinity rejected; seed had NaN and inf |

### P0: watchlists and alerts (ISOLATED)

| Check | Result | Evidence |
|---|---|---|
| CRUD, bulk add, note | PASS | per plan: create, bulk add (invalid listed), note, remove, rename, delete |
| Duplicates, invalid input, ownership, plan limits | PASS | 409 duplicate name/alert; 422 bad alert input; 404 cross-account; Free 1 / Pro 5 / Premium 25 alerts → 403 at the limit |
| Concurrent creation vs limits | **FAIL → fixed** | before: a Pro account (limit 5) reached up to 10 alerts in 40 concurrent requests; identical alerts saved twice; one watchlist name saved 4 times. After: 5/5 in all 5 trials, duplicates saved once, one name once |
| Same records as the web app | PASS | API calls the same `db.watchlists` / `db.alerts` functions and tables |
| No email to real users | PASS | alert thresholds that can't fire (999,999 % move); no alert_events rows created |
| Real users untouched; only own records cleaned up | PASS | seeded "real user" watchlist, note and alert unchanged; every record created by the run deleted by id |

### P1: reliability and performance (ISOLATED)

| Check | Result | Evidence |
|---|---|---|
| Database outage | PASS | Postgres stopped: `/healthz` 200, every data route 503 with `Retry-After: 30` (no tracebacks), recovers on restart |
| DB failure during the scan read | **FAIL → fixed** | `list_runs` falls back to an empty list on error, which read as "no scan yet" for up to 60 s; an empty list is now only believed after a DB ping |
| Bounded caches, no user data in shared caches | PASS | scan cache 32 entries, stock cache 128; user watchlists/alerts are never cached; redaction applied per request on copies |
| Connection hygiene | **FAIL → fixed** | worker threads sat idle in a transaction holding table locks (an `ALTER TABLE` elsewhere waited forever and stalled readers); `db.earnings` leaked one connection per stock page |
| Modest concurrency (20 clients, 1,000 requests) | PASS | 0 errors, 93 req/s, peak 24 DB connections, 0 left idle in a transaction, `ALTER TABLE` during load 1 ms |
| Request IDs, structured logs, freshness | **Missing → added** | `X-Request-ID` on every response, one JSON log line per request (no headers/bodies/tokens), `/readyz` scan age |

## 3. Fixes made (commit `d6dc4b2`) and tests

| Defect | Fix | Regression test |
|---|---|---|
| Alert limit and duplicate race | `db.alerts.create_alert`: per-user advisory lock, checks + insert in one transaction, `max_alerts` | `tests/test_alerts_atomic_create.py` (fails on the old code) |
| Same-name watchlist race, cap race | same lock in `create_watchlist` / `rename_watchlist`, `max_watchlists` | same file |
| Idle-in-transaction pool connections | `_WarmConn.close()` rolls back; API ends each request's transaction on its worker thread | `tests/test_api_db_sessions.py` (all 3 fail without the fix) |
| Earnings connection leak | `load_earnings_map` / `load_earnings_details_map` close the connection they open | `tests/test_api_db_sessions.py` |
| Watchlist default repair rewrote all rows on every read | update only rows whose flag is wrong | `tests/test_alerts_atomic_create.py` (row xmin unchanged) |
| Outage read as "no scan yet" | empty run list → DB ping → 503 | `tests/test_api_v1_data.py::ReliabilityTests` |
| No readiness / request ids / access log | `/readyz`, `X-Request-ID`, JSON access log | `tests/test_api_v1_data.py::ReliabilityTests` |

Tests run: full suite 2,523 passed, 11 skipped; the Postgres-only tests ran against the
isolated database (`HSF_TEST_PG_URL`) and pass; CI 9/9 on `d6dc4b2`. The isolated
acceptance journey (`scripts/api_acceptance.py`) passes 137/137, and the isolated-only
extras pass 8/8. Scores, ranking, model behavior and Streamlit pages are unchanged. The pool
change is shared with the Streamlit app, but it only matches what a real `close()`
already did (end the transaction).

## 4. Latency (ISOLATED only; local Postgres, so database round trips are near zero)

| Route | p50 | p95 |
|---|---|---|
| /v1/scans/latest | 135 ms | 261 ms |
| /v1/me | 138 ms | 255 ms |
| /v1/alerts | 178 ms | 284 ms |
| /v1/watchlists | 188 ms | 323 ms |
| /v1/today | 189 ms | 293 ms |
| /v1/stocks/{ticker} | 489 ms | 661 ms |

At 20 concurrent clients. Single client, warm: /healthz 3 ms. First sign-in after process
start: 1.1 s (first connections plus bcrypt), then about 12 ms. Production adds a Neon round trip
per query and a Render cold start after 15 idle minutes; **neither has been measured**.
No load test was run against production.

## 5. Remaining launch blockers

1. **Live verification**: allow `hsf-api.onrender.com` in this environment's network
   settings, create five dedicated test accounts (Free, Pro, a second Pro, Premium, Admin),
   promote `d6dc4b2`, then run `scripts/api_acceptance.py --allow-writes` against the live URL.
2. **Render free plan**: sleeps after 15 min (cold starts on first app open); move to Starter before users.
3. **`API_CORS_ORIGINS`** must list the new frontend's origin (unknown today).
4. **Per-IP sign-in rate limit**: only per-account today; open since P1-59 step 1.
5. **Account flows missing from the API**: sign-up, email verification, password reset,
   change password, delete account; billing checkout / portal (the billing service is
   called by the Streamlit app, not reachable with an API token).
6. **Mobile push**: no device-token registration; alerts reach phones only by email.

Non-blocking findings: the reason line can say "3 confirming signals" while a Pro user
sees 2 (PreBreakout counted, then hidden; same on the web). `ensure_earnings_table` runs
two full-table `UPDATE`s on every earnings read. The stock page makes one query per watchlist.

## 6. Frontend readiness mapping

Auth for every screen: `POST /v1/auth/login` → keep the refresh token in secure storage
(Keychain / Keystore; httpOnly cookie via a BFF on web). On a 401, call `POST /v1/auth/refresh` once,
retry, and sign out if that fails. A retry within 30 s is safe. Show 503 as "Service
temporarily unavailable" and retry after `Retry-After`. Send `X-Request-ID` and show it in
error reports.

| Screen | Endpoints | Data available | Missing in the API | States to design |
|---|---|---|---|---|
| **Today** | `GET /v1/today`, `GET /v1/me` | market phase; Before the open / After the close movers (Pro+, `locked` below); top setups; last-session recap; per-section `errors` | "New since your last visit" and "Your watchlist in the latest scan" sections of the web Today page; market brief / regime | loading; empty scan (`top_setups.state = empty_scan`); weak market (`no_qualifying`); locked cards (Free); stale (compare `scan_at` with now; `/readyz` has scan age); section error; 503 |
| **Scanner** | `GET /v1/scans/latest` (limit/offset/min_score/signal) | ranked setups with score, status, signals, price, change, gap, RVOL, breakout score, `prob` (Premium) | sort options, ticker search, sector/price filters, rising/falling vs the previous scan, pre/after-hours session scans, CSV export (Pro), custom scans | loading; empty (`scan_at: null`); filtered to nothing; `limited: true` upsell (plan cap); `signal=prebreakout` 403 below Premium; stale scan |
| **Stock Details** | `GET /v1/stocks/{ticker}`, watchlist + alert endpoints | score + components, status, signals, reasons, risks, watch next, lifecycle, history/cohort context, up to 120 daily bars, your watchlists and alerts on it | intraday bars, live quote (price is from the last scan), news, AI notes (Premium), earnings date beyond 5 days | loading; not in latest scan (`in_latest_scan: false`); score from history (`from_history`); no chart (`bars: []`); Premium fields null; invalid ticker 422 |
| **Watchlists** | `GET/POST /v1/watchlists`, `GET/PATCH/DELETE /v1/watchlists/{id}`, `POST .../tickers`, `PATCH/DELETE .../tickers/{t}` | lists, default flag, items with added date and note | quotes / scores per item (needs a batch quote or join with the scan), duplicate / move / copy, file import | empty list; duplicate name 409; 50-list cap 403; invalid tickers listed in the response; not found 404 |
| **Alerts** | `GET/POST /v1/alerts`, `PATCH/DELETE /v1/alerts/{id}`, `GET /v1/alerts/events` | alerts with limit/used/email_enabled; fired events | alert-type metadata (form rules per type), outcome scorecards, read/unread events, push device registration | none yet; at plan limit 403 (upsell); duplicate 409; validation 422; email off below Pro |
| **Account / Plan** | `GET /v1/me` | email, name, plan, label, alert limit, entitlement flags | sign-up, email verification, password reset, change password, upgrade (checkout) / manage (portal), email preferences, delete account | plan badge; upgrade prompts driven by `entitlements` and `alert_limit` |

## 7. Recommended next implementation run

"API step 7: account and billing endpoints, then live acceptance."

1. Unblock the live run: allow the host, create the five test accounts as secrets, promote
   `d6dc4b2`, run `scripts/api_acceptance.py` live, and record cold and warm latency.
2. `/v1/auth/signup`, `/v1/auth/verify-email`, `/v1/auth/password-reset` (request and
   confirm), `/v1/auth/password`, reusing the web app's account functions and its emails.
3. `/v1/billing/checkout` and `/v1/billing/portal`: API-authenticated calls to the billing
   service, returning Stripe URLs.
4. Per-IP sign-in rate limit.
5. Small read gaps for the first screens: batch quotes/scores for watchlist items, alert-type
   metadata, and "new since last visit" (needs a per-user last-seen run id).
6. Push: `POST /v1/devices` (APNs/FCM token) and a sender hook in the alert runners.
