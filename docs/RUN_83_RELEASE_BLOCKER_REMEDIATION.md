# Run 83 — Release Blocker Remediation

## Executive Summary

Both Run 82 launch blockers are fixed and proven by reproduction before and after, plus
new tests that run in normal CI.

- **B1 (billing portal) is FIXED.**
  - The billing service's checkout and portal endpoints now require proof of the
    signed-in HSF account: a single-use, 10-minute token issued by the app and verified
    against the shared database.
  - The Stripe customer is resolved server-side from that account. A client-supplied
    email that doesn't match gets a 403. Client-supplied customer IDs are ignored.
  - Unauthenticated, forged, replayed or cross-account requests never reach Stripe.
- **B2 (account-state carryover) is FIXED.**
  - Logout now clears all account-derived session state.
  - Independently, `app.py` enforces an identity boundary after every sign-in: if the
    signed-in account differs from the one the session state belongs to, account state
    is cleared before anything renders.
  - The narrow audit found one more instance of the same defect: user A's cached
    **Stripe portal link** (`_portal_url`) could be shown to user B on the Billing page.
    It is fixed too.
- **Stripe restore token is SINGLE-USE.**
  - `?rt=` was a reusable 14-day login session. It is now a separate, single-use,
    2-hour restore token (one per link).
- **Tornado is PATCHED:** 6.5.7 → 6.5.8, the smallest release that fixes all three
  advisories and is compatible with Streamlit 1.54.0.

Run 82 NO-GO conditions removed: **YES.**

## Baseline

| Item | Value |
|---|---|
| Branch | `dev` (main = dev at `c01598a`; dev had the Run 82 report commit `92dba85`) |
| HEAD before | `92dba85` |
| Affected files | `billing_service/main.py`, `pages/billing.py`, `ui/checkout.py`, `ui/auth.py`, `ui/app_session.py`, `app.py`, `requirements.lock`, `.github/workflows/smoke.yml` |
| Existing tests | `tests/test_billing_service.py` (billing contract), `tests/test_run80_mobile_state.py` (logout isolation), `tests/test_boot_stale_module.py` (startup) |

**B1 reproduced** (billing test harness, Stripe and DB mocked, no credentials sent):

```
REPRO portal:   200 {'portal_url': 'https://billing.stripe.test/p/VICTIM'}
REPRO checkout: 200 {'portal_url': 'https://billing.stripe.test/p/VICTIM', 'mode': 'portal'}
REPRO stripe portal created for: ['cus_VICTIM', 'cus_VICTIM']
```

**B2 reproduced** (the app's own logout clearing and Market Brief reader):

```
after logout, survived: ['_loaded_user_settings', '_watchlist_prior_rows', 'active_watchlist_quote_rows']
Bob's Market Brief watchlist rows: [{'ticker': 'ALICEPICK', …}] | DB queried for Bob: False
```

**Restore token before:**
- Created by `ui.auth_sessions.create_session` (a 14-day `auth_sessions` row, random
  UUID).
- Put into both the checkout-success and portal-return URLs (the same ID in both).
- Validated by `get_username_for_session`, which is a plain `SELECT` with no
  consumption. The token was only removed from the address bar.
- Any copy of the URL signed that user in again until expiry.

**Tornado before:** `tornado==6.5.7` in `requirements.lock`, with advisories
PYSEC-2026-3928, GHSA-wwv5-g3v4-889x and GHSA-8423-8fgw-73vq. All three are fixed in
6.5.8.

## B1 — Billing Portal Authentication

### Root Cause

- `POST /create-portal-session` took `{"email": …}` from the request body, looked up that
  user's `stripe_customer_id` and returned a Stripe billing-portal URL.
- `POST /create-checkout-session` did the same for a user with an active subscription.
- Neither endpoint had any caller authentication.
- The service is public on Render, and its URL is the committed default in a public
  repository.

Every path that can create or return a portal session (whole repository searched):

| Entry point | Before | After |
|---|---|---|
| `billing_service/main.py` `/create-portal-session` | unauthenticated | token required |
| `billing_service/main.py` `/create-checkout-session` (portal for active subscribers) | unauthenticated | token required |
| `pages/billing.py` "Manage subscription" → `_create_portal_url` | sent email only | sends a fresh token per attempt |
| `ui/checkout.py` `create_checkout_url` (Billing page and sidebar upgrade card via `ui/app_runtime.py`) | sent email only | sends token; refuses to call the service if no token can be issued |
| `ui/admin_results_tab.py` | calls `/health` only | unchanged |
| `billing_service/main.py` `/debug/status` | public config/DB status | 404 unless `BILLING_DEBUG_STATUS=1` on the service |

### Fix

- **App side (`ui/auth_tokens.py`, new):**
  - `issue_token(username, "billing")` creates a `secrets.token_urlsafe(32)` token.
  - Only its SHA-256 hash is stored in `hsf_auth_tokens`, with a purpose and a 10-minute
    expiry.
  - The app sends the raw token in the `X-HSF-Auth` header (`ui/checkout.billing_auth_headers`).
- **Service side (`billing_service/main.py`):**
  - `_authenticated_user` consumes the token with one atomic
    `DELETE … WHERE token_hash=… AND purpose='billing' AND expires_at>now() RETURNING username`.
  - That returned username is the identity used for the rest of the request.
  - Its Stripe customer comes from `users.stripe_customer_id` for that username.
  - The service can't import app modules (flat deploy on Render), so the SQL is
    duplicated. A test asserts the two copies are identical.
- **Why this design:** it needs no new shared secret. The billing service already reads
  the same database, so there is nothing new to configure across the three secret
  stores.
- **Error messages:** user-facing errors on these endpoints no longer interpolate
  exception text (DB hosts, Stripe IDs). Details are logged server-side as exception
  class names only.

### Authorization Model

```
signed-in HSF session (app, server-side)
 → issue single-use billing token for session_state["username"]
 → billing service consumes token → trusted username
 → users row for that username → stripe_customer_id
 → Stripe portal / checkout for that customer only
```

- Missing, unknown, expired, replayed or wrong-purpose token → **401**.
- Token DB unavailable → **503**.
- Payload `email` ≠ token identity → **403**.
- No mapped customer → **400**.
- All of these happen before any Stripe call.
- Client fields such as `customer`, `customer_id` and `stripe_customer_id` are never read.
- Stripe errors → **502/500** with a generic message. No entitlement is changed.

### Tests

`tests/test_run83_billing_auth.py` runs in the CI `billing-contract` job. It uses the
real token SQL on an in-memory SQLite stand-in, with Stripe mocked.

- signed-out request refused, with no Stripe or user lookup;
- malformed, unknown, oversize and wrong-purpose (restore) tokens refused;
- token single-use (second call 401, one Stripe call total);
- valid user gets their own portal;
- **user A cannot get user B's portal** (403, no Stripe, "cus_BOB" never appears);
- forged `customer`/`customer_id`/`stripe_customer_id` ignored (Stripe called with
  `cus_ALICE`);
- missing mapping → 400, no Stripe;
- account DB down → 503, fails closed, no internals in the error;
- Stripe error → 502, no internals, `_set_user_plan_by_email` never called;
- checkout endpoint: anonymous → 401, cross-account → 403, own identity → own portal;
- `/debug/status` hidden;
- SQL parity with `ui/auth_tokens.py`.

The existing `test_billing_service.py` checkout tests now send a token; their
assertions are unchanged.

App side (`tests/test_run83_restore_token.py::AppCheckoutCallTests`, both CI envs):
- checkout sends `X-HSF-Auth` plus two distinct restore tokens;
- **no token → billing service never contacted**;
- the Billing page mints a fresh token for each retry attempt.

### Result

**FIXED.** After the fix, the same reproduction gives:

```
REPRO portal:   401 {'detail': 'Please sign in again to manage billing.'}
REPRO checkout: 401 {'detail': 'Please sign in again to manage billing.'}
REPRO stripe portal created for: []
```

## B2 — Account-State Isolation

### Root Cause

Logout clears `ACCOUNT_SESSION_KEYS`, but several account-derived session keys were not
in that list:
- `active_watchlist_quote_rows` (My Stocks tiles cache the user's watchlist quotes);
- `_watchlist_prior_rows`;
- `_loaded_user_settings`.

`ui/market_brief._watchlist_rows` returns the session rows **before** querying the
current user. So after A → logout → B on the same tab, B's "📋 Your watchlist today"
listed A's tickers, and "📧 Email me this brief" would send them to B.

### Fix

- **Logout clearing (`ui/app_session.ACCOUNT_SESSION_KEYS`):** now includes all
  account-derived non-widget state found by enumerating every session key the app writes:
  - watchlist quotes, prior rows and settings flag;
  - cached **portal URL** and post-checkout flags;
  - pending watchlist actions;
  - Day Trader watch symbols and baseline;
  - the user's latest results and related signatures;
  - earnings enrichment of those results;
  - Market Brief diff state;
  - pending AI screener and 3-step settings;
  - first-run activation flags;
  - alert-form prefills;
  - Alpaca paper key inputs.
- **Identity boundary (`ui/app_session.enforce_account_boundary`, called in `app.py`
  immediately after `auth_ui()`):**
  - Before any page switch, it compares a hash of the signed-in username with
    `_hsf_account_owner`, the owner of the session's account state.
  - The owner marker deliberately survives logout, and it stores a hash, not the email.
  - On a mismatch it clears every account key except the identity the new sign-in just
    set (`user_id`, `username`, `display_name`, `tier`, `plan`, `is_admin`,
    `authentication_status`) and the pending shared-link destination the new user chose.
- **Browser preferences** (Scanner view, lenses, saved screens, tour) are untouched by
  both paths, as intended.

### Identity Boundary

- **Where identity is set:** a signed-in identity is only ever established by
  `auth_ui()` in `app.py` (login form, `rt` restore, cookie restore). Sub-pages only
  read `session_state["username"]`.
- **Why it is placed there:** running the boundary right after `auth_ui()` and before the
  shared-link and Today redirects covers logout → login, session restore, expiry and any
  future auth path.
- **Stale-module safety:** the new `app.py` import is inside the Run 81 hotfix's guarded
  block, so a stale cached `ui.app_session` self-heals (`tests/test_boot_stale_module.py`
  passes).

### Tests

`tests/test_run83_account_isolation.py` (6 tests in both CI envs; 4 screen tests where
Streamlit is installed, i.e. the CI dependency job):
- same-user reruns keep state;
- logout clears account keys and keeps browser prefs;
- **identity change without logout** clears A's state, keeps B's identity and shared
  link;
- the owner marker survives logout and is a hash;
- the audited keys are all in the clear list;
- `app.py` calls the boundary after auth and before any page switch;
- **Market Brief** shows B's own watchlist (B's DB queried);
- **B's empty watchlist does not fall back to A's**;
- **Today** watchlist is B's own;
- **Stock Intelligence** "★ Watching" does not carry over.

Each screen test runs both via logout and without logout.

### Result

**FIXED.** The Run 82 reproduction now gives:

```
after logout, survived: []
Bob's Market Brief watchlist rows: [] | DB queried for Bob: True
```

## Other Account-State Audit

| Surface | Finding | Action |
|---|---|---|
| **Billing portal URL** (`_portal_url`) | **Same defect.** A's live Stripe portal link survived logout; a paid user B opening Billing would see "Customer Portal ready!" with A's link. | Fixed (cleared on logout and at the boundary) |
| Tier / entitlements / is_admin | Already cleared on logout; tier re-read from the DB; `ui/user_lookup` tier cache compares the user | No change needed |
| My Stocks / Today / Stock watch status | `active_watchlist_*` cleared; DB reads keyed by user; Run 71 caches keyed by (user, per-user version) | Covered by B2 fix + tests |
| Alerts | Alerts are read from the DB per user; the only session carry-over was alert-form prefills | Prefills added to the clear list |
| Paper-trading keys | Stored encrypted per user; the `pt_key`/`pt_secret` inputs added to the clear list as defense in depth | Added |
| Server-persisted preferences | `user_settings` keyed by user; `_loaded_user_settings` flag now cleared | Covered |
| Billing customer mapping | Now server-side only (B1) | Covered |
| Browser preferences (cookie jar) | Intentionally browser-scoped (Run 80 policy): view mode, lenses, saved screens, tour | Unchanged by design |

No other leak of this kind was found.

## Stripe Restore Token

### Before

- Reusable 14-day login session ID, shared by both return links.
- Validated with no consumption.
- Replayable from browser history, Stripe records or any copied URL.

### Fix

- `ui/auth_tokens.issue_token(user, "restore")`: separate tokens for the
  checkout-success and portal-return links (`ui/checkout.py`, `pages/billing.py`).
- 256-bit random value; only a SHA-256 hash is stored; bound to purpose `restore` and a
  **2-hour** expiry.
- `ui/auth.py` consumes it with `consume_token(rt, "restore")`, an atomic
  `DELETE … RETURNING`.
- **Consumption semantics:**
  - a successful restore consumes the token;
  - an unknown, expired or wrong-purpose attempt consumes nothing and restores nothing;
  - a billing token cannot sign anyone in, and a restore token cannot open billing.
- Tokens live in a separate table (`hsf_auth_tokens`), so they can never be used as a
  login cookie. Login sessions (`auth_sessions`) are unchanged.
- Nothing logs a token (a test checks the touched files).
- Old `rt` links minted before this change no longer restore; the user simply signs in.

### Replay Test

`tests/test_run83_restore_token.py` runs in both CI envs and the billing job:
- valid token restores once, then replay is rejected and the row is gone;
- expired token rejected;
- invalid, empty and oversize tokens rejected, and a failed attempt consumes nothing;
- wrong-purpose tokens rejected in both directions without being consumed;
- a token is bound to its own user;
- 20 tokens are all distinct, at least 40 characters, and only hashes are stored;
- unknown purpose and blank user refused;
- DB unavailable fails closed.

### Result

**SINGLE-USE.**

## Tornado Security Patch

### Before

`tornado==6.5.7`: three advisories (PYSEC-2026-3928, GHSA-wwv5-g3v4-889x,
GHSA-8423-8fgw-73vq), all fixed in 6.5.8. Tornado is Streamlit's internet-facing web
server.

### After

- `tornado==6.5.8` in `requirements.lock`. This is a one-line pin change, the smallest
  patched release; 6.5.9 and 6.5.10 exist but aren't needed.
- No other package changed.
- The lock is normally produced by the Freeze Dependency Lock workflow. This single pin
  was edited by hand and verified as below.
- `cryptography`, `soupsieve` and `GitPython` are left for Run 74, as Run 82 classified
  them.

### Compatibility Verification

- Streamlit 1.54.0 requires tornado `!=6.5.0,<7,>=6.0.3`, so 6.5.8 is allowed.
- **A fresh venv installed from `requirements.txt` + `requirements-dev.txt` with the
  modified lock** resolves cleanly:
  - `pip check`: "No broken requirements found";
  - versions: tornado 6.5.8, streamlit 1.54.0, pyarrow 23.0.1, pandas 2.3.3.
- In that production-parity environment:
  - the full test suite passes (below);
  - `scripts/streamlit_smoke.py` boots the app (rc 0);
  - `deployment_doctor` reports 22 [OK];
  - the CI import smoke passes (`config`, `scheduler.cron_runner`,
    `_load_universe('SP500')`).
- This environment also removes the local drift noted in Run 82 (Streamlit
  1.49.1 / pyarrow 25).
- Because the lock also feeds scheduled scans, **re-run autonomy certification after
  this reaches main** (the Run 74 rule).

## Files Changed

| File | Change |
|---|---|
| `ui/auth_tokens.py` (new) | Single-use, purpose-bound, hashed tokens (`issue_token`, `consume_token`) |
| `billing_service/main.py` | `X-HSF-Auth` verification on checkout and portal; identity from the token; customer from the DB; generic errors; `/debug/status` gated |
| `ui/checkout.py` | Billing token header; separate single-use restore tokens; fail closed without a token |
| `pages/billing.py` | Fresh billing token per attempt; portal return uses a restore token |
| `ui/auth.py` | `rt` consumed as a single-use restore token (line count 724, within budget) |
| `ui/app_session.py` | Expanded `ACCOUNT_SESSION_KEYS`; `IDENTITY_KEYS`, `ACCOUNT_OWNER_KEY`, `enforce_account_boundary` |
| `app.py` | Calls the identity boundary right after `auth_ui()` (831 lines, within budget) |
| `requirements.lock` | `tornado==6.5.8` |
| `.github/workflows/smoke.yml` | `billing-contract` job also runs `test_run83_billing_auth.py` and `test_run83_restore_token.py` |
| `tests/test_billing_service.py` | Checkout tests send a token (assertions unchanged) |
| `tests/_token_db.py` (new) | In-memory SQLite stand-in for the token table |
| `tests/test_run83_billing_auth.py`, `tests/test_run83_account_isolation.py`, `tests/test_run83_restore_token.py` (new) | Run 83 tests |

## Test Results

| Suite | Collected | Passed | Failed | Skipped | xfail | Warnings | Duration |
|---|---:|---:|---:|---:|---:|---|---:|
| **Production-parity venv** (locked deps incl. tornado 6.5.8), `-X dev -W always`, outbound network blocked | 1901 | 1863 | 0 | 38 (all "fastapi not installed"; covered by the billing job) | 0 | 1 (Streamlit AppTest temp dir, third-party) | 127 s |
| Lightweight CI env (`requirements-dev`) | 1901 | 1784 | 0 | 117 | 0 | 0 | 17.7 s |
| `unittest discover -s tests` (production-parity) | 1858 run | OK | 0 | 38 | — | — | 61.3 s |
| Billing-contract job (exact CI command, isolated venv) | 111 | 110 | 0 | 1 | 0 | 81 (FastAPI/Starlette deprecations, third-party + known `on_event`) | 1.8 s |
| `ruff check .` | — | All checks passed | — | — | — | — | — |

Subtests: 171 passed.

By area (all passing):
- **B1:** `test_run83_billing_auth.py`, 12 tests, billing job.
- **B2:** `test_run83_account_isolation.py`, 10 tests (6 in the lightweight env).
- **Restore token:** `test_run83_restore_token.py`, 13 tests, all three environments.
- **Dependency:** boot smoke, deployment doctor, CI import smoke and `pip check` with
  tornado 6.5.8.
- **Existing critical tests:**
  - billing (`test_billing_service`, `test_run72_pricing`,
    `test_run73_tier_differentiation`, `test_tier_enforcement`, `test_app_session`);
  - alerts (`test_run77_breakout_alert_scale`, `test_alert_evaluation`);
  - state (`test_run80_mobile_state`), hierarchy (`test_run79_information_hierarchy`);
  - startup (`test_boot_stale_module`);
  - consistency and performance (`test_run70_*`, `test_run71_performance`);
  - research and frozen core (`test_autonomy_certification` incl. Gate U,
    `test_maturation_*`, `test_observation_*`, `test_signal_evidence`);
  - the 4 scikit-learn model tests, which now run in the production-parity venv.

## CI Verification

Every CI job's commands were reproduced locally in its own environment: `compileall`,
lint (same rules), import smoke, the lightweight `pytest tests/`, `unittest discover`,
`deployment_doctor`, `streamlit_smoke` and the `billing-contract` command.

- The new tests run in normal CI with no secrets, no network and no real Stripe:
  - `billing-contract` runs B1 and the restore-token tests;
  - `smoke` runs the restore-token tests and 6 of the isolation tests;
  - the `core-dependency-import-smoke` job (`unittest discover`, Streamlit installed)
    runs all isolation, restore-token and startup tests.
- CI was only extended; nothing was made optional or skipped.

**GitHub Actions on dev @ `4b5f0f2` (Smoke Checks run 36283588719):**

| Job | Result | Run 83 tests executed |
|---|---|---:|
| `billing-contract` | success: 110 passed, 1 skipped | 25 (12 billing-auth + 13 restore-token) |
| `smoke` | success: 1784 passed, 117 skipped | 19 (13 restore-token + 6 isolation) |
| `core-dependency-import-smoke` | success: Ran 1858, OK (42 skipped: FastAPI tests covered by `billing-contract`, 4 scikit-learn) | 23 (10 isolation + 13 restore-token) |
| `full-dependency-import-smoke` | success | — |
| `dependency-audit` (non-blocking) | failure, down from 34 vulnerabilities in 4 packages to **31 in 3**; tornado no longer reported (cryptography, soupsieve and GitPython remain for Run 74) | — |

ResourceWarnings in the CI log: 0.

## Frozen-Core Verification

`git diff` over `scan/`, `research/`, `analytics/`, `scheduler/`, `jobs/`, `data/`,
`db/`, `scripts/`, `ml_prebreakout.py`, `market_data.py`: **no changes.**

No change to:
- HSF Score, Breakout Score, ranking, `run_breakout_scan`;
- scheduled scans, models, research capture, cohorts, maturation;
- Gate U, Autonomous Research Mode, Run 56/58/61 behaviour.

Prices, plans, feature gates, alert limits, upgrade copy and checkout product choice are
unchanged.

## Run 82 Blocker Reconciliation

| Finding | Before | After | Evidence |
|---|---|---|---|
| B1 | BLOCKER | **FIXED** | Repro 200 with victim portal → 401, zero Stripe calls; 12 CI tests incl. A/B, forged ID, replay, fail-closed |
| B2 | BLOCKER | **FIXED** | Repro "survived 3 keys, Bob saw ALICEPICK" → "survived [], Bob's own DB queried"; identity boundary; 10 tests incl. no-logout transition and empty-watchlist fallback; `_portal_url` leak also fixed |
| Tornado | HARDENING | **PATCHED** | `tornado==6.5.8` in lock; clean resolve with Streamlit 1.54.0; full suite, boot smoke and doctor pass on it |
| Restore token | HARDENING | **SINGLE-USE** | Atomic consume; replay, expired, invalid and wrong-purpose rejected (13 tests); separate token table |

## Release Gate

Run 82 NO-GO conditions removed: **YES**

**Deployment notes (operational, not blockers):**

- **Deploy order:**
  - Streamlit Cloud deploys `dev` and now always sends `X-HSF-Auth`; the old billing
    service ignores the header.
  - Once the billing service on Render runs this code, it **requires** the header.
  - Keep the order dev → main, and confirm which branch Render deploys.
  - If Render updates before Streamlit Cloud, checkout and portal return 401 for a few
    minutes (no data exposure).
- **Table creation:** the new `hsf_auth_tokens` table is created on first use by either
  side (same database the service already uses for `stripe_processed_events`). No manual
  migration is needed.
- **Re-certification:** re-run autonomy certification once the tornado pin is on main.

## Recommended Next Step

**Run 84 — focused release-gate recheck.** Don't repeat Run 82 in full. Verify:
1. B1, B2, the restore token and tornado on the deployed services (after dev → main and
   the Render deploy):
   - an unsigned `/create-portal-session` returns 401;
   - Manage subscription still opens the right portal.
2. The manual checklist from Run 82: a Stripe test-mode round trip, Premium AI key and
   model, email, the A → logout → B check on the live app, and a phone walkthrough.
3. The autonomy re-certification result after the lock change.

Then issue the final GO / CONDITIONAL GO / NO-GO.
