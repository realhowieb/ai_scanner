# B4 — Global Post-Login Initialization Defect

## Summary

**B4: FIXED** (`79138ef`, CI green, run 36295060100). **Severity: P1.**

Every account type (Basic, Pro, Premium, Admin) rendered its first page after sign-in
without its account context. The sidebar plan fell back to "Basic", `entitlements` was
empty so paid features showed as locked, and admin status was missing. It only became
correct after visiting Scanner. The values were **empty**, not another user's and not
elevated, which is why this is P1 and not P0.

## Root Cause

`app.py` `main()` (before the fix):

```
auth_ui()                               # identity established (login form / cookie / rt restore)
enforce_account_boundary(...)           # Run 83
if hsf_after_login_page: switch_page(…) # ← redirect #1 (shared link)
… import checks, normalise username …
if should_land_on_today(...): switch_page("pages/today.py")   # ← redirect #2 (Today)
render_price_ticker(); render_header()
load_user_map → session_tier_state → is_admin → compute_entitlements
session_state["tier"], ["tier_key"], ["is_admin"], ["entitlements"] = …   # ← context written HERE
```

**Why Scanner "fixed" it:**
- `st.switch_page` ends the run. The first page after sign-in (Today, or a shared-link
  destination) therefore rendered with `tier_key`, `is_admin` and `entitlements` unset.
- Sub-pages read those keys from the session. Empty meant Basic with everything gated.
- Scanner *is* `app.py`, so opening it ran `main()` to the end and wrote the context.
- Scanner wasn't meant to initialize the account; it was simply the only page whose code
  ran past the redirect.

**Lifecycle captured headlessly** (real `app.py`, a freshly signed-in user, the session
recorded at the moment of `switch_page`):

| Point | Before fix | After fix |
|---|---|---|
| A. Before sign-in | no identity | no identity |
| B. Auth succeeded | `username` set | `username` set |
| C. At the redirect to Today | `tier_key=None, is_admin=None, entitlements=None` (all four tiers) | `tier_key=<tier>`, `is_admin` correct, `entitlements` resolved |
| D. First Today render | "Plan: Basic", paid features locked | correct plan and entitlements |
| E. After Scanner | correct (written by `main()`) | unchanged (already correct) |

**Live evidence before the fix:** the owner's admin account showed **"Plan: Basic"** in
the Today sidebar on a fresh session (Run 84 screenshot).

## Fix

**Pure reorder inside `main()`:** the existing tier / admin / entitlement resolution now
runs **before** the two redirects. The header now renders after them.

- No new code paths and no new imports, so there is no stale-module risk.
- `app.py` stays at 839 lines.
- There is no extra rerun, so no rerun loop is possible.

```
auth_ui() → enforce_account_boundary → import checks → normalise username
→ load_user_map → tier state → is_admin → compute_entitlements → write session context
→ shared-link redirect (wins) → Today redirect → header → Scanner…
```

**Other properties:**
- **One bootstrap:** identity is only ever established by `app.py` (login, cookie and
  restore-token paths all go through `auth_ui()`), and sub-pages consume the resolved
  context. Scanner is no longer the only page that sees it.
- **Unresolved is never shown as Basic:** the first rendered page always has the resolved
  tier. If tier resolution itself fails, the existing fail-closed fallback (Basic) applies,
  as before.
- **Run 83 isolation preserved:** the identity boundary still runs first. Old account
  state is cleared before the new context is written.
- **Deep links preserved:** the shared-link destination still takes precedence over Today,
  and now also receives the resolved context.

## Tests

`tests/test_b4_post_login_context.py` (3 tests, run in the CI dependency job). They run
the real `app.py` headlessly with **no Scanner visit and no extra rerun**, and assert
**identity and entitlements**, not just the label:

- **Basic, Pro, Premium, Admin:** at the redirect to Today, the user, `tier_key` and
  `is_admin` are correct, the tier's key entitlement is on, and the next tier's is off
  (not elevated). Admin is modelled the way production does it, via `ADMIN_USERS`.
- **Shared-link destination** (`pages/stock.py`): same context, and the deep link wins.
- **Six account transitions** after logout clearing and a previous-owner marker:
  basic→pro, basic→premium, basic→admin, admin→basic, premium→basic, pro→admin.
- **All three fail on the pre-fix `app.py`** with `tier_key=None, is_admin=None,
  entitlements=None`.

**Regression** (production-parity venv with the Run 83C lock, outbound network blocked):
- **Full suite:** 1919 collected, **1881 passed, 0 failed, 38 skipped** (all FastAPI,
  covered by the billing job), 172 subtests, 137 s.
- **Other suites:** lightweight CI env 1788 passed; `unittest discover` 1876 run, OK;
  billing contract 110 passed; boot smoke rc 0; lint clean.
- **Existing ordering tests still pass:** Run 75 Today landing and precedence, P2-5 shared
  link, Run 83 boundary-before-redirect, Run 83B Scanner layout.
- **GitHub Actions run 36295060100:** all five jobs green.

## Manual Verification Matrix

| Account | First page after login | Correct on first render? | Scanner visit required? | Basis |
|---|---|---|---|---|
| Basic | Today | PASS | NO | Automated (real `app.py`) |
| Pro | Today | PASS | NO | Automated |
| Premium | Today | PASS | NO | Automated |
| Admin | Today | PASS | NO | Automated |
| A → logout → B (6 pairs) | Today | PASS | NO | Automated |
| Any, live | Today | **UNVERIFIED** | — | Needs a live sign-in (below) |

**Live check:**
- After the fix deployed, the browser pane's saved session came up signed out: "Your
  session has expired". The database is reachable (billing `/health`: DB reachable). Only
  logout deletes session records, most likely during the owner's own manual multi-account
  testing in that shared pane.
- B4 doesn't touch session storage: cookie restore runs inside `auth_ui()`, before the
  reordered code.
- **The owner should verify:** sign in to each account type → the Today sidebar shows the
  right plan and paid features are unlocked on the first screen → reload the page → still
  signed in.

**Pre-existing observation (not B4, P2):** a stale session cookie shows "Your session has
expired" on every visit instead of once, so the cookie clear doesn't appear to persist.
Worth a small follow-up.

## Release Gate

| Blocker | Status |
|---|---|
| B1 billing auth | FIXED (live) |
| B2 account isolation | FIXED |
| B3 Scanner routing | FIXED (live) |
| **B4 post-login context** | **FIXED** (automated; live sign-in check pending) |
| Restore token | SINGLE-USE |
| Dependencies | PATCHED (P1-21 done) |

**Run 84's verdict stands: CONDITIONAL GO.** Add one item to its manual conditions (C3):
*first render after sign-in shows the correct plan and features for each account type,
and a reload stays signed in.*
