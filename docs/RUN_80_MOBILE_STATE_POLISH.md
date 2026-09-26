# Run 80 — Mobile UX & Persistent User-State Polish

## Executive Summary

Run 80 makes phone navigation singular and predictable, gives important phone
controls practical touch targets, removes the nested Custom scan popover, and
keeps the existing first-run tour usable at narrow widths. Scanner Cards/Table
and lens choices now use the existing encrypted browser-preference store, so
ordinary reruns, page navigation, refreshes, and later browser sessions do not
unexpectedly reset an explicit choice.

The state audit also found an account-isolation defect: logout cleared core
identity fields but could leave tier, entitlement, watchlist, Stock Intelligence,
alert-prefill, and post-login navigation values in Streamlit session state. Those
keys are now cleared centrally while non-account browser conveniences remain.

No score, ranking, scan, model, scheduled job, research, or billing behavior was
changed.

## Mobile Audit

Browser checks used explicit 375, 390, 430, 768, and 1440 pixel viewports. The
local environment lacked `COOKIE_PASSWORD` and a configured authenticated user,
so authenticated surfaces are not represented as visually verified.

| Surface | 375 | 390 | 430 | Tablet | Desktop | Notes |
|---|---|---|---|---|---|---|
| Signed-out landing/auth | PASS | PASS | PASS | PASS | PASS | No horizontal overflow at any tested width |
| Today | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Source/regression tests pass; authenticated browser unavailable |
| Scanner/cards/table | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Responsive default and persistence covered by tests |
| Stock Intelligence | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Deep-link precedence covered by tests |
| Custom scan | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Nested popover removed; one entry point retained |
| My Stocks | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Account cleanup/isolation covered by tests |
| Billing | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Run 78 billing tests pass |
| Onboarding | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | One tour retained; two-column actions plus full-width Skip |

## Navigation

At widths up to 640px, the `☰ Menu` is the primary navigation. It now includes
account identity, plan, and logout, and the duplicate Streamlit sidebar and its
collapsed control are hidden only inside the phone media query. Desktop sidebar
behavior is unchanged. Important buttons, page links, tabs, and radio options
receive a 44px minimum target at phone widths.

## State Inventory

| State | Storage | Lifetime | Account-specific? | Reason |
|---|---|---|---|---|
| Scanner Cards/Table | encrypted browser preference + session mirror | browser | No | convenience that should survive navigation/refresh |
| Scanner lenses | encrypted browser preference + session mirror | browser | No | investigation context; saved-screen overrides remain explicit |
| Saved screens | encrypted browser preference | browser | No | existing intended architecture |
| Tour completion | encrypted browser preference | browser | No | one onboarding system; avoids unexpected restart |
| Tour step/open panels/scan progress | Streamlit session | session | No | transient interaction state |
| New since last visit | existing browser/session mechanism | browser/session | No | existing visit comparison semantics |
| Watchlists/My Stocks | database + account-scoped session composition | account | Yes | meaningful cross-device user data |
| Alerts/preferences | database | account | Yes | meaningful cross-device user data |
| Subscription/entitlements | server-derived session state | session/account | Yes | authorization data, cleared at logout |
| Selected ticker | Streamlit session | session | Potentially | navigation handoff, cleared at logout |
| Shared ticker target | URL/query + post-login session destination | URL/session | No | explicit destination outranks Today default |
| Custom scan controls | Streamlit widgets/existing saved defaults | session/existing account policy | Mixed | unchanged; no extra entry point |
| Return-after-login destination | Streamlit session | authentication transition | No | consumed before Today default; cleared at logout |

## Persistence Policy

- **Account persistent:** watchlists, alerts, alert preferences, subscription and
  other existing server-owned user records.
- **Browser persistent:** Cards/Table, lenses, saved screens, tour completion,
  and existing visit history. These contain no account data.
- **Session-only:** open panels, tour step, temporary scan progress/messages,
  selected Stock Intelligence handoff, and loaded account-derived values.
- **URL-driven:** shared ticker destinations. An explicit destination continues
  to take priority over the default Today landing.

## Authentication Safety

`ui.app_session.ACCOUNT_SESSION_KEYS` is the canonical logout purge list. It
includes identity, tier/entitlements, active watchlist composition, scan results,
Stock Intelligence handoff, alert prefill, Today landing, and pending post-login
destination. Browser-only preferences are intentionally preserved. Tests model
User A logout and verify that account values disappear while browser preferences
remain available to the browser, without becoming account data.

## Streamlit State Findings

- Scanner view and lenses were widget-only session values. They now load once
  from browser preferences and save only on explicit widget changes.
- An unset mobile view defaults to Cards; any explicit saved choice wins.
- Pending lens navigation and saved-screen application still override current
  lenses and are persisted after application.
- Custom scan no longer nests its filters in a phone-hostile popover.
- No duplicate widget keys, rerun loops, or assignment-after-widget errors were
  found in the touched paths.

## Performance Verification

The new convenience state uses the existing cookie-backed browser-preference
helper and is loaded once per Streamlit session. It adds no database reads.
Run 71's second-rerun regression tests pass. Account data continues through the
existing cached/batched paths.

## Files Changed

- `app.py`
- `ui/app_session.py`
- `ui/auth.py`
- `ui/chrome.py`
- `ui/discover.py`
- `ui/nav.py`
- `ui/tour.py`
- `tests/test_run62_trust_layer.py`
- `tests/test_run67_68_leftovers.py`
- `tests/test_run80_mobile_state.py`
- `docs/RUN_80_MOBILE_STATE_POLISH.md`

## Tests

- Focused Run 80/navigation/state suite: **49 passed, 23 subtests passed**.
- Run 79/watchlist/alert/billing regression selection: **43 passed, 26 skipped**.
- Complete repository suite: **1,838 passed, 26 skipped, 172 subtests passed**.
- Ruff on modified Python files: **passed**.
- `scripts/streamlit_smoke.py --timeout 60`: **passed**.
- Browser widths: signed-out landing/auth **PASS** at 375, 390, 430, 768,
  and 1440 pixels; authenticated surfaces **UNVERIFIED** as documented above.

## Frozen-Core Verification

Run 80 did not modify HSF Score, Breakout Score, ranking formulas, scanner
engine, `run_breakout_scan`, scheduled scanning, model inference/training,
research capture, Gate U, maturation, Autonomous Research Mode, certified
Run 56/58/61 behavior, or billing semantics. Runs 70–79 regression coverage
remains green, including P1-22 billing protection.

## Backlog Status

- **P2-13: REMAINS OPEN** — implementation and source/regression validation are
  complete, but authenticated Today/Scanner/Stock/My Stocks visual verification
  at the required phone widths could not be performed in this environment.
- **P2-14: DONE** — persistence policy is explicit; Scanner preferences survive
  expected reruns/navigation; shared-link precedence remains; logout and
  cross-user account-state isolation have regression coverage.
- **P2-11, P2-12, P2-15: remain DONE.**
- **P1-22: remains protected in CI.**

The next highest-value work remains Run 81 (P2-17 + P2-18 dead-code cleanup and
test/resource-warning hygiene), after authenticated Run 80 mobile QA closes the
remaining P2-13 verification item. After Run 81, perform a fresh release-candidate
audit against current `main` rather than adding backlog features.
