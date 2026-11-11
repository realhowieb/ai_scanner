# Run 85B — Canonical Plan Naming Across All Pages

## Final Status

**PLAN LABEL CONSISTENCY: FIXED.** An internal `basic` account is shown as **Free** on
every page. Pro → Pro, Premium → Premium, admin → Admin.

## Root Cause

The internal tier and the customer-facing plan name were never separated in one place.
Each page turned the tier into text itself:

| Where | Before | Showed |
|---|---|---|
| `ui/nav.py` `_render_identity`: sidebar on every sub-page (Today, Market Brief, Day Trader, Stock Intelligence, How HSF Works, My Stocks, Journal, Settings, Billing) | `tier_key.title()` | **Basic** |
| `ui/app_user_profile.py` `render_account_sidebar`: Scanner sidebar | the tier object's `name` from the tier config | **Free** |
| `pages/settings.py` Account block | `.upper()`, special-casing BASIC→FREE; ignored admin | **FREE** |
| `auth/tiering.py` `require_min_tier`: Premium-feature gate on Scanner | `current_key.capitalize()` | "not available on your current plan (**`Basic`**)" |
| `app.py` plan-change warning | `.upper()` | "from **PRO** to **BASIC**" |
| `ui/auth.py` post-checkout toast | `.title()` | "Plan upgraded to **Pro**" (correct only by coincidence) |
| `pages/billing.py` `_current_plan_label` | its own mapping | Free (already correct) |
| `ui/pricing.py` `TIER_NAMES` | its own mapping | Free (already correct) |

That is why one account (realtest123) showed "Plan: Basic" on Stock Intelligence and
"Plan: Free" / "You're on Free" on Scanner.

## Internal vs Customer-Facing Representation

- **Internal tier identity, unchanged:** `basic`, `pro`, `premium`, `admin`, used in the
  users table, the Stripe price mapping (`_price_to_plan` → pro/premium, cancellation →
  `basic`), `TIER_ORDER`, `FEATURE_MIN_TIER`, `compute_entitlements`, alert limits and
  every `== "basic"` comparison.
  - No data migration and no Stripe change.
  - No comparison was edited; a test asserts entitlements are still keyed on `basic`.
- **Customer-facing label:** `ui/plan_labels.plan_label(tier, is_admin=False)` (new
  module) maps `basic`/`free` → **Free**, `pro` → **Pro**, `premium` → **Premium**,
  `admin` (tier or role) → **Admin**. Unknown or missing → Free.
  - It accepts a key string, a display name or a tier object.
  - It's for display only: never used for gating, and nothing compares against its
    output.

## Files Changed

| File | Change |
|---|---|
| `ui/plan_labels.py` (new) | `PLAN_LABELS`, `plan_label()`. A new module, so it can't be stale during a redeploy. |
| `ui/nav.py` | Sidebar plan → `plan_label(tier_key or tier, is_admin=…)` |
| `ui/app_user_profile.py` | Scanner sidebar plan → `plan_label(tier_key or tier, is_admin=…)` |
| `pages/settings.py` | Account "Plan" → `plan_label(…)` (now also shows Admin, and title case) |
| `pages/billing.py` | `_current_plan_label` delegates to `plan_label` |
| `auth/tiering.py` | Premium-feature gate message uses `plan_label` (with a guarded import fallback) |
| `app.py` | Plan-change warning uses `plan_label`. Two no-op lines removed (`_ = st.session_state.get("authentication_status") is True` and its comment) to stay within the 840-line budget: 838. |
| `ui/auth.py` | Post-checkout toast uses `plan_label`. Sign-up bullet "Basic Stock Intelligence…" → "HSF Score and basic Stock Intelligence…" (lowercase, so it can't read as a plan). |
| `ui/pricing.py` | `TIER_NAMES` derived from `PLAN_LABELS` (same values) |
| `tests/test_run85b_plan_labels.py` (new) | 11 tests |
| `tests/test_run73_tier_differentiation.py` | The truthful-Free-copy assertion now checks "basic Stock Intelligence" (lowercase); same intent |

## Test Results

**New: `tests/test_run85b_plan_labels.py` (11 tests):**
- **Mapping:** all keys, case, tier objects, missing/unknown → Free; admin role wins; no
  label is "Basic"; pricing uses the same names; entitlements still keyed on internal
  `basic`.
- **Source checks:** every plan display uses the helper; no raw tier `.title()` /
  `.upper()` / `.capitalize()` in customer-facing strings (including the tier gate); no
  "You're on / Plan: / Upgrade from Basic" or "Basic plan" copy.
- **Cross-page render** (real page scripts headlessly, for **each of Basic, Pro, Premium
  and Admin**): the sidebar **Plan** on Today, Market Brief, Day Trader, Stock
  Intelligence, How HSF Works, My Stocks, Journal, Settings and Billing is the single
  expected label. For internal `basic`, no customer-visible text on any page contains
  "Basic".
- **Scanner** (real `app.py`, each tier): the sidebar plan matches, and for internal
  `basic` no visible text contains "Basic" (this caught the Premium-feature gate message).
- **On the old code, 4 of the 6 display tests fail** (both cross-page tests and 2 source
  tests).

**Regression** (production-parity venv, locked deps, outbound network blocked):

| Suite | Collected | Passed | Failed | Skipped | xfail | Warnings | Duration |
|---|---:|---:|---:|---:|---:|---|---:|
| Full, `-X dev -W always` | 1952 | 1914 | 0 | 38 (all FastAPI; covered by the billing job) | 0 | 1 (Streamlit AppTest temp dir, third-party) | 145 s |
| Lightweight CI env | 1952 | 1819 | 0 | 133 | 0 | 0 | 17.3 s |
| `unittest discover` | 1909 run | OK | 0 | 38 | — | — | 105 s |
| Billing-contract job (B1) | 111 | 110 | 0 | 1 | 0 | 81 (FastAPI/Starlette deprecations) | 1.7 s |

**Named suites, all passing:**
- Run 85B: 11
- B4 post-login: 3
- B2 isolation: 10
- B3 Scanner: 11
- Run 85 AI claims: 22
- tier enforcement / app session / Run 73: 50
- frozen core / research (certification incl. Gate U, maturation, observation, signal
  evidence): 95

**Other checks:**
- Dependency audit: no known vulnerabilities.
- Boot smoke: rc 0.
- Lint: clean.

## Cross-Page Verification

| Page | Internal `basic` | Pro | Premium | Admin | Basis |
|---|---|---|---|---|---|
| Today | Free | Pro | Premium | Admin | Automated render |
| Market Brief | Free | Pro | Premium | Admin | Automated render |
| Scanner | Free | Pro | Premium | Admin | Automated render (real `app.py`) |
| Day Trader | Free | Pro | Premium | Admin | Automated render |
| Stock Intelligence | Free | Pro | Premium | Admin | Automated render |
| How HSF Works | Free | Pro | Premium | Admin | Automated render |
| My Stocks | Free | Pro | Premium | Admin | Automated render |
| Journal | Free | Pro | Premium | Admin | Automated render |
| Settings | Free | Pro | Premium | Admin | Automated render |
| Billing | Free | Pro | Premium | Admin | Automated render |

- **B4:** the first screen after sign-in already has the resolved `tier_key`, so the
  first render shows the right label with no navigation needed (B4 tests pass).
- **Live check:** UNVERIFIED. The browser pane has no signed-in session. Owner check:
  sign in as realtest123 and open each page; the sidebar should say "Plan: Free"
  everywhere (tracked in launch check P1-29).

## Frozen-Core Verification

`git diff` over `scan/`, `research/`, `analytics/`, `scheduler/`, `jobs/`, `data/`,
`db/`, `scripts/`, `ml_prebreakout.py`, `market_data.py`, `ui/headline_score.py` and the
entitlement code `ui/app_session.py`: **no changes.** No change to pricing, Stripe
products, subscriptions, entitlements, scoring, ranking or research.

**Next:** Run 86, the final live launch check.
