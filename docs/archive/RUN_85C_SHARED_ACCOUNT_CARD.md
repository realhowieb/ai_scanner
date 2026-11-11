# Run 85C — Shared Account / Plan Sidebar Card

## Summary

One component, `ui/account_card.py`, now renders the account block on **every**
authenticated page. It shows:
- name;
- plan (via Run 85B's `plan_label`);
- a plan summary;
- the upgrade CTA;
- "Compare all plans";
- "Log out".

Nothing else renders a sidebar plan or upgrade card, so changing pages can't change what
it shows.

## Before

| Surface | Implementation | Showed |
|---|---|---|
| Scanner (`app.py`) | `ui/app_user_profile.render_account_sidebar` + `ui/app_runtime.render_sidebar_upgrade_card` | name, plan, "You're on Free" card with Upgrade to Pro (Free only), logout |
| Every sub-page (Today, Brief, Day Trader, Stock, How HSF works, My Stocks, Journal, Settings, Billing, Alerts, Kalshi) | `ui/nav._render_identity` | name, plan, logout only (no summary or CTA) |
| Phone "☰ Menu" | `_render_identity(key_suffix="mobile")` | name, plan, logout |

## After

- **`render_account_card(key_suffix, compact)`:** the only renderer.
  - `ui/nav._render_identity` (every sub-page sidebar; the phone menu in `compact` form)
    and Scanner's `render_account_sidebar` both call it.
  - Scanner keeps only its admin-only tier-debug lines.
- **`render_sidebar_upgrade_card`:** no longer called, but kept defined because `app.py`
  imports it at startup. Removing it could break a stale-module redeploy.
- **Data source:** the card reads only the account context `app.py` resolves after sign-in
  (`username`, `display_name`, `tier_key`, `is_admin`; see B4). No database or
  subscription lookup, verified by a test that makes those lookups raise.
- **Copy** (`plan_card`), built from the canonical alert limits and the Run 73 and Run 85
  packaging:

| Plan | Headline | Summary | Primary CTA | Supporting copy | Compare |
|---|---|---|---|---|---|
| Free (internal `basic`) | You're on Free | Discover today's market opportunities with HSF Score and basic Stock Intelligence. | Upgrade to Pro | Pro adds monitoring and investigation: 5 alerts, email delivery, interactive results, exports and history. | ✓ |
| Pro | You're on Pro | Monitor and investigate: 5 alerts with email delivery, interactive results and CSV export, Nasdaq and combined scans, earnings and scan history. | Upgrade to Premium | Premium adds research and workflow: 25 alerts, AI scan summaries and results chat, setup notes, Early Breakout research, full-market custom scans and paper trading. | ✓ |
| Premium | You're on Premium | Research and workflow: 25 alerts, AI scan summaries and results chat, setup notes, Early Breakout research, full-market custom scans and paper trading. | — | — | ✓ |
| Admin | Admin access | All features are enabled for testing and operations. | — | — | — |

- **Upgrade CTA:** the existing inline Stripe checkout button (`ui/app_runtime._upgrade_button`).
  With Run 83 it sends the account token, and it keeps the email-verification gate.
- **Navigation highlight:** unaffected. The card sits above the page links, which are
  unchanged.

## Responsive Check

Checked with a temporary local preview (a real sidebar and phone menu with a seeded
session), deleted afterwards:

| Width | Result |
|---|---|
| Desktop, 300 px sidebar, all four tiers | PASS: copy exactly as above; no overflowing elements; navigation directly below |
| 390 px phone (sidebar collapsed, ☰ menu) | PASS: compact card (plan, headline, CTA, Compare all plans, Log out) in a 320 px popover within the 390 px viewport; no overflow; navigation below |

The phone menu uses the same component with `compact=True`; there is no second
implementation.

## Tests

**New: `tests/test_run85c_account_card.py` (11 tests):**
- **Copy per tier:** label, headline, CTA and compare; the Free copy is exactly the
  approved text.
- **Claims backed by entitlements:** every capability named for Pro/Premium is enabled in
  that tier's entitlements, and alert counts equal `ALERT_LIMIT_BY_TIER`. No "Basic",
  and no "AI setup notes".
- **No database access:** the card makes no DB, tier or user lookup.
- **Single implementation:** every sidebar path uses the shared card, and no other code
  renders a sidebar plan or "You're on …" card.
- **Cross-page render:** Today, Market Brief, Day Trader, Stock Intelligence, How HSF
  works, My Stocks, Journal, Settings, Billing, Alerts and Kalshi BTC, for **Free, Pro,
  Premium and Admin**, each render the identical card, with the right plan, CTA buttons
  and "Compare all plans" presence.
- **Scanner:** the real `app.py` renders the same card.

**Updated tests (same intent):**
- `tests/test_run80_mobile_state.py`: the logout key now lives in the shared card.
- `tests/test_run85b_plan_labels.py`: the sidebars use the card, which uses `plan_label`.

**Results** (production-parity venv, outbound network blocked):

| Suite | Collected | Passed | Failed | Skipped |
|---|---:|---:|---:|---:|
| Full, `-X dev -W always` | 1963 | 1925 | 0 | 38 (FastAPI; covered by the billing job) |
| Lightweight CI env | 1963 | 1827 | 0 | 136 |
| `unittest discover` | 1920 run | OK | 0 | 38 |
| Billing contract | 111 | 110 | 0 | 1 |

**Named suites, all passing:**
- 85C: 11
- 85B: 11
- B4: 3
- B2: 10
- B3: 11
- Run 85 AI claims: 22
- Run 80 mobile: 8

**Other checks:** dependency audit clean; boot smoke rc 0; lint clean.

**Frozen core:** no changes to scoring, ranking, research, scheduling, `app.py`,
entitlements (`ui/app_session.py`), pricing or Stripe.

## Status

**SHARED ACCOUNT CARD: DONE.** A live signed-in check is part of launch check P1-29.
