# Run 83B — Scanner Tier/State Routing Hotfix

## Executive Summary

**B3 is FIXED.** Clicking Scanner showed "My Watchlists" as the main content for a
Premium account. The subscription tier did not cause this.

The Scanner page drew its canonical results **last**. The results slot sat at the top of
the page but was filled at the end of `main()`, after:
- the My Watchlists panel, which fetches a live quote for every watched ticker;
- the whole Custom scan section.

For any account with a **populated watchlist**, on any tier, the page therefore showed
"## Scanner" followed directly by "My Watchlists" and its prices until the quotes
returned. If anything after the panel failed, the results never appeared. The Basic test
account worked because nothing slowed or broke that stretch; most likely it had no
watchlist (not verified, since I can't see those accounts).

**The fix:**
- The canonical results render first, before the watchlist panel and Custom scan.
- A watchlist failure is contained to its own section.
- The Premium 3-step scan reruns once so its results appear at the top.

Nothing changed in tiers, data, scoring or the Run 83 security fixes.

## Observed Behavior

| Account | Path | Result |
|---|---|---|
| Basic | Login → Today → Scanner | Canonical Scanner (correct) |
| Premium | Login → Scanner | Sidebar and heading say Scanner; main content is **My Watchlists** with symbols and prices instead of opportunities |

## Root Cause

**Where the Scanner page was built** (`app.py` `main()`, before the fix):

```
st.markdown("## Scanner")
results_slot = st.container()          # placed at the top …
st.markdown("---")
render_watchlists_panel(username)      # ### My Watchlists + live-quote card wall
… Custom scan: filters, earnings, render_scan_controls, render_three_step_scanner …
with results_slot:                     # … but filled LAST
    default_results(get_results_df()) → HSF ranking → render_results_tabs
```

**Why a watchlist triggered it:**
- Streamlit draws elements as the script runs.
- `render_watchlists_panel` → `_render_watchlist_tiles` → `market_data.build_day_trader_metrics`
  fetches live quotes for every ticker on the active list. That is a network call, before
  the results fill.
- The panel also had no error guard, so any exception inside it (or later in the Custom
  scan section) ended the run with the results slot still empty.

**Measured with the real `app.py` headless** (Streamlit AppTest, quote fetch slowed to 3 s,
populated watchlist):

| | Before fix | After fix |
|---|---|---|
| Basic | quotes 0.00 → 3.01 s, **results at 3.05 s** | **results at 0.00 s**, quotes 0.04 → 3.05 s |
| Premium | quotes 0.00 → 3.01 s, **results at 3.03 s** | **results at 0.00 s**, quotes 0.39 → 3.39 s |

With a failing watchlist backend, the old code rendered **no results**: the page showed
"UI section failed" after the heading.

**Tier was incidental:**
- With a populated watchlist, Basic reproduced exactly like Premium.
- With an empty watchlist, both tiers were fine before the fix. Those were the only two
  new tests that passed on the old code.
- Premium does add extra work after the panel (the 3-step scanner, the AI summary), which
  widens the window, but that work wasn't the cause.
- No entitlement check routes Scanner to watchlists.

Checked and ruled out:
- **Widget-key collisions:** the 3-step keys (`scan_market`, `scan_profile`, …) are used
  nowhere else.
- **Navigation/data key coupling:** no key doubles as "which page" and "which dataset";
  Scanner's dataset is always `default_results(get_results_df())`.
- **Persisted lens or view:** browser prefs hold the view mode, lenses and saved screens,
  and none of them selects a watchlist dataset.
- **"Watchlist only" filter:** opt-in, defaults off, not persisted.
- **Stale identity:** the Run 83 boundary clears account state, and cross-account runs are
  tested below.

## Fix

The smallest change that makes the Scanner independent of anything drawn below it:

1. **`app.py`:** the `with results_slot:` block now runs immediately after the
   `## Scanner` heading, before the watchlist panel, Custom scan, earnings controls, scan
   controls and 3-step scanner.
   - The results code itself is unchanged: `default_results(get_results_df())` → earnings
     prep → why → HSF Score → `rank_hsf_opportunities` → discover bar → `render_results_tabs`.
   - The panel call is wrapped so a failure shows a scoped "your watchlists" message
     instead of breaking the page.
   - `app.py` is 837 lines (budget 840).
2. **`ui/watchlists.ensure_active_watchlist_state(user_id)` (new):** because results now
   render before the panel, this seeds `active_watchlist_id`/`_tickers` from the user's
   cached default list, **only when the session has none**.
   - This keeps the "★ Watching" badges and the opt-in "Watchlist only" filter working on
     the first render.
   - It never replaces or filters the dataset.
   - It uses the Run 71 per-user cache (same keys as the panel), so it adds no extra DB
     reads.
3. **`ui/three_step_scanner.py`:**
   - Before the fix, a Premium 3-step scan wrote `results_df` and relied on the late fill.
   - It now sets a flash message and calls `st.rerun()`, so the new results appear at the
     top.
   - The completion message is shown after the rerun.
   - Manual scans already rerun via `force_results_refresh`.
4. **`ui/app_session.ACCOUNT_SESSION_KEYS`:** adds `_three_step_flash` (the user's own
   scan message) so it can't cross accounts.

## State Ownership

| State | Owner | Meaning | Notes |
|---|---|---|---|
| Scanner dataset | `results_df` (own scan) else latest `cron`/`US_MARKET` run | What Scanner shows | Never derived from watchlists |
| `active_watchlist_id`, `active_watchlist_tickers` | My Watchlists panel (seeded by `ensure_active_watchlist_state`) | Which list is active, for badges and explicit filters | Account state, cleared on logout and at the Run 83 identity boundary |
| `active_watchlist_quote_rows` | Watchlist card wall | Quotes for the card wall, Market Brief and Alerts strips | Account state (Run 83) |
| `<prefix>_watchlist_only` | Results tab checkbox | Explicit "Watchlist only" filter | Session widget, default off, not persisted |
| `hsf_results_view`, `hsf_lens`, saved screens | Browser prefs (Run 80) | UI preferences | Never select a dataset |
| `scan_market` / `scan_strategy` / `scan_profile` | 3-step scanner | Custom-scan settings | Unique keys |
| `_three_step_flash` | 3-step scanner | One-shot status after rerun | Account state |

## Tier Verification

- **Same results for every tier:** the real `app.py` with Basic, Pro and Premium
  entitlements and a populated watchlist renders `## Scanner` → `### HSF Opportunities`
  first, with all 3 market rows.
- **Same ranking for every tier:** the HSF-ranked ticker order is identical across tiers.
- **Tier differences stay inside the Scanner:**
  - Pro/Premium: scan history, CSV/interactive table;
  - Premium: AI Scan Summary and results chat.
- `compute_entitlements`, `FEATURE_MIN_TIER` and pricing are unchanged.

## Cross-Account Verification

- **Tested:** Premium → Basic, Basic → Premium and Premium → Premium, with the session
  holding Alice's `ALICEPICK` watchlist and owner marker.
- **Result:** after Bob signs in and opens Scanner, the results are canonical and the
  active watchlist is Bob's own `BOBPICK`. Alice's never appears.
- **Run 83 still passes:** `test_run83_account_isolation` (10), `test_run83_billing_auth`
  (12, billing job), `test_run83_restore_token` (13). The account-state clearing was only
  extended (one key).

## Tests

**New: `tests/test_run83b_scanner_state.py`** (11 tests; 2 source-contract tests run
everywhere, 9 run the real `app.py` where Streamlit is installed, i.e. the CI dependency
job):
- every tier with a populated watchlist opens the canonical Scanner, with an identical HSF
  ranking;
- empty watchlist (Basic, Premium);
- slow watchlist quotes can't delay results (results rendered before the quote fetch);
- failing watchlist backend can't blank results (scoped message instead);
- after a My Stocks selection of a non-default list plus a Stock Intelligence hand-off,
  Scanner is still canonical;
- repeated navigation and reruns;
- persisted view preference keeps the dataset;
- "★ Watching" state seeded from the current user's default list;
- cross-account transitions in both tier directions;
- source contract: results fill before the watchlist, Custom scan and 3-step sections, and
  the 3-step scan reruns after persisting.

**The new tests catch the defect:** against the pre-fix `app.py`, `ui/watchlists.py` and
`ui/three_step_scanner.py`, **9 of 11 fail**. They include every tier, slow, failing,
after My Stocks, reruns and cross-account. Only the empty-watchlist and
persisted-preference cases pass, which matches the production observation.

**Updated test (intentional contract change):** `tests/test_p06_scanner_declutter.py`.
- The old `test_results_fill_after_the_scan_tools_run` asserted the ordering that caused
  B3.
- Its purpose ("a scan started below still shows its results") is kept, now via the rerun
  paths.
- It is replaced by `test_results_fill_before_the_scan_tools_and_scans_rerun_to_show`,
  which asserts the new order and that both scan paths rerun.
- No other test was changed or removed.

**Results** (production-parity venv: locked deps, Streamlit 1.54.0, tornado 6.5.8):

| Suite | Collected | Passed | Failed | Skipped | xfail | Warnings | Duration |
|---|---:|---:|---:|---:|---:|---|---:|
| Full, `-X dev -W always`, outbound network blocked | 1912 | 1874 | 0 | 38 (all "fastapi not installed"; covered by the billing job) | 0 | 1 (Streamlit AppTest temp dir, third-party) | 168 s |
| Lightweight CI env | 1912 | 1786 | 0 | 126 | 0 | 0 | 17.3 s |
| `unittest discover -s tests` | 1869 run | OK | 0 | 38 | — | — | 104 s |
| Billing-contract job (CI command) | 111 | 110 | 0 | 1 | 0 | 81 (FastAPI/Starlette deprecations) | 1.7 s |
| `ruff check .` | — | All checks passed | — | — | — | — | — |

Named suites run together in one process, 104 passed:
- Run 83B Scanner (11) and Run 83 account isolation (10);
- state/persistence `test_run80_mobile_state`;
- HSF hierarchy `test_run79_information_hierarchy`;
- startup `test_boot_stale_module`;
- data consistency `test_run70_data_consistency`, performance `test_run71_performance`;
- frozen core `test_autonomy_certification` (incl. Gate U);
- layout `test_p06_scanner_declutter`.

Billing: 110 passed (above). Boot smoke: rc 0.

**Harness note:** an earlier run of these suites together showed 2 Run 70 failures. They
were caused by the new test's module stubs leaking into later tests in the same process.
`run_app` now restores every patched attribute and clears the Streamlit cache after each
run, and they pass.

## Manual Verification

Browser verification of signed-in accounts was **not** performed. There is no test
account available to this session, and the local app has no database to sign in against.
Every row below was exercised headlessly by running the real `app.py` under Streamlit
AppTest, with stubbed market and watchlist data.

| Account | Starting page/state | Action | Expected | Headless (real app.py) | Live browser |
|---|---|---|---|---|---|
| Basic | Today | Scanner | Canonical Scanner | PASS | UNVERIFIED |
| Pro | Today | Scanner | Canonical Scanner | PASS | UNVERIFIED |
| Premium | Today | Scanner | Canonical Scanner | PASS | UNVERIFIED |
| Premium | My Stocks (non-default list selected) | Scanner | Canonical Scanner | PASS | UNVERIFIED |
| Premium | Stock Intelligence hand-off state | Scanner | Canonical Scanner | PASS | UNVERIFIED |
| Premium | Populated watchlist (slow and failing quotes) | Scanner | Canonical Scanner | PASS | UNVERIFIED |
| Premium | Fresh login (after Today landing) | Scanner | Canonical Scanner | PASS | UNVERIFIED |
| User A → User B | Identity transition (both tier directions) | Scanner | Current-user Scanner | PASS | UNVERIFIED |

Run 84 should repeat the Premium-with-watchlist path on the live app.

## Frozen-Core Verification

`git diff` over `scan/`, `research/`, `analytics/`, `scheduler/`, `jobs/`, `data/`,
`db/`, `scripts/`, `ml_prebreakout.py`, `market_data.py`, `ui/headline_score.py`,
`ui/results_intelligence.py`, `ui/market_default.py`, `ui/market_scans.py`: **no
changes.**

No change to:
- HSF Score, Breakout Score, ranking, `run_breakout_scan`;
- scheduled scans, models, research capture, cohorts, maturation;
- Gate U, Autonomous Research Mode, Run 56/58/61 behaviour.

The canonical dataset selection (`default_results`/latest `cron`/`US_MARKET`) and the HSF
ranking pipeline run the same code as before, only earlier in the page.

## Release Gate

| Finding | Status |
|---|---|
| **B3 Scanner routing/state** | **FIXED** |
| B1 billing auth | **FIXED** (Run 83; tests still pass) |
| B2 account isolation | **FIXED** (Run 83; tests still pass, extended by one key) |
| Stripe restore token | **SINGLE-USE** (Run 83) |
| Tornado | **PATCHED** (6.5.8, Run 83) |

All known release blockers are fixed. **Next: Run 84, the final release-gate recheck**,
run on the deployed app after promotion. Include:
- the live Premium-with-watchlist → Scanner check;
- a signed-in phone walkthrough;
- the Run 82 manual checklist;
- autonomy re-certification after the tornado lock change.
