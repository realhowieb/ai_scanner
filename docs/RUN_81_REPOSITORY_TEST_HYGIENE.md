# Run 81 — Repository & Test Hygiene (P2-17 + P2-18)

Base: `main` @ `4dcf880` (Run 80). Branch: `run81-hygiene`. Conservative cleanup only:
no product, scoring, ranking, scanning, research, pricing, billing or entitlement change.

## Executive Summary

- **Dead code removed:** 10 unreferenced modules and 5 unreferenced functions
  (≈720 lines), plus 18 genuinely unused imports in product modules.
- **Test hygiene:** every first-party resource leak the suite exposes is fixed. Four were
  unclosed-resource leaks (a SQLite connection, a file handle, three temp directories)
  and one was a real network call from a unit test. The unit test downloaded live prices
  from yfinance in CI; that call left yfinance's cache connections unclosed, which is
  what CI's Python 3.13 run reported.
- **Warnings:** the only remaining ResourceWarning comes from Streamlit's test harness
  (third-party). No first-party ResourceWarning remains.
- The suite now runs **with outbound network blocked** and makes zero connection attempts.
- **Frozen core untouched.** Test count unchanged; no test removed.

The repository is healthier. There is less dead surface to keep compatible, and CI no
longer depends on Yahoo being reachable.

## Dead-Code Audit

Method:
- A reference sweep over every first-party `.py`, workflow `.yml`, `.toml`, `.sh` and
  `.md` file.
- It covers dotted imports, relative imports (`from .x import`), path strings,
  string/dynamic imports (`importlib`, lazy `__getattr__` maps in `ui/__init__.py` and
  `db/__init__.py`), and the `python -m` / `python scripts/…` entry points in all
  GitHub Actions workflows.
- It also checks every exported symbol by name.
- Test references were then checked by **content**, not just by name, because several
  source-contract tests assert literal source text.

### Modules

| Candidate | Decision | Evidence | Action |
|---|---|---|---|
| `ui/earnings_admin.py` | SAFE TO REMOVE | No import, string, workflow or test reference. Earnings admin now lives in `ui/admin_results_tab.py` (`_render_earnings_refresh`). | Deleted |
| `ui/diagnostics.py` | SAFE TO REMOVE | No references. Superseded by `ui/scan_diagnostics.py`, `ui/provider_health.py`, `ui/system_health_view.py`. | Deleted |
| `data/market.py` | SAFE TO REMOVE | `fallback_universe` / `fetch_market_heat` referenced nowhere. | Deleted |
| `auth/tiering_utils.py` | SAFE TO REMOVE | `derive_tier_flags` referenced nowhere. Entitlements come from `ui/app_session.py`. | Deleted |
| `data/yf_adapters.py` | SAFE TO REMOVE | Exception classes referenced nowhere. | Deleted |
| `utils/chunking.py` | SAFE TO REMOVE | Unreferenced duplicate of `data/price_utils.chunks` (the one in use). | Deleted |
| `utils/export.py` | SAFE TO REMOVE | `download_zip_button` referenced nowhere. | Deleted |
| `utils/market_time.py` | SAFE TO REMOVE | Referenced nowhere. | Deleted |
| `utils/retry.py` | SAFE TO REMOVE | `with_backoff` referenced nowhere. | Deleted |
| `utils/state.py` | SAFE TO REMOVE | Empty file. | Deleted |
| `scan/session.py` | KEEP (UNCERTAIN) | Unreferenced headless runners, but it lives in the frozen scan package and looks like a hand-run entry point. | Kept |
| `db/universe.py` | KEEP (UNCERTAIN) | Unreferenced `reeval_inactive_symbols`, but it is universe/DB code (frozen area) and may be run by hand. | Kept |
| `jobs/daily_snapshot.py` | KEEP (UNCERTAIN) | No workflow calls it; it is a scheduled-job module and "scheduled jobs" are frozen. | Kept |
| `scripts/*.py` (diag_*, run48/54, scoreboard, …) | KEEP | Hand-run diagnostics and research CLIs; not meant to be imported. | Kept |
| `analytics/intelligence_view.py` | KEEP | Used by the showcase contract tests; touched by Run 79. | Kept |
| `ui/heat_strip.py` (module) | KEEP | `pill_color` has a contract test (tests/test_whats_new.py). | Renderer removed only |
| `ui/pages.py`, `ui/pages_main.py` | KEEP | Kept on purpose in Run 75 (facade + scheduler compatibility tests). | Kept |

### Functions (Run 75 leftovers and superseded helpers)

| Candidate | Decision | Evidence | Action |
|---|---|---|---|
| `auth/tiering.render_pricing_section` | SAFE TO REMOVE | Half-finished pricing stub (placeholder comments, no Stripe wiring); unreferenced. Superseded by `ui/pricing.py` (Run 72). | Removed |
| `ui/app_runtime.render_onboarding_hint` | SAFE TO REMOVE | Old "Quick start" panel, unreferenced since Run 75. The only test mention asserts it is *absent* from `app.py`. | Removed |
| `ui/heat_strip.render_watchlist_heat` | SAFE TO REMOVE | Unreferenced renderer; the module's tested color helper stays. | Removed |
| `ui/admin_results_tab._ensure_earnings_table` | SAFE TO REMOVE | Unreferenced private helper. | Removed |
| `ui/market_brief._render_breadth_sectors` | SAFE TO REMOVE | Unreferenced private helper. | Removed |
| `ui/three_step_scanner._step_label` | KEEP | Unreferenced, but `test_cleanup_source_checks` asserts its icon map in source. | First removed, then restored |
| `billing_service/main.py` route handlers | KEEP | Registered by FastAPI decorators (not by name). | Kept |
| `db/user_settings.get/set_onboarding_dismissed` | KEEP | Unreferenced after the hint was removed, but they wrap an existing `user_settings` column; DB schema is out of scope. | Kept |
| Tested helpers without production callers (`onboarding.user_is_first_run`, `product_copy.find_prohibited_claims`, `showcase.honest_display`, `discover.apply_lens`, …) | KEEP | Exercised by tests / used as guards. | Kept |

Run 75 verification:
- No duplicate Custom-scan entry points remain.
- No legacy watchlist UI, old history path or superseded navigation remains.
- The only orientation leftover was `render_onboarding_hint`, now removed.

### Imports

`pyproject.toml` ignores F401, so CI never checked unused imports. With F401 enabled there
were 45 in the repo. Removed:

| File | Removed | Why safe |
|---|---|---|
| `ui/results.py` | `re`; private aliases `_disable_yfinance_for_session`, `_is_yahoo_crumb_error`, `_warn_yfinance_disabled_once` | No use, no patch target, no re-import |
| `ui/scans.py` | `datetime`, `timedelta`, `ZoneInfo`, `pandas` | No use; not patch targets |
| `ui/watchlists.py` | `list_watchlists`, `get_default_watchlist_id` (from `db.watchlists`) | Not used there; tests patch `db.watchlists.*`, not `ui.watchlists.*` |
| `ui/scan_providers.py` | `streamlit` | No use; not a patch target |
| `ui/header.py` | `datetime.date` | No use |
| `ui/charts.py`, `ui/hsf_calibration_report.py` | `typing` names | No use |
| `billing_service/main.py` | `json` | No use; billing contract suite passes |
| `db/password_reset.py` | `os` | No use |
| 3 test files | stdlib names | No use |

Kept on purpose (documented):
- `ui/results.get_results_df`: re-exported to `app.py`. Now marked `# noqa: F401`.
- `ui/scans.render_three_step_scanner`: re-exported to `app.py`. Marked `# noqa: F401`.
- `ui/scans.render_data_provider_diagnostics`: required by an extraction contract test.
- `app.py` fallback `USERS_DB`: mirrors the primary import set in the `except` branch.
- Package `__init__` re-exports (`auth`, `data`, `scheduler`).
- Frozen-core files (`scan/engine.py`, `db/runs.py`, `data/prices.py`, `data/fetch.py`,
  `analytics/*`): not touched in a cleanup run.
- Test imports that exist to prove a module imports (`scheduler.morning_digest`,
  `analytics.autonomy_certification`).

## Deleted Surface

| File | Lines | Why it was safe |
|---|---:|---|
| `ui/earnings_admin.py` | 205 | Unreferenced; replaced by the admin results tab |
| `ui/diagnostics.py` | 141 | Unreferenced; replaced by scan/provider/system health views |
| `data/market.py` | 65 | Unreferenced |
| `auth/tiering_utils.py` | 47 | Unreferenced |
| `utils/market_time.py` | 23 | Unreferenced |
| `utils/export.py` | 23 | Unreferenced |
| `utils/retry.py` | 21 | Unreferenced |
| `utils/chunking.py` | 16 | Unreferenced duplicate |
| `data/yf_adapters.py` | 14 | Unreferenced |
| `utils/state.py` | 0 | Empty |

Totals vs `origin/main`:
- 31 files changed, +28 / −725 lines.
- 10 modules removed, 5 functions removed, 18 production imports removed.
- **No tests removed**. The test-file changes are the hygiene fixes plus 6 unused
  stdlib imports.
- The subtest count moved from 172 to 170 because `test_run62_trust_layer` runs one
  subtest per `ui/*.py` file, and two UI modules were deleted.

## ResourceWarning Audit

How it was run:
- The full suite ran with `-X dev -W always` (plus `tracemalloc` for the baseline).
- A scratch tracer recorded the creation stack and test ID of every `sqlite3.Connection`
  that was garbage-collected unclosed. On Python 3.12, SQLite leaks do not warn; they do
  on 3.13, which CI uses.
- A second scratch guard blocked outbound network and reported any socket or DNS
  attempt, with the test ID.
- The Python 3.13 CI log of main's latest smoke run was also read.

### Before (main @ 4dcf880)

| # | Warning | Source | Seen in |
|---|---|---|---|
| 1 | `ResourceWarning: unclosed file 'ui/showcase.py'` | `tests/test_showcase_mode.py:84`: `open(...).read()` | local + CI |
| 2 | `ResourceWarning: unclosed database in sqlite3.Connection` (×2) | yfinance's peewee cache DBs, opened because `test_day_trader_watchlist::test_chart_receives_pick` rendered a **real** chart and downloaded NVDA prices | CI (3.13) |
| 3 | `ResourceWarning: unclosed database` (attributed to `test_maturation_e2e.py:26`) | `test_already_matured_horizons_not_refetched` opened `sqlite3.connect(":memory:")` and never closed it | CI (3.13) |
| 4 | Three temp dirs leaked on disk (no warning) | `tempfile.mkdtemp()` in `test_us_market_universe.py` and 2× `test_scan_reliability.py` | local disk |
| 5 | `ResourceWarning: Implicitly cleaning up TemporaryDirectory` | `streamlit/testing/v1/app_test.py:95`: module-level `TMP_DIR = tempfile.TemporaryDirectory()` | local dev mode |

### After

| # | Status | Fix |
|---|---|---|
| 1 | **FIXED** | `Path(...).read_text()` |
| 2 | **FIXED** | The test stubs `ui.charts` via `sys.modules` (no live download; also runs without streamlit/plotly) |
| 3 | **FIXED** | `self.addCleanup(conn.close)` |
| 4 | **FIXED** | `self.addCleanup(shutil.rmtree, d, True)` |
| 5 | **EXPECTED / THIRD-PARTY** | Streamlit's AppTest harness creates it at import and relies on interpreter shutdown. No production risk (test harness only). Not suppressed. |

After the fixes:
- The tracer reports **zero** unclosed first-party SQLite connections.
- The network guard reports **zero** connection or DNS attempts across the whole suite.

### Other warnings (classified, not ResourceWarning)

| Warning | Source | Class | Production risk |
|---|---|---|---|
| `DeprecationWarning: on_event is deprecated, use lifespan` (28×) | `billing_service/main.py:26` `@app.on_event("startup")` | **UNRESOLVED (first-party)** | Low. FastAPI still supports it. Moving to `lifespan` changes billing-service startup, which is out of scope for a no-behavior-change run. |
| `DeprecationWarning` (28×) | `fastapi/applications.py` (the same deprecation, raised inside FastAPI) | EXPECTED / THIRD-PARTY | None |
| `DeprecationWarning: anyio.abc.BlockingPortal alias` | `starlette/testclient.py` | EXPECTED / THIRD-PARTY | None (test client) |
| `st.cache is deprecated` log | `streamlit_authenticator`, `streamlit_cookies_manager` | EXPECTED / THIRD-PARTY | None; already filtered in `ui/app_boot.py` |

### Database, file and HTTP resource review

- **SQLite / Postgres helpers:**
  - `db/hsf_observations` and similar helpers close only connections they open
    (`opened` flag); caller-owned connections stay open by design.
  - `db/engine.get_sqlite_conn` / `get_neon_conn` hand ownership to callers.
  - `analytics/autonomy_certification` closes its in-memory DBs in `finally`.
  - No accidental leaks were found in production code.
- **Intentional long-lived resources (not leaks):**
  - Run 71's `st.cache_data` per-user caches hold data, not connections.
  - Streamlit's cache and yfinance's own cache DBs are process-lifetime by design.
- **Files:** every first-party `open()` outside tests is already context-managed. The
  single exception was the test fixed in #1.
- **HTTP:** no first-party session leaks found. Production networking is unchanged. The
  only issue was a unit test reaching the real network (#2).

## Test Results

| Suite | Collected | Passed | Failed | Skipped | Subtests | Warnings |
|---|---:|---:|---:|---:|---:|---|
| Full (`.venv`, py3.12, `-X dev -W always`, network blocked) | 1864 | 1838 | 0 | 26 | 170 passed | 1 (Streamlit AppTest temp dir, third-party) |
| Lightweight CI env (`requirements-dev` only, no streamlit) | 1864 | 1764 | 0 | 100 | 170 passed | 0 |
| `unittest discover -s tests` (the CI dependency job's runner) | 1821 run | OK | 0 | 26 | — | — |
| Billing contract (isolated `requirements-billing-test.txt` venv) | 86 | 85 | 0 | 1 | 6 passed | 57 deprecation (classified above) |

Baseline before changes:
- Full: 1838 passed / 26 skipped / 172 subtests.
- Lightweight: 1764 passed / 100 skipped.

The 26 full-suite skips are tests needing optional pieces not installed locally (e.g.
FastAPI/Stripe billing tests, which run in their own CI job). The 100 lightweight skips
are the streamlit-dependent UI tests, by design.

## CI Verification

Reproduced locally, step by step, from `.github/workflows/smoke.yml`:

- `python -m compileall -q .`: OK.
- `ruff check . --select E9,F,I --ignore F401,F811`: All checks passed.
- Import smoke: `import config`, `import scheduler.cron_runner`,
  `_load_universe('SP500')`: OK.
- Unit smoke tests in the lightweight env: 1764 passed, 0 failed. These include the
  Run 77 (5), Run 79 (8) and Run 80 (8) tests, all executed, none skipped.
- `unittest discover -s tests`: OK.
- `scripts/deployment_doctor.py`: all `[OK]`. `scripts/streamlit_smoke.py --timeout 60`:
  exit 0.
- **Billing contract job (P1-22 / Run 78):** `test_billing_service.py`,
  `test_run72_pricing.py`, `test_run73_tier_differentiation.py`,
  `test_tier_enforcement.py`, `test_app_session.py` ran with the fake Stripe environment:
  85 passed, 1 skipped, no network.

CI configuration was not changed.

**GitHub Actions on dev @ b044903 (Smoke Checks run 36278574952):**

| Job | Result |
|---|---|
| `smoke` | success |
| `core-dependency-import-smoke` | success (`Ran 1821 tests`) |
| `full-dependency-import-smoke` | success |
| `billing-contract` | success (85 passed, 1 skipped) |
| `dependency-audit` | failure, same as before: the open P1-21 dependency pins |

- ResourceWarning lines in the Python 3.13 CI log: **8 on main @ 4dcf880 → 0**.
- The remaining `ReadTimeout` log line comes from `test_price_utils`, which simulates the
  timeout with a mock; it is not a network call.

Entry-point imports checked:
- UI: app, Today, Scanner (`ui.scans`, `ui.results`), Stock Intelligence, My Stocks
  (`ui.watchlists`), nav, auth, pricing, landing.
- Scheduler and research: `scheduler.cron_runner`, `scheduler.jobs`, `scan.engine`,
  `scan.pre_post`, `scripts.mature_observations`, `scripts.autonomous_recovery`,
  `scripts.autonomy_certification`, `scripts.forward_evidence_readiness`,
  `analytics.btc_outcome_logger`.
- All pages parse.
- `billing_service.main` imports in its own environment. It needs `psycopg2`, which the
  app venv does not install; that is unchanged.

## Frozen-Core Verification

`git diff origin/main` over the frozen and research paths:
- Paths checked: `scan/`, `research/`, `analytics/`, `scheduler/`, `jobs/`, `.github/`,
  `db/`, `scripts/`, `ml_prebreakout.py`, `market_data.py`, `data/us_market_universe.py`,
  `data/prices.py`, `data/fetch.py`, `ui/app_session.py`.
- Result: the only change is one unused `import os` removed from `db/password_reset.py`
  (password-reset helper, not research).

No change to:
- HSF Score, Breakout Score, ranking, `run_breakout_scan`, the scan engine.
- Scheduled scans or workflows.
- Model inference or training, Gate U, research capture, maturation, cohorts.
- Autonomous Research Mode, Run 56/58/61 behavior.
- Entitlements, pricing or billing logic.

## Backlog Status

- **P2-17 = DONE.** The confirmed-dead modules, functions and imports are removed with
  per-item evidence. The remaining unreferenced items are kept deliberately
  (frozen-adjacent or contract-tested) and are listed above.
- **P2-18 = DONE.** The showcase ResourceWarning and every other first-party leak the
  suite exposes (on 3.12 and CI's 3.13) are fixed. The one remaining ResourceWarning is
  Streamlit's own and is documented.

Previous work intact:
- P1-22: the billing CI job passes locally with CI's exact command.
- P2-11, P2-12, P2-15: the Run 79 tests pass.
- P2-14: the Run 80 tests pass.
- P2-13: still open, as recorded in Run 80. Its code tests pass; the signed-in phone-width
  visual check is still outstanding. Run 81 did not change that.

## Remaining Technical Debt

1. `billing_service/main.py` uses the deprecated `@app.on_event("startup")`. Migrating to
   FastAPI `lifespan` is a small, separate billing change with its own verification.
2. `scan/session.py`, `db/universe.py`, `jobs/daily_snapshot.py` are unreferenced but sit
   in frozen or scheduled areas. Remove them only in a run that is allowed to touch those
   areas.
3. 29 unused imports remain (F401 view), in frozen-core files, package `__init__` re-exports, the `app.py` fallback and tests that import modules on purpose
   (F401 stays ignored in CI).
4. The dependency security pins (P1-21 / Run 74) are still open. They are unrelated to this
   run and unchanged by it.
