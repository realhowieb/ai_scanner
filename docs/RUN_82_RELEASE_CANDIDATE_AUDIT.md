# Run 82 — Release Candidate Audit

## Final Verdict

# NO-GO

Two verified launch blockers make taking real payments unsafe today:

- **B1:** anyone who knows a paying customer's email can open that customer's Stripe
  billing portal.
- **B2:** one account's watchlist carries over to the next account signed in on the same
  browser tab.

Both are small, contained fixes. Nothing else found in this audit blocks a controlled
launch. Once B1 and B2 are fixed and the launch conditions in §Manual Pre-Launch
Checklist are met, this release candidate should move to **CONDITIONAL GO**.

## Executive Summary

HSF is much closer to launch than it was at Run 69.
- **Run 69 findings:** both data-consistency P0s (D1, D2) and 13 of the 15 P1s are closed
  in current code. They were verified by reading the code and running tests, not taken
  from later run notes.
- **Core loop works:** the product loop (landing → plans → sign-up → Today → Scanner →
  Stock Intelligence → My Stocks → alerts → billing) is coherent.
- **One source of truth:** Today, the Scanner, the trust banner and Stock Intelligence all
  read the same canonical scheduled market run (`cron` · `US_MARKET`).
- **Pricing:** it is generated from the real entitlement map, so it cannot drift.
- **Claims:** the public copy is honest. It says HSF Score is "not a probability of profit
  … not a prediction or guarantee", and the Track Record no longer headlines a
  cherry-picked number.
- **Tests and CI:** 1,840 tests pass with outbound network blocked, and every GitHub
  Actions job passes on this commit except the non-blocking dependency audit.
- **Frozen core:** untouched since the Run 61 certification, apart from two display-label
  renames.

This audit found two new problems that earlier runs did not look for. Both are
verified, not theoretical.

1. **B1 · Unauthenticated billing portal (security).**
   - `POST /create-portal-session` and `POST /create-checkout-session` on the public
     billing service accept any email address. They return a live Stripe billing-portal
     URL for that customer, with no proof that the caller is that user.
   - The repository is **public**, and the service URL is committed as a default
     (`config.py:102`, `pages/billing.py:14`).
   - Anyone who knows a subscriber's email can therefore cancel their subscription,
     change their card, and read their invoices and billing address.
2. **B2 · Cross-account watchlist carryover (privacy).**
   - Logout does not clear `active_watchlist_quote_rows` (nor two related keys).
   - When user B signs in on the same browser tab after user A signs out, Market Brief's
     **"📋 Your watchlist today"** shows **A's tickers and prices**.
   - B's "📧 Email me this brief" would email A's list to B.
   - Reproduced headlessly with the app's own logout-clearing function and Market Brief
     reader.

There is also a small set of P1 launch conditions:
- the tornado/cryptography security pins;
- the checkout "restore" token, which is a reusable 14-day login;
- confirming that the Anthropic key and model behind Premium's AI features actually work
  in production;
- one real Stripe test-mode round trip.

## Release Candidate

| Item | Value |
|---|---|
| Repository | `realhowieb/ai_scanner` (**public**) |
| main HEAD | `c01598ac68024e4da00b8aaaa58f644dfd7d98a0`, "Fix startup ImportError from a stale module after a Streamlit Cloud redeploy" (2026-09-26 16:20 PT) |
| dev | Same SHA (main = dev) |
| Audit time | 2026-09-27 00:13 UTC (Saturday, US market closed) |
| Local runtime | Python 3.12.6 (CI: 3.13.15) |
| Locked versions (production) | streamlit 1.54.0, pandas 2.3.3, numpy 2.5.1, xgboost 3.3.0, pyarrow 23.0.1, tornado 6.5.7, cryptography 49.0.0 |
| Billing service | FastAPI 0.115.6, stripe 11.1.1, psycopg2-binary 2.9.10 (`billing_service/requirements.txt`), on Render |
| App deploy | Streamlit Cloud from `dev`, `requirements.txt` → `-c requirements.lock` + core/ml/extended groups; `.streamlit/config.toml` `toolbarMode="minimal"`, custom nav |
| Scheduling | GitHub Actions on `main`, dispatched by cron-job.org; `scheduled-scans.yml` |
| CI | `.github/workflows/smoke.yml`: `smoke`, `core-dependency-import-smoke`, `full-dependency-import-smoke`, `billing-contract`, `dependency-audit` (non-blocking) |

Environment note: the local venv has drifted from the lock (streamlit 1.49.1,
pyarrow 25.0.0). CI and production use the lock. One ad-hoc timing harness segfaulted
inside pyarrow 25 on this machine. The same flow passes under pytest, and the CI
environment is unaffected. This is recorded as a local-environment artifact, not a
product defect.

## Run 69 Reconciliation

Each finding below was re-verified against current code (file references given), not
taken from the run that claimed to fix it.

| Finding | Orig. | Original problem | Current status | Evidence | Addressed by | Residual risk |
|---|---|---|---|---|---|---|
| D1 | P0 | Stock Intelligence ignored the scan the user came from (Today and cards) | **CLOSED** | `ui/today.py:52` and `ui/result_cards.py:84` call `stock_handoff.open_in_stock_intelligence(..., scan_df=…)`. `pages/stock.py:62-72` uses `session_context`, which is ticker-guarded, then falls back to `latest_market_context` | Run 70 | Low. The ticker guard stops a stale opportunity from another ticker from being shown |
| D2 | P0 | Market Brief and header read "latest run of any user" | **CLOSED** | `ui/market_brief.py:85` and `ui/app_user_profile.py:154` use `ui.market_scans.safe_recent_runs` (`cron`/`US_MARKET` only) | Run 70 | Low. Remaining unfiltered `list_runs` callers are admin-only (`provider_health`), user-scoped (`results_tabs` passes `username`) or label-scoped to scheduler labels users cannot produce (`day_trader` premarket/postmarket) |
| D3 | P1 | Row order vs headline score | **CLOSED** | `ui/headline_score.py` sorts by HSF Score, then BreakoutScore, then ticker, then source order (stable merge sort) | Run 76 | None |
| S1 | P1 | Breakout alert threshold on a hidden scale | **CLOSED** | `ui/alerts.py:264-279` shows the scale copy, a legend, a "Breakout Score ≥ threshold" help text and an observed-scale expander; `tests/test_run77_breakout_alert_scale.py` | Run 77 | None |
| ST1 | P1 | No way back to the market view after your own scan | **CLOSED** | `ui/market_default.render_back_to_market` wired at `app.py:764`; `test_run70…test_back_to_latest_market_scan_after_own_scan` | Run 70 | None |
| T1 | P1 | "AI-powered rankings" sign-up claim | **CLOSED** | No match in `ui/`, `pages/`, `app.py` | Runs 72–73 | None |
| T2 | P1 | "Diagnostics / Retrain" sold as Premium | **CLOSED** | Pricing rows are derived; `can_diagnostics`/`can_admin_panel` are admin-only and never sold (checked programmatically) | Run 72 | None |
| MZ1 | P1 | Pricing copy didn't match the product | **CLOSED** | `ui/pricing.py` builds the table, cards and highlights from `FEATURE_MIN_TIER`; live /billing matches | Runs 72–73 | Low. The two per-plan captions in `pages/billing.py:238,247` are still hand-written; they match today |
| MZ2 | P1 | Weak paid differentiation | **CLOSED** (decision made) | Free = Discover, Pro = Monitor & Investigate, Premium = Research & Workflow (`ui/pricing.TAGLINES`) | Run 73 | Commercial risk only |
| MZ3 | P1 | Pricing hidden before sign-up | **CLOSED** | Live landing shows three plan cards; signed-out /billing shows the full table (checked at 375/390/430/768 px, no overflow) | Run 72 | None |
| PF1 | P1 | About 7 uncached DB reads per Scanner click | **CLOSED** | `tests/test_run71_performance.py`: the second rerun makes 0 additional watchlist/tier reads, and a write invalidates on the next rerun (passes at HEAD) | Run 71 | None |
| PF2 | P1 | Stock Intelligence history uncached | **CLOSED** | `ui/stock_intelligence._history_cached` (`st.cache_data`), tested | Run 71 | None |
| U1 | P1 | Three onboarding systems stacked | **CLOSED** | `app.py` renders only `render_tour("scanner")`; the old "Quick start" hint was removed in Run 81 | Runs 75, 81 | None |
| U2 | P1 | Three scan entry points | **CLOSED** | One collapsed `st.expander("Custom scan")` (`app.py:631`) contains the filters, scan controls and 3-step scanner | Run 75 | None |
| U3 | P1 | Scanner, not Today, is the landing page | **CLOSED** | `should_land_on_today` once per session (`app.py:443`); shared-link destination takes precedence (`app.py:429-431`) | Runs 75, 80 | None |
| SEC1 | P1 | Known CVEs in the lock | **OPEN** | Lock still pins tornado 6.5.7 (3 advisories, fix 6.5.8), cryptography 49.0.0 (1, fix 50.0.0), soupsieve 2.8.4 (2), GitPython 3.1.51 (21) | Planned Run 74 | See §Dependencies |
| CI1 | P1 | Billing/Stripe tests never ran | **CLOSED** | `billing-contract` job: 85 passed, 1 skipped on HEAD | Run 78 | None |

**Summary:** 2/2 P0 closed; 14/15 P1 closed (13 fixed in code, plus MZ2, a packaging
decision that has been made); 1 P1 open (SEC1).

## Remaining Backlog

The tracker's open items match the code:

| Item | Status | Launch blocker? |
|---|---|---|
| P1-21 / Run 74: dependency CVEs + re-certification | todo | **Partly.** The tornado patch bump is a launch condition (L1). The rest is acceptable temporary risk. |
| P2-13: signed-in phone QA | doing | No. A manual check before launch (checklist). |
| P2-19: move custom scans off the page request | blocked (frozen core) | No. Mitigated by the market-scan default. |
| P2-7: email recap | blocked | No |
| P2-8 / Run 83 (evidence page) | blocked on Run 56 | No, and it must stay blocked. Publishing evidence early would be a trust risk. |

Stale items: none found. Two findings from this audit (B1, B2) and four P1 conditions
are **not in the tracker yet**.

## User Journey

**Signed out (VERIFIED on the live site, 375/390/430/768 px and desktop).**
- **The first screen says what HSF does:** "HSF continuously scans thousands of tradable
  U.S. stocks and organizes noteworthy setups so you can focus your research."
- **Supporting sections:** "What HSF does" (Scan / Rank / Explain / Track), an example
  result clearly labelled "Illustrative names and values, not live data or
  recommendations", trust points, three plan cards ($0 / $19/mo / $39/mo) and a disclaimer.
- **Understood within 10–15 seconds:** what HSF does, what it does not promise (the
  disclaimer), what Free provides, and why Pro and Premium exist. The taglines and
  "Everything in Free/Pro, plus" lists do this.
- **Weak point:** the landing page shows HSF Scores but never says what the score
  **represents**. That is only on /methodology, which states clearly: "an
  opportunity-ranking score from 0 to 100 … not a probability of profit … not a
  prediction or guarantee." (P2, F9)
- **Unsupported claim:** Pro carries a **"MOST POPULAR"** badge with no paying users yet.
  That is a small claim the product cannot yet back. (P2, F8)
- **Signed-out /billing:** plans only, a sign-up call to action, and no misleading "isn't
  linked to an email" or "currently on Basic" text. There is no signed-in nav.

**Authentication (code and tests; live signed-in UNVERIFIED because no test account was
available to this audit).**
- **Passwords:** bcrypt hashes (`ui/auth.py:408`), with legacy plaintext auto-migrated
  on login.
- **Rate limiting:** failed logins are rate-limited per username (`db/users.py:647`,
  enforced at `ui/auth.py:513`).
- **Sessions:** random UUIDs with a 14-day expiry (`ui/auth_sessions.py`).
- **Landing and deep links:** Today is the landing page once per session. A shared-ticker
  link survives sign-in and wins over the Today redirect.
- **Logout:** clears `ACCOUNT_SESSION_KEYS`, except the watchlist keys in B2.
- **Checkout return link:** `?rt=` restores a login after Stripe. See P1 finding F3.

## Data Integrity

| Check | Result | Evidence |
|---|---|---|
| Same "latest scan" everywhere | PASS | Today (`safe_recent_runs()[0]`), Scanner default (`pick_market_run` → `trust_banner.latest_market_scan`), trust banner and Market Brief all select the newest `cron`/`US_MARKET` run |
| Same HSF Score everywhere | PASS | All paths use `ui.results_intelligence.consolidate_scanner_results` → `score_breakdown`; the Stock Intelligence hand-off passes the canonical opportunity; Run 70 end-to-end tests pass |
| Partial runs | PASS | `scheduler/cron_runner.py:484` refuses to save a run with too few rows from a large universe |
| Wrong-ticker hand-off | PASS | `stock_handoff.session_context` discards an opportunity or row whose ticker differs (`test_session_context_rejects_stale_values`); cards and Today pass the ticker explicitly, not an index |

Live cross-screen comparison of real tickers is **UNVERIFIED**: it requires signing in.
The code path is shared, so a contradiction would need a shared-code bug. None was found.

## Today

- **Data source:** canonical scheduled run.
- **Freshness:** a trust banner with the scan age and market session (holiday-aware).
- **Degraded states are handled distinctly:**
  - unavailable or empty scan: "The latest market scan is unavailable right now…";
  - scan present but nothing qualifies: `no_qualifying`, explicitly tested
    (`test_today_distinguishes_weak_market_from_empty_scan`), so a quiet market does not
    look broken;
  - a section failure shows a scoped error (`safe_errors.show_error`), not a crash.
- **Actions:** open the Scanner or Market Brief; watchlist context included.
- **Status:** PASS (code and tests); live view UNVERIFIED (sign-in).

## Scanner

- **Default view:** the latest scheduled market scan, with a caption such as "Latest
  full-market scan · Fri … (N h ago) · N ranked setups".
- **Ordering:** HSF Score is the headline and default order.
- **Custom scan:** optional, inside one collapsed expander.
- **Way back:** a "Back to latest market scan" control after your own scan.
- **Lenses and saved screens:** AND semantics, tested (P2 work).
- **Card and table views:** built from the same frame.
- **Model outputs:** secondary outputs are renamed ("5D outcome probability",
  "PreBreakout setup probability") and placed under Model details.
- **Status:** PASS (code and tests).

## Stock Intelligence

- **Order on the page:** HSF Score, then explanation, then context, then Model details
  (Run 79 hierarchy tests pass).
- **Entry points:** direct entry, Scanner card, Today, shared `?ticker=` link (consumed
  once) and the post-sign-in deep link are all wired.
- **Wrong ticker:** not possible through the tested hand-off.
- **History:** cached per ticker.
- **Status:** PASS.

## My Stocks

- **Isolation:** every DB call is keyed by `user_id`; the Run 71 caches are keyed by
  (user, per-user data version), so another user never matches the cache.
- **Legacy code:** the old watchlist UI was removed in Runs 75 and 81.
- **Exception, B2:** the session-level quote cache survives logout and reaches
  Market Brief, which is why this area is a **FAIL (P0)**.

## Alerts

- **Limits:** 1 / 5 / 25 (Free / Pro / Premium) in `ALERT_LIMIT_BY_TIER`, matching
  pricing. The UI cap is in `ui/alerts.py:238`.
- **Enforced when alerts run:** the scheduled runner re-resolves the tier from the DB
  and fires only the newest N alerts the plan allows (`scheduler/alert_runner.py:395-417`).
- **Email:** Pro and up only. A tier-lookup failure falls back to Basic, so errors never
  grant extra alerts or email.
- **Threshold:** on a visible, explained scale (Run 77).
- **Status:** PASS (code and CI tests).
- **Live email delivery UNVERIFIED.** It needs the production Resend/SMTP secrets.

## Billing / Stripe

**CODE/CI VERIFIED:**
- **Webhooks:**
  - rejected when `STRIPE_WEBHOOK_SECRET` is missing (500) or the signature is bad (400);
  - deduplicated by event ID (`stripe_processed_events`).
- **Checkout completion:** maps the subscription price to a plan and refuses an
  undeterminable plan (500, so Stripe retries).
- **Subscription updated:** Pro ↔ Premium via price mapping; a scheduled cancellation
  keeps the tier until period end; an immediate cancel downgrades to Basic.
- **Subscription deleted:** downgrades to Basic.
- **Unknown or forged tier:** rejected.
  - Unknown tier values are coerced to Basic (`_set_user_plan_by_email`).
  - The app reads the tier only from the DB.
  - `?checkout=success` only triggers a tier poll; it grants nothing.
- **CI:** the `billing-contract` job runs the mocked contract suite (85 passed, 1 skipped,
  no network).

**FAIL (P0), B1:**
- **Portal endpoint:** `create_portal_session` (`billing_service/main.py:395`) takes
  `{"email": …}` and returns `stripe.billing_portal.Session.create(customer=…).url`.
- **Checkout endpoint:** `create_checkout_session` does the same for an email that already
  has an active subscription (`"mode": "portal"`, `tests/test_billing_service.py:277`).
- **No authentication on either:** no header, token or shared-secret check anywhere in
  `billing_service/main.py`.
- **The service is findable:** its URL is the committed default (`config.py:102`) in a
  public repository.

**Also:** `GET /debug/status` is public. It shows which secrets are set, DB reachability
and alert-worker errors (P2, F10).

**LIVE STRIPE E2E UNVERIFIED:**
- A real test-mode checkout → webhook → tier → portal → cancel round trip is required
  before launch (checklist).

## Entitlements

Current matrix (from `ui/app_session.FEATURE_MIN_TIER`, computed):

| Capability | Free | Pro | Premium | Admin |
|---|:-:|:-:|:-:|:-:|
| Latest market ranking, HSF Score, Today, basic Stock Intelligence, lenses, saved screens, My Stocks | ✅ | ✅ | ✅ | ✅ |
| Own S&P 500 scans (`can_scan_sp500`) | ✅ | ✅ | ✅ | ✅ |
| Alerts | 1 | 5 | 25 | 25 |
| Email alerts, CSV/interactive table, Nasdaq/combined scans, premarket/after-hours/unusual volume, earnings, scan history, track record | ❌ | ✅ | ✅ | ✅ |
| AI notes/summaries/chat, Early Breakout, full-universe custom scans, paper trading | ❌ | ❌ | ✅ | ✅ |
| Admin panel, diagnostics | ❌ | ❌ | ❌ | ✅ |

- **No drift:** landing, /billing and upgrade prompts are generated from this map. The
  two hand-written plan captions (`pages/billing.py:238,247`) match it.
- **Free keeps the core product:** core discovery, HSF Score, latest opportunities and
  basic Stock Intelligence.
- **Admin-only features are never marketed.**
- **Caveat (P1 condition L3):** Premium's AI features only work if
  `ANTHROPIC_API_KEY` is set in Streamlit secrets. It is **not** listed in
  `.streamlit/secrets.toml.example`, and the default model ID is `claude-opus-4-8`
  (`config.py:107`). Without a valid key and model, Premium customers see "AI is not
  configured" for a headline paid feature.
- **Paper trading:** uses each user's own encrypted Alpaca paper keys (`db/secret_box`),
  so it also depends on an encryption secret.

## Mobile

| Surface | 375 | 390 | 430 | 768 | Desktop | Basis |
|---|---|---|---|---|---|---|
| Landing | PASS | PASS | PASS | PASS | PASS | Live, no horizontal overflow; 17 controls under 32 px tall at 375 (tabs/links, P2 F11) |
| Billing (signed out) | UNVERIFIED | PASS | UNVERIFIED | UNVERIFIED | PASS | Live |
| Methodology | UNVERIFIED | PASS | UNVERIFIED | UNVERIFIED | PASS | Live |
| Today, Scanner, cards, Stock Intelligence, My Stocks, Custom scan, alerts, onboarding, nav (signed in) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Needs a signed-in session. Run 80 source/regression tests pass (8/8). |

## State / Persistence

- **Browser-level preferences:** card/table view, lenses, saved screens and tour
  completion live in the encrypted cookie jar (`ui/browser_prefs`). They are not account
  data (Run 80 policy), and that is correct.
- **Account state:** cleared on logout via `ACCOUNT_SESSION_KEYS`, **except**
  `active_watchlist_quote_rows`, `_watchlist_prior_rows` and `_loaded_user_settings`
  (B2).
- **Reproduction** (with the app's own functions):

```
after logout, survived: ['_loaded_user_settings', '_watchlist_prior_rows', 'active_watchlist_quote_rows']
Bob's Market Brief watchlist rows: [{'ticker': 'ALICEPICK', …}] | DB queried for Bob: False
```

- `_loaded_user_settings` affects only the legacy `ui/pages_main` path. The next user
  gets defaults rather than their saved filters (no data exposure).
- `_watchlist_prior_rows` feeds the "changed since last view" comparison on the watchlist
  intelligence feed. It leaks A's per-symbol states into B's diff baseline (minor, fixed by
  the same change).

## Performance

- **Run 71 intact at HEAD:**
  - `test_run71_performance`: three full Scanner runs (first load plus two reruns) take
    0.84 s with stubbed data;
  - reruns make **0** extra watchlist/tier reads;
  - a write is visible on the next rerun.
- **Other paths:** secondary panels stay lazy (`ui/lazy_panel`); Stock Intelligence
  history is cached; the returning-user summary is not on the Scanner.
- **Not measured:** live Streamlit Cloud timings (cold start, first Today paint) were not
  timed in this audit. No user-visible performance blocker was found in code.

## Deployment Reliability

| Condition | Behaviour | Class |
|---|---|---|
| Stale first-party module after redeploy | `app.py:81` catches `(KeyError, ImportError)`, drops the stale module only when `err.name` is a first-party package (ui/auth/db/scan/data/utils/analytics), reruns at most 3 times, then re-raises; `tests/test_boot_stale_module.py` reproduces the production error on the old code | **recoverable** |
| Genuine ImportError (missing dependency) | Same bounded 3 retries, then the real error surfaces | fatal-but-clear |
| Auth module import failure | `_startup_problem` message, `st.stop()` | fatal-but-clear |
| Missing DB | `list_runs`/`safe_*` return empty. Today shows "latest market scan is unavailable". Sessions can't be created, so sign-in fails with a message | graceful / fatal-but-clear |
| Missing market snapshot | Today and Scanner "unavailable" states (Run 70) | graceful |
| Missing Stripe env in billing service | `_require_env` → HTTP 500 with the missing names; `/health` → 503 | fatal-but-clear |
| Missing `ANTHROPIC_API_KEY` | "AI is not configured" message | graceful (but see L3) |

- **CI boot checks:** the app boots headless in CI (`scripts/streamlit_smoke.py`), and
  `deployment_doctor` passes.
- **Startup is not dangerous:** no infinite loop or unbounded retry path was found.

## Scheduled Market Pipeline

- **Universe:** refreshed weekly (`refresh-universe.yml`, last run 2026-09-20 success).
- **Schedule:** scans run on trading days. The last 12 `scheduled-scans.yml` runs all
  succeeded, the most recent on Fri 2026-09-25 23:55 UTC. None ran on Saturday, as
  expected.
- **Weekends and holidays:** the scheduler checks the Alpaca market calendar and skips
  non-trading days (`scheduler/cron_runner.py:224-266`, fail-safe). The trust banner
  treats a weekend or holiday snapshot as current, and labels the session holiday-aware
  (`ui/trust_banner.market_session_label`).
- **Early close:** scans at fixed ET times still run after an early close and read closing
  data. The banner's timestamp and age keep that honest. Not specially labelled (P2
  observation; no action required).
- **Stale data:**
  - health older than 36 h cannot vouch for results;
  - "Latest market scan unavailable" is a named state;
  - every result shows its scan time.
- **Automation health:** maturation, system health and autonomous recovery ran
  successfully on 2026-09-26.
- **Scheduler:** not modified.

## Research / Trust

- **Track Record:**
  - no longer headlines the best ranking × horizon ("Every combination is shown; none is
    singled out", `ui/track_record.py:66-90`);
  - small samples are flagged;
  - the badge needs n ≥ 150 and says "past performance is not…".
  - `_best_summary` remains but is not rendered to customers.
- **Methodology:** clearly separates HSF Score (ranking) from model outputs (research
  estimates) and makes no predictive claims.
- **Evidence page:** correctly still blocked until Run 56 forward evidence is ready.
- **Model labels:** "AI Confidence" is renamed to "5D outcome probability" everywhere
  customers see it.
- **Unsupported claims:** none found beyond "MOST POPULAR" (F8).

## Security / Privacy

| Check | Result |
|---|---|
| Secrets in repo (working tree + full history of a public repo) | **PASS.** Pattern scan over all commits: the only Stripe/DB matches are placeholders (`sk_live_...`, `whsec_...`, example DSNs) and short test fixtures. `.streamlit/secrets.toml` and `.env` are git-ignored. |
| Billing endpoints | **FAIL (P0), B1** |
| Account data leakage | **FAIL (P0), B2** |
| Session tokens | UUIDv4, 14-day expiry. **P1 (F3):** the Stripe success/portal-return URLs carry `rt=<session_id>`, which is a full reusable 14-day login, not a one-time token. It is only removed from the address bar. The URL also persists in browser history and Stripe records. |
| Forged tier | PASS. The tier comes only from the DB, written by signed webhooks. |
| Webhook validation | PASS |
| SQL construction | PASS. The f-strings found insert only placeholders or fixed clauses; values are parameterized. |
| Passwords / brute force | PASS (bcrypt, per-username rate limit) |
| Admin/debug surfaces in the app | PASS. Provider health and admin tabs are gated by `can_diagnostics`/admin. |
| Public `/debug/status` on the billing service | P2 (F10): configuration and DB status disclosure |

## Dependencies

| Package | Advisories | Fix | Reachable in HSF? | Classification |
|---|---|---|---|---|
| tornado 6.5.7 | 3 (PYSEC-2026-3928, GHSA-wwv5-g3v4-889x, GHSA-8423-8fgw-73vq) | 6.5.8 (patch) | **Yes.** It is Streamlit's internet-facing web server. | **Launch condition (L1).** A patch bump plus lock refresh. Advisory details not reviewed offline, so the real severity is UNVERIFIED. |
| cryptography 49.0.0 | 1 (PYSEC-2026-3552) | 50.0.0 (major) | Yes: cookie encryption and `secret_box` | ACCEPTABLE TEMPORARY RISK until Run 74 (major bump needs testing) |
| soupsieve 2.8.4 | 2 | 2.9.0 | Low: HTML parsing of provider pages in scheduled jobs | ACCEPTABLE TEMPORARY RISK |
| GitPython 3.1.51 | 21 | 3.1.60 | Very low: no user-facing path uses git | ACCEPTABLE TEMPORARY RISK |

The lock also feeds the scheduled scanner, so any bump must be followed by the
autonomy-certification re-run (Run 74 plan).

## Repository Hygiene

- **Run 81 removed nothing needed.** Every scheduler, research, billing and alert entry
  point imports; CI's boot and doctor checks pass; no test was deleted (details in the
  Run 81 report).
- **Warnings:** CI's 3.13 log has 0 ResourceWarnings.
- **Remaining warnings:** Streamlit's AppTest temp directory (third-party) and FastAPI's
  `on_event` deprecation (`billing_service/main.py:26`, P2).

## Test Results

| Suite | Collected | Passed | Failed | Skipped | xfail | Warnings | Duration |
|---|---:|---:|---:|---:|---:|---|---:|
| Full, `.venv` py3.12, `-X dev -W always`, **outbound network blocked** | 1866 | 1840 | 0 | 26 | 0 | 1 (Streamlit AppTest temp dir, third-party) | 109.8 s |
| Lightweight CI env (`requirements-dev`, no streamlit) | 1866 | 1765 | 0 | 101 | 0 | 0 | 17.5 s |
| `unittest discover -s tests` | 1823 run | OK | 0 | 26 | — | — | 57.0 s |
| Billing contract (isolated venv, CI command) | 86 | 85 | 0 | 1 | 0 | 57 (FastAPI/Starlette deprecations) | 1.8 s |
| `ruff check .` (CI rules) | — | All checks passed | — | — | — | — | — |

Subtests: 170 passed. Named suites, all passing:
- billing (`test_billing_service`, `test_run72_pricing`, `test_run73_tier_differentiation`,
  `test_tier_enforcement`, `test_app_session`);
- alerts (`test_run77_breakout_alert_scale`, `test_alert_evaluation`);
- state (`test_run80_mobile_state`);
- hierarchy (`test_run79_information_hierarchy`);
- startup (`test_boot_stale_module`);
- frozen core and research (`test_autonomy_certification` incl. Gate U,
  `test_maturation_*`, `test_observation_*`, `test_signal_evidence`).

## CI Results

Smoke Checks on HEAD `c01598a` (run 36281585234):

| Job | Result |
|---|---|
| `smoke` | 1765 passed, 101 skipped |
| `core-dependency-import-smoke` | Ran 1823, OK (30 skipped: 26 need FastAPI, which `billing-contract` covers; **4 need scikit-learn and run nowhere in CI**) |
| `full-dependency-import-smoke` | success |
| `billing-contract` | 85 passed, 1 skipped |
| `dependency-audit` | failure (non-blocking), SEC1 |

**Coverage gaps:**
- The 4 scikit-learn model tests in `tests/test_ai_confidence.py` and
  `tests/test_prebreakout_model_persistence.py` don't run in CI. They pass locally (P2,
  F12).
- There is no CI test for billing-service authorization, which is why B1 went unnoticed.
- CI protection is otherwise adequate for a controlled launch.

## Frozen-Core Integrity

`git diff e3a2d5d..c01598a` (Run 61 certified → HEAD) over the frozen and research paths
(`scan/`, `ml_prebreakout.py`, `market_data.py`, `data/us_market_universe.py`,
`data/prices.py`, `data/fetch.py`, `scheduler/`, `research/`, `analytics/`, the maturation,
recovery, certification and readiness scripts, `db/hsf_observations.py`, `db/runs.py`,
`.github/workflows/`):

- **`analytics/intelligence_view.py`:** display metadata label "AI Confidence" → "5D
  outcome probability". Source, range and version key are unchanged.
- **`analytics/signal_leaderboard.py`:** two display labels renamed. Columns and ranking
  are unchanged. The labels are stored in the operational leaderboard table, not in
  research observations.
- **`.github/workflows/smoke.yml`:** adds the `billing-contract` job (CI only).

**Conclusion:** presentation and integration only. No scoring, ranking, scan-engine,
model, Gate U, cohort, maturation or scheduling behaviour changed. The certification test
suite passes at HEAD.

## Open Findings

| ID | Severity | Area | Problem | Evidence | Required before launch? | Recommended action |
|---|---|---|---|---|---|---|
| **B1** | **P0** | Billing / security | Unauthenticated `/create-portal-session` (and `/create-checkout-session` for active subscribers) returns any customer's Stripe billing-portal URL given only their email; public repo exposes the service URL | `billing_service/main.py:301-432`; no auth anywhere in the service; `config.py:102`; repo visibility PUBLIC | **Yes** | Require proof of identity on both endpoints: the Streamlit app signs each request (shared HMAC secret between app and billing service, with timestamp) or passes a server-verified session. Reject unsigned calls. Add CI tests for rejection. |
| **B2** | **P0** | Privacy / state | `active_watchlist_quote_rows` (plus `_watchlist_prior_rows`, `_loaded_user_settings`) survive logout; next account on the same tab sees the previous user's watchlist on Market Brief and can email it | `ui/app_session.ACCOUNT_SESSION_KEYS`; `ui/market_brief.py:225-230,1164,1296`; headless reproduction above | **Yes** | Add the three keys to `ACCOUNT_SESSION_KEYS`; make `_watchlist_rows` ignore session rows unless they belong to the current user; regression test (A → logout → B). |
| L1 | P1 | Dependencies | tornado 6.5.7 (Streamlit web server) has 3 advisories fixed in 6.5.8 | CI `dependency-audit` | Yes (launch condition) | Patch bump tornado in the lock, rerun CI and autonomy certification. Leave the cryptography major bump to Run 74. |
| F3 | P1 | Auth | Stripe return URLs carry a reusable 14-day login session (`rt`) | `ui/checkout.py:14-27`, `ui/auth.py:204-222` | Yes (launch condition) | Make `rt` single-use and short-lived (e.g. delete it on first use, 15-minute TTL). |
| L3 | P1 | Premium value | Premium's AI features depend on `ANTHROPIC_API_KEY` (absent from the secrets template) and default model `claude-opus-4-8` | `config.py:105-109`, `ui/ai.py` | Yes (manual check) | Confirm the key and a valid model ID in production secrets; add both to `secrets.toml.example`. If they are not set, don't sell Premium yet. |
| L4 | P1 | Billing | Live Stripe round trip never exercised | — | Yes (manual) | One test-mode checkout → webhook → tier → portal → cancel, plus Pro ↔ Premium switch, before inviting payers. |
| F8 | P2 | Trust copy | "MOST POPULAR" badge on Pro with no customers | Live landing | No | Remove until true. |
| F9 | P2 | Onboarding copy | The landing page never says what HSF Score represents | Live landing | No | One sentence plus a link to Methodology. |
| F10 | P2 | Billing ops | Public `/debug/status` reveals config/DB status | `billing_service/main.py:262` | No (fixed naturally by B1's auth) | Protect or remove. |
| F11 | P2 | Mobile | 17 controls under 32 px on the landing page at 375 px; signed-in phone widths unverified | Live measurement | No | Finish P2-13 manual check. |
| F12 | P2 | CI | 4 scikit-learn model tests never run in CI | CI skip reasons | No | Add scikit-learn to the core CI job. |
| F13 | P2 | Maintenance | FastAPI `on_event` deprecation; local venv drifted from lock (streamlit 1.49.1 / pyarrow 25) | Warnings; `pip` metadata | No | Migrate to `lifespan`; rebuild local venv from lock. |

## Manual Pre-Launch Checklist

1. **Stripe (test mode first, then live):**
   - `STRIPE_PRICE_PRO`/`PREMIUM` point at $19 and $39 monthly prices;
   - the webhook endpoint is registered to `/webhook` with the matching
     `STRIPE_WEBHOOK_SECRET`;
   - run one checkout → Pro visible in the app → portal → switch to Premium → cancel →
     back to Free.
2. **Billing service on Render:** `/health` returns 200 with no missing env; after B1,
   an unsigned portal request is refused.
3. **Streamlit secrets:** `database_url`, `APP_BASE_URL`, `BILLING_API_BASE`,
   `ANTHROPIC_API_KEY` plus a valid `ANTHROPIC_MODEL`, `APP_ENCRYPTION_KEY` (or `COOKIE_PASSWORD`, used for login cookies and paper-trading keys),
   email (Resend) settings and `ADMIN_USERS`.
4. **Email:** sign up with a fresh address, receive the verification email, reset the
   password, and trigger one Pro alert email.
5. **Account isolation:** sign in as A, open My Stocks, sign out, sign in as B, and
   confirm Market Brief shows B's watchlist only (after the B2 fix).
6. **Market data:** the latest `cron · US_MARKET` scan is from the last trading day, and
   the Today banner shows the correct age.
7. **Phone:** walk through Today, Scanner (cards), Stock Intelligence, My Stocks, alerts
   and Billing on a real phone around 390 px, signed in (closes P2-13).
8. **Scheduled workflows:** cron-job.org dispatches are active; the GitHub PAT used for
   dispatch is not near expiry.

## Launch Scope Recommendation

After Run 83 (B1, B2, L1, F3) and the checklist:

- **Scope: small paid beta, invite-only.** A limited number of people you can contact
  directly, so billing or data issues are caught personally rather than at scale.
- **Offer Pro: yes.** Its value (email alerts, more alerts, export, Nasdaq/combined scans,
  history) is implemented, enforced when alerts run, and CI-tested.
- **Offer Premium only if L3 is confirmed.** Its headline benefit is the AI features.
  Label "Early Breakout candidate research" and AI notes as **beta**.
- **Keep labelled as research:** model outputs and the Track Record, as they are today.
- **User cap:** no capacity evidence was gathered. Streamlit Cloud, the Neon connection
  limits and Render's free/paid tier were not load-tested. Don't commit to a number; grow
  invites gradually while watching the system-health workflow and Render logs.

## Final Decision Rationale

- **Most of the release candidate is sound:**
  - the Run 69 problems are closed;
  - screens agree on the same canonical data;
  - entitlements are enforced where they matter, including when alerts run;
  - claims are honest;
  - the pipeline is healthy and self-labels stale data;
  - the frozen research core is intact.
- **But two verified defects directly contradict the release definition:**
  - B1 lets a stranger manage a paying customer's billing. That is not acceptable for
    real payments, and it would only be exploitable once there are paying customers.
  - B2 shows one user's saved stocks to another account.
- **Why NO-GO:** under the audit's own standard (account-data leakage and billing
  security are P0), the verdict has to be NO-GO even though each fix is small.
- **What was not done:** no findings were fixed during this audit.

**Recommended Run 83 — launch blockers and launch conditions only:**

1. **B1:** authenticate the billing service's checkout and portal endpoints (signed
   requests from the app); protect `/debug/status`; CI tests for rejection.
2. **B2:** clear the three watchlist/settings keys on logout, and scope Market Brief's
   session rows to the current user; A → logout → B regression test.
3. **L1:** patch-bump tornado to 6.5.8 in the lock and re-run autonomy certification.
4. **F3:** make the Stripe `rt` restore token single-use and short-lived.

Then run the manual checklist, including the Stripe test-mode round trip and the
Premium AI key check. If the checklist passes, the verdict becomes CONDITIONAL GO for an
invite-only paid beta.
