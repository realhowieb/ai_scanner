# Run 69 — Production Readiness Audit

**Scope:** `realhowieb/ai_scanner` at `bae70e0` (main = dev), after Runs 62–68 and the P0–P2 backlog.
**Mode:** read-only. No production code, scoring, ranking, scheduler, scan-engine or research behaviour was changed. The only file added is this report.
**Frozen core:** the Run 61 certified research/scanner core (scan engine, scoring, cohorts, maturation, Run 56/58, Gate U) was not touched.

**Evidence labels:** *VERIFIED* = reproduced here (code read, test, headless run, or live signed-out site). *UNVERIFIED* = needs the signed-in production app or the production database, which this audit could not access (no credentials were used).

---

## 1. Executive summary

HSF is now a coherent product on paper: a clear signed-out front door, one headline score, a Today page, a results-first Scanner that opens on the latest full-market scan, Stock Intelligence with a change timeline, and My Stocks. Mobile layouts hold at phone and tablet widths with no horizontal overflow. The trust layer is solid: no performance claims, historical research is separated, and the methodology matches the product.

**It is not yet ready for paying users.** Three things stand in the way:

1. **The same stock can show different facts on different screens (P0).**
   - Opening a top setup from Today or a result card sends Stock Intelligence to the ticker's alert history instead of the scan the user just saw. It can then show a different HSF Score, or "Not currently ranked", for a stock Today ranked first.
   - Market Brief and the Scanner header read a different scan than Today and the Scanner. That scan can even be another user's personal scan.
2. **The paid offer is weak and partly inaccurate (P1).**
   - The product's core value (full-market ranking, Today, Stock Intelligence, lenses, cards, screens) is free.
   - The pricing table claims features Premium users don't get ("Diagnostics / Retrain", "Full Universe Mode" as a differentiator), and sign-up promises "AI-powered rankings" that don't exist for customers.
   - Pricing isn't visible before sign-up.
3. **Operational hygiene gaps (P1).**
   - The dependency audit has failed silently on every smoke run, including CVE fixes for tornado, the web server Streamlit runs on.
   - The Stripe/billing tests never run anywhere.
   - Every Scanner click makes about 7 uncached database round-trips.

None of these requires touching the frozen core. The smallest set of changes before putting HSF in front of paying users is the **P0 + P1 list in §12**.

**Recommended Run 70: data-consistency hardening.** One canonical source for the latest full-market scan and the opportunity passed between screens, so Today, Scanner, Stock Intelligence and Market Brief always agree (§13).

---

## 2. Current product architecture

| Layer | What it is | Key files |
|---|---|---|
| Frozen research core (certified Run 61) | Scheduled full-market scans (cron-job.org → `scheduled-scans.yml`), research capture, maturation, health, recovery | `scheduler/cron_runner.py`, `scan/engine.py`, `analytics/*`, `db/*` |
| Saved scan data | `runs` table; scheduled full-market runs are `username="cron"`, `label="US_MARKET"`; a daily snapshot flag is also updated by the cron | `db/runs.py` |
| Canonical scoring | HSF Score 0–100 = `ui/opportunities.score_breakdown` via `ui/results_intelligence.consolidate_scanner_results` | frozen formula, UI-side presentation |
| Market data layer (UI) | `ui/market_default.py` (Scanner default), `ui/market_scans.py` (Today, recap, new-since-visit), `ui/trust_banner.py` | read-only, cached |
| Pages | Today, Market Brief · Scanner, Day Trader · Stock Intelligence, How HSF works · My Stocks, Journal · Settings, Billing · Kalshi (Labs) | `pages/*.py`, `app.py` |
| Per-browser state | Encrypted cookie jar (sign-in, last visit, tour, saved screens) | `ui/auth_sessions.py`, `ui/browser_prefs.py`, `ui/last_visit.py` |
| Tiers | Basic / Pro / Premium via `FEATURE_MIN_TIER`; Stripe via `billing_service/` | `ui/app_session.py`, `pages/billing.py` |

**Where market data comes from today (three loaders):**

| Consumer | Loader | Rule |
|---|---|---|
| Today, Scanner default, trust banner, recap, new-since-visit | `ui/market_scans.py`, `ui/market_default.py`, `ui/trust_banner.py` | newest `cron` + `US_MARKET` run |
| Market Brief (incl. its Top Opportunities, PreBreakout picks) | `scheduler/morning_digest._latest_snapshot_df` | newest *snapshot* among the 25 latest runs of **any user**, else newest non-empty run of any user |
| Scanner header "Today's Market Snapshot" | `ui/app_user_profile._load_saved_snapshot_df` | session scan, else newest snapshot among the 10 latest runs of any user, else `runs[0]` |

---

## 3. User-journey findings

### A. Brand-new signed-out visitor (VERIFIED on the live site at 375/390/768/desktop in Run 62; re-checked after each deploy)

| Question | Answer from the landing page |
|---|---|
| What HSF does | Yes: hero, "Know what matters in the market right now.", Scan/Rank/Explain/Track |
| What HSF Score means | Partly: an example shows "HSF Score 82"; the definition is one click away (methodology) |
| What HSF does **not** claim | Yes: disclaimer plus "Research-first methodology" trust point |
| Why create an account | **Weak.** No pricing, no "free to start" statement near the hero, no example of Today |

- Methodology is public and reachable. Disclaimers are present. The sign-in form is in the first phone screen.
- **Gap (P1):** a signed-out visitor never sees pricing. `pages/billing.py` stops at line 299, before the pricing table at line 360, and shows a misleading warning that the account "isn't linked to an email address". The landing page has no pricing link or plan summary. VERIFIED in code.

### B. First-time signed-in user (code + headless run; live UI UNVERIFIED)

- **Today → Scanner → Stock Intelligence → My Stocks:** every page renders and is reachable from the grouped sidebar and the phone menu.
- **Market state and top setups** come from the trust banner, the snapshot and "Top setups right now".
- **Why HSF shows a stock:** the "Why" column/cards and Stock Intelligence's "Why HSF is showing this".
- **What changed:** new-since-visit, the recap and the Stock Intelligence timeline.
- **Saving:** Watch/Alert buttons on Stock Intelligence, and My Stocks.

**Problems:**

- **P0 · Inconsistent Stock Intelligence from Today/cards.** See D1 in §4.
- **P1 · Three onboarding systems stack for a new user on the Scanner** (`app.py`):
  - the Run 62+ tour, `render_tour("scanner")`;
  - the Run 28 first-run panel, `render_hsf_onboarding_entry`, with its own "HSF Market Intelligence" header and "Add your first stock" form;
  - the per-page "Got it" orientation, `render_scanner_orientation`, whose dismissal is session-only (`ui/onboarding.py::_render_dismissible_orientation`), so it reappears every session.
- **P1 · Three ways to run your own scan on one page:**
  - preset buttons inside the Scan filters popover ("🚀 Breakouts", "📈 High Volume", "💎 Swing Trades", "💰 Under $20");
  - the legacy S&P/NASDAQ/COMBO buttons (`ui/scans.py`);
  - the Premium Market/Strategy/Profile scanner (`ui/three_step_scanner.py`).

  A new user can't tell which to use, or that they aren't needed because the full market is already shown.
- **P1 · No way back to the market view after running your own scan.** `results_df` persists for the session and only logout clears it (`ui/auth.py` `auth_keys`); no control resets it.
- **P2 · Terminology mix:**
  - "Opportunity", "setup" and "signal" are used interchangeably.
  - Status words (STRONG/WATCH/CAUTION) are unexplained outside Stock Intelligence.
  - The Early Breakout tab says "pre-breakout predictions" (`ui/prebreakout_tab.py:63`).

### C. Returning daily user

- **Today works as a home:** market status and scan freshness (trust banner), snapshot, top 5 by HSF Score, new since your last visit, your watchlist in today's scan, session recap, and continuation buttons (Open, Show all in the Scanner).
- **P1 · The Scanner, not Today, is the post-login landing page**, so the daily loop starts one click away from the daily screen.
- **P2 · Today's "Top setups" has no minimum HSF Score.** With few qualifying names it lists "Caution" names. The headless run showed HSF 19 and 5 listed as "Top setups right now".
- **P2 · The Scanner's returning-user block** ("Your HSF market view: N need attention…") is computed from Market Brief snapshots, a third data source (§4 D2).

---

## 4. Data-consistency findings

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| **D1** | **P0** | Stock Intelligence ignores the scan the user came from when opened from **Today** (`ui/today.py::_open_button`) or a **result card** (`ui/result_cards.py`). Both pop `hsf_stock_opp`, and `pages/stock.py` passes only `current_opp` (never `current_row`). `build_stock_intelligence` then falls back to `signal_outcomes` opportunity history (fired-alert snapshots), or shows "Not currently ranked as an HSF opportunity". The results-intelligence panel, Market Brief, My Stocks and onboarding all set `hsf_stock_opp` correctly, so only the new P1 paths are affected. | VERIFIED in code: `grep hsf_stock_opp` shows `today.py:37` and `result_cards.py:84` pop it; `stock_intelligence.build_stock_intelligence` precedence is `current_opp` → `current_row` → history. |
| **D2** | **P0** | Market Brief (`ui/market_brief._compute_brief` → `scheduler/morning_digest._latest_snapshot_df`) and the Scanner header snapshot (`ui/app_user_profile._load_saved_snapshot_df`) choose "the latest scan" from the most recent runs of **any user**. So:<br>• Brief's Top Opportunities and PreBreakout picks can disagree with Today's top setups;<br>• if a user's personal scan is the newest usable run, **other users see that person's scan on Market Brief and in the header**. | VERIFIED in code (`list_runs(limit=25)` / `list_runs(limit=10)` with no username or label filter; fallback to newest non-empty run / `runs[0]`). How often it happens in production is UNVERIFIED (depends on whether cron snapshots are always among the newest runs). |
| D3 | P1 | **Row order vs headline score.** Scanner rows keep the scanner's order (BreakoutScore); the headline is HSF Score. In your screenshot: MXL 71, GRAL 45, AKAM 64. Today's "Top setups" is ordered by HSF Score, so Today's #1 is not necessarily the Scanner's first row. | VERIFIED (your screenshot; `headline_score.add_hsf_score_column` preserves order by design). |
| D4 | P2 | **Session label.** The Scan filters popover caption uses `ui/app_runtime.get_market_session` (no holiday calendar); the trust banner uses the holiday-aware `analytics.market_calendar`. They disagree on NYSE holidays. | VERIFIED in code. |
| D5 | P2 | **Empty or corrupt latest run.** `market_default` and `market_scans` return None and the Scanner falls back to "You haven't run a scan in this session yet…". The actual cause (no usable market data) isn't named; the trust banner does say "Latest market scan unavailable". | VERIFIED (unit tests: loader returns None on empty results). |
| D6 | OK | Latest-run selection, timestamps, ages and universe size are consistent across the trust banner, Today, the Scanner caption, recap and new-since-visit (all use `cron` + `US_MARKET`, newest first). HSF Score values are identical across the Scanner column, results-intelligence panel, Today top setups and recap (one canonical `consolidate_scanner_results` path). | VERIFIED (tests `test_p07_headline_score`, `test_p1_backlog`, `test_run63_market_default`). |

---

## 5. HSF Score consistency

- **Canonical path everywhere HSF Score is shown:**
  - Scanner column, detail cards, results-intelligence panel;
  - Today top setups and watchlist, recap standouts;
  - Stock Intelligence (for `current_opp` / `current_row`).

  All go through `score_breakdown` (formula unchanged). VERIFIED.
- **Where secondary model outputs still compete with HSF Score:**

| ID | Sev | Where | Issue |
|---|---|---|---|
| S1 | P1 | `ui/alerts.py:257` | The Breakout alert asks for **"Breakout score ≥"**. That scale (≈0–60) is now hidden by default and differs from the headline 0–100. Users can't pick a sensible threshold. Fix options: explain the scale in the form (UI-only), or express the alert in HSF Score (changes alert evaluation in `scheduler/alert_runner.py`; production scheduled behaviour, not the certified research core). |
| S2 | P2 | `ui/market_brief.py:1266` "🧠 PreBreakout picks" | Its own ranked list with "% likelihood" on the same page as HSF Top Opportunities. |
| S3 | P2 | `ui/watchlist_intelligence.py:335` | A PreBreakout column in the watchlist table. |
| S4 | P2 | Early Breakout tab (`ui/prebreakout_tab.py`) | "Confidence" column and "predictions" wording for a model likelihood. |
| S5 | OK | Day Trader "DT Score", watchlist "Alert Priority" | Page-specific and labelled ("attention signal, not a prediction"). |

---

## 6. Performance findings

**Measured (VERIFIED, 100-row scan, local):** the presentation work on each Scanner rerun is about **7 ms of CPU**:

| Step | Time |
|---|---|
| Why column | 0.7 ms |
| HSF Score column | 1.1 ms |
| Lens counts | 3.0 ms |
| Lens filter | 1.0 ms |
| Today top-5 | 0.9 ms |

Latency is dominated by **database round-trips** (each opens a new Neon connection) and Streamlit rendering.

**Uncached DB work on every Scanner rerun (VERIFIED in code; latency UNVERIFIED):**

| Source | Queries | File |
|---|---|---|
| Tier Sync direct tier query | 1 | `auth/tier_sync.py` via `app._resolve_tier_state` |
| Watchlists panel | 2 | `ui/watchlists.render_watchlists_panel` |
| Returning-user onboarding block | ≈4 (watchlist ×2, market-brief snapshots, intelligence alerts) | `ui/onboarding.render_hsf_onboarding_entry` → `analytics/watchlist_intelligence.build_watchlist_intelligence` |
| **Total** | **≈7 per click** | |

With a typical Neon connection plus query at 20–100 ms (UNVERIFIED), that's about 0.15–0.7 s per interaction before rendering.

| ID | Sev | Finding | Recommendation |
|---|---|---|---|
| PF1 | P1 | ≈7 uncached DB round-trips on every Scanner interaction (above). | Cache per-user reads for 30–60 s and clear them on that user's writes, or move the returning-user block to Today, where it belongs. Not frozen. |
| PF2 | P1 | Stock Intelligence reads `fetch_ticker_opportunity_history` uncached on every rerun (`ui/stock_intelligence.py`). | Short cache keyed by ticker. |
| PF3 | OK | **Lazy panels work.** Historical research, Early Breakout and Scan History do no work until switched on. | VERIFIED by AppTest (`tests/test_run67_68_leftovers.py`: heavy block absent until toggled). |
| PF4 | OK | Scheduled-run loads are cached (run list 120 s, each run's results 30 min, immutable). Today loads at most 4 run frames. | VERIFIED in code. |
| PF5 | P2 | The Scanner imports the Premium three-step scanner, which imports `ml_prebreakout` (xgboost) for every user on process start. | Lazy-import inside the Premium path. |
| PF6 | P2 | The score map rebuilds its Plotly figure on every rerun. | Cache the figure per run id. |
| PF7 | (decision) | Your own full-universe scans run inside the request through `scan/engine.run_breakout_scan`. | **Frozen-core architectural decision**, out of scope: decoupling needs the certified engine to stop calling Streamlit directly, then re-certification (Gate U). Mitigated by the market default. |

---

## 7. Mobile findings

**VERIFIED with a local harness rendering the real signed-in components** (Today, Discover bar, cards, tour, phone menu, filters) with sample data, at 390, 430, 768 and desktop widths. The live signed-in app itself is UNVERIFIED.

| Width | Horizontal overflow | Phone menu | Notes |
|---|---|---|---|
| 390 | none (`scrollWidth == 390`) | shown | tour buttons stack into 3 full-width rows |
| 430 | none | shown | — |
| 768 | none | hidden (sidebar used) | tour buttons in one row; pills wrap |
| desktop | none | hidden | — |

| ID | Sev | Finding |
|---|---|---|
| M1 | P2 | **Smallest buttons are 28 px tall** (tour Back/Next/Skip, secondary buttons), below the ~44 px phone guideline. |
| M2 | P2 | The tour card takes about half the first phone screen (three stacked buttons). |
| M3 | P2 | **Two navigations on phones:** the "☰ Menu" plus Streamlit's collapsed-sidebar arrow. They work, but duplicate. |
| M4 | P2 | Pro/Premium interactive grid (`st.dataframe`) still needs sideways swiping. The Cards view solves it, but Table stays the default (Streamlit can't detect screen width server-side). |
| M5 | OK | Landing, auth, methodology, trust banner, cards, Today and filters popover all fit phone width. Streamlit "Fork"/GitHub chrome is hidden on every page, including sign-in gates (VERIFIED live). |

---

## 8. State / persistence findings

| ID | Sev | Scenario | Finding |
|---|---|---|---|
| ST1 | P1 | Own scan vs market default | After your own scan there's no way back to the latest market scan for the session (see §3B). |
| ST2 | P2 | Lens and Table/Cards selections | Both reset after visiting another page: Streamlit drops widget state for widgets that aren't rendered. Saved screens mitigate. |
| ST3 | P2 | Shared computer | Tour completion, saved screens and last-visit follow the **browser**, not the account. Two people on one browser share them; logout doesn't clear them. |
| ST4 | P2 | Shared ticker link | A signed-in user opening `/stock?ticker=X` in a new tab still sees "Sign in…" and needs one click on "Go to login". Streamlit restores sign-in only on the main page. VERIFIED live (signed out). |
| ST5 | P2 | Pending post-login redirect | `hsf_after_login_page` stays set if the user never signs in during that session; a later sign-in in the same session jumps to Stock Intelligence. |
| ST6 | OK | Logout | Shows "You've signed out." (fixed); clears auth and results. VERIFIED by tests. |
| ST7 | OK | Expired session | "Your session has expired" only for genuinely invalid sessions. |
| ST8 | OK | Widget-key issues | Saved-screen apply and Today's "Show all new" use a pending key (no "modified after instantiation" error); the filters popover keeps widget state every run. VERIFIED by AppTest. |
| ST9 | UNVERIFIED | Cookie size | Saved screens are capped at 10 and 2,000 characters; real-browser cookie behaviour on Streamlit Cloud wasn't exercised. |

---

## 9. Trust / claims findings

- **Clean:**
  - No guaranteed-return, beat-the-market, win-rate or probability-of-profit language in user copy (swept `ui/`, `pages/`, scheduler emails).
  - Historical research is labelled and collapsed, and the cherry-picked "best" figures are gone (Run 62).
  - The methodology matches behaviour (scheduled full-market scans, HSF Score as ranking, point-in-time results).
  - The disclaimer is present on landing, methodology and footer.

| ID | Sev | Finding | File |
|---|---|---|---|
| T1 | **P1** | Sign-up promises **"AI-powered rankings"**; no AI ranking is shown to customers (AI Confidence is admin-only). | `ui/auth.py:353` |
| T2 | **P1** | Pricing lists **"Diagnostics / Retrain" as Premium**; `can_diagnostics` is admin-only. | `pages/billing.py` pricing table; `ui/app_session.FEATURE_MIN_TIER` |
| T3 | P2 | The methodology describes AI Confidence, which customers never see. | `ui/methodology.py` |
| T4 | P2 | "pre-breakout predictions" wording. | `ui/prebreakout_tab.py:63` |

---

## 10. Monetization findings

**What each tier gets today (VERIFIED from `FEATURE_MIN_TIER` and the UI):**

- **Free (Basic):**
  - latest full-market ranking (about 100 setups) with HSF Score and "Why";
  - Today, lenses, cards, saved screens, Stock Intelligence (with timeline), My Stocks;
  - 1 alert, S&P scans, methodology.
- **Pro ($19):** email alerts and 5 alerts, CSV export plus the interactive grid, earnings, scan history, historical research, Nasdaq scans.
- **Premium ($39):** AI notes/summary/chat, Early Breakout tab, own full-universe scans, paper trading, three-step scanner, 25 alerts.

**Most realistic upgrade reasons today:**

1. **Email alerts (Pro).** This is the only paid feature that brings users back without opening the app. It's strong.
2. **More alerts plus CSV and the interactive grid (Pro).** Moderate, for power users.
3. **AI notes (Premium).** Moderate. It's the only Premium feature that clearly adds value beyond the free ranking.

**Assessment: weak.** The free tier already includes the product's core value, and Premium's headline differentiators are either invisible (Early Breakout, because the market scan usually has no PreBreakout data), moot (Full Universe, since the full market is free), or wrongly claimed (Diagnostics).

| ID | Sev | Finding |
|---|---|---|
| MZ1 | P1 | Pricing copy doesn't match the product: see T1 and T2. Paid features that exist aren't listed (Historical research: Pro; paper trading: Premium). "Curated Breakout Scans" is outdated wording. |
| MZ2 | P1 | Premium differentiation is unclear (above). This is a packaging decision for you, not a bug. |
| MZ3 | P1 | Pricing isn't visible before sign-up (§3A). |
| MZ4 | OK | Gating doesn't interrupt the core journey: Today, the ranking, Stock Intelligence and My Stocks work on Free. Locked features show "🔒" notes. |
| MZ5 | OK | Nothing important is accidentally ungated. The free full-market default is a product choice to confirm. |

---

## 11. Dead / duplicate surface (cleanup list — nothing deleted)

| Item | Status | Evidence |
|---|---|---|
| `ui/pages.py`, `ui/pages_main.py` | Dead (legacy multi-page loader, admin run buttons) | Only referenced by each other and `__init__` |
| `ui/components.py` | Dead | No importers |
| `ui/heat_strip.py` | Dead in the app (retired; only a test imports it) | `tests/test_whats_new.py` |
| `ui/watchlist_alerts.py` | Dead (duplicate `_latest_snapshot_df`, unused `run_watchlist_alerts`) | No callers |
| `ui/history.render_history_expander` | Imported in `app.py`, never called | `app.py:221` |
| `pages/alerts.py` | Duplicate of the My Stocks → Alerts tab; kept for "Alert me on this" shortcuts | Out of the nav; reachable by URL |
| Onboarding × 3 | Tour, Run 28 first-run panel, per-page orientation | §3B |
| Scan entry points × 3 | Filter presets, legacy quick scans, Premium three-step | §3B |
| Market Brief vs Today | Overlapping daily summaries with **different data sources** | §4 D2 |
| Early Breakout tab vs Early breakout lens | Same concept, two surfaces | — |
| Day Trader vs "Live movers" lens link | Link only; fine | — |
| `ui/showcase.py` screenshot mode | Hidden mode, still maintained | 12 references |

---

## 12. Test / CI results and prioritized findings

### Tests (VERIFIED, local, Python 3.12)

- **Full suite:** 1,761 passed, 9 skipped, 1 warning, 171 subtests, 20.6 s.
- **Smoke-equivalent environment** (requirements-dev, no streamlit/requests): 1,695 passed, 75 skipped.
- **CI:** Smoke Checks pass on `bae70e0`; **dependency-audit fails** on every run (non-blocking).
- **Skipped:** all 9 are `tests/test_billing_service.py` (FastAPI not installed). The smoke job installs only `requirements-dev.txt`, so **the Stripe/billing webhook tests run nowhere**.
- **Warning:** `ResourceWarning: unclosed file` in `tests/test_showcase_mode.py:84`.
- **Slowest:** `tests/test_signal_evidence.py` end-to-end tests at about 2.2 s each (research statistics; fine).

**Security (from the failing audit):**

| Package | Pinned | Fix available |
|---|---|---|
| tornado (Streamlit's web server) | 6.5.7 | 6.5.8 |
| cryptography (cookie/auth encryption) | 49.0.0 | 50.0.0 |
| soupsieve | 2.8.4 | 2.9.0 |
| gitpython | 3.1.51 | 3.1.60 |

**Missing regression coverage created by Runs 62–68:**

- **Stock Intelligence opened from Today or a card** shows the scanned opportunity. There's no test, and D1 slipped through.
- **Market Brief and header snapshot select the cron `US_MARKET` run.** No test (D2).
- **An end-to-end AppTest of `app.py` `main()`** with auth mocked, covering results slot, market default, lenses and cards together. Today only individual pieces are tested.
- **Real cookie round-trip** for tour and saved screens through `EncryptedCookieManager`. Only a fake jar is tested.
- **The post-login redirect** to a shared ticker is only source-checked.
- **Mobile/navigation** is CSS-string and harness checks only; there's no visual regression test.

### Prioritized findings table

| ID | Sev | Finding | Files | User impact | Recommended fix | Frozen core? |
|---|---|---|---|---|---|---|
| D1 | **P0** | Stock Intelligence disagrees with Today/cards for the same ticker | `ui/today.py`, `ui/result_cards.py`, `pages/stock.py` | "Top setup" opens as "Not currently ranked" or a different score; trust damage on the main daily path | Pass the canonical opportunity for that ticker (from the shown scan) as `hsf_stock_opp`; regression test | No |
| D2 | **P0** | Market Brief / header snapshot read "latest runs of any user" | `ui/market_brief.py`, `ui/app_user_profile.py`, (`scheduler/morning_digest.py` loader) | Brief disagrees with Today; can surface another user's personal scan | Point both at `ui/market_scans` (cron `US_MARKET`), leaving the scheduler's digest loader untouched; tests | No |
| ST1 | P1 | No way back to the market view after your own scan | `ui/results_tabs.py`, `ui/market_default.py` | Users lose the full-market view for the session | "Back to latest market scan" control that clears the session scan | No |
| D3 | P1 | Row order (scanner) vs headline (HSF Score) | `ui/headline_score.py` | Scanner's first rows ≠ Today's top setups | Decision: default display sort by HSF Score, or label "scanner order" | No (display only; product decision) |
| S1 | P1 | Breakout alert uses the hidden Breakout-score scale | `ui/alerts.py` (`scheduler/alert_runner.py` if changed) | Users can't choose a threshold | Explain the scale in the form (UI), or move to HSF Score (alert evaluation change) | No (alerting, not research) |
| T1/T2/MZ1 | P1 | Inaccurate pricing and sign-up claims | `ui/auth.py`, `pages/billing.py` | Paying users expect features they won't get | Correct the copy to match `FEATURE_MIN_TIER` | No |
| MZ2 | P1 | Weak paid differentiation | `ui/app_session.py` | Low conversion | Packaging decision (not a code fix) | No |
| MZ3 | P1 | Pricing hidden before sign-up | `pages/billing.py`, `ui/landing.py` | Visitors can't evaluate cost | Public plan summary on landing or methodology | No |
| PF1 | P1 | ≈7 uncached DB round-trips per Scanner click | `ui/onboarding.py`, `ui/watchlists.py`, `auth/tier_sync.py` | Sluggish interactions | Short per-user caches with write invalidation; move returning-user block to Today | No |
| PF2 | P1 | Stock Intelligence history read uncached | `ui/stock_intelligence.py` | Slow reruns on that page | Cache per ticker | No |
| U1 | P1 | Three onboarding systems stacked | `app.py`, `ui/onboarding.py`, `ui/tour.py` | Cluttered first run | Keep the tour; retire the first-run panel and session-only orientations | No |
| U2 | P1 | Three scan entry points on one page | `ui/filters.py`, `ui/scans.py`, `ui/three_step_scanner.py` | Confusion | One "Custom scan" entry point | No |
| U3 | P1 | Scanner, not Today, is the landing page | `app.py` | Daily loop starts off the daily screen | Redirect to Today after sign-in (once per session) | No |
| SEC1 | P1 | Known CVEs in the lock (tornado, cryptography, …); audit failing silently | `requirements.lock`, `.github/workflows/smoke.yml` | Security exposure | Bump the pins; the lock also feeds the cron, so **re-run certification afterwards** as a precaution | Environment only; re-certify |
| CI1 | P1 | Billing/Stripe tests never run | `.github/workflows/smoke.yml`, `requirements-dev.txt` | Payment regressions undetected | Install FastAPI in a test job | No |
| D4 | P2 | Session label not holiday-aware in the filters popover | `app.py`, `ui/app_runtime.py` | Wrong label on holidays | Use `trust_banner.market_session_label` | No |
| D5 | P2 | Empty/corrupt run shows the "no scan yet" message | `ui/results_empty.py` | Misleading empty state | Distinct "market data unavailable" message | No |
| C1 | P2 | Today "Top setups" has no floor | `ui/today.py` | Weak names shown as "top" | Minimum HSF Score or "Watch+" only | No |
| S2–S4 | P2 | Secondary model outputs compete with HSF Score | `ui/market_brief.py`, `ui/watchlist_intelligence.py`, `ui/prebreakout_tab.py` | Score confusion | Move under "model details" | No |
| M1–M4 | P2 | Touch targets, tour height, dual nav, table swipe | `ui/tour.py`, `ui/chrome.py` | Phone friction | Larger buttons on phones, compact tour, hide sidebar arrow on phones | No |
| ST2–ST5 | P2 | Widget-state resets, per-browser prefs, extra sign-in click, stale redirect | `ui/discover.py`, `ui/browser_prefs.py`, `pages/stock.py` | Minor friction | Persist lens/view in session keys; clear redirect on logout | No |
| T3/T4 | P2 | Methodology mentions a hidden model; "predictions" wording | `ui/methodology.py`, `ui/prebreakout_tab.py` | Minor confusion | Copy edits | No |
| PF5/PF6 | P2 | xgboost import for everyone; score-map figure rebuilt | `ui/three_step_scanner.py`, `ui/score_map.py` | Cold-start and rerun cost | Lazy import; cache the figure | No |
| DEAD | P2 | Dead modules and imports (§11) | see §11 | Maintenance cost | Delete in a cleanup run | No |
| W1 | P2 | ResourceWarning in a test | `tests/test_showcase_mode.py` | — | Use a context manager | No |
| PF7 | — | Your own scans run in-request | `scan/engine.py` | Slow custom scans | **Future frozen-core decision** requiring re-certification | **Yes** |

---

## 13. Recommended Run 70 — Data-consistency hardening

**Objective:** make every screen agree on the same market facts. One canonical source for "the latest full-market scan", and one canonical opportunity handed between screens.

**Scope (all UI-side, no frozen-core changes):**

1. **D1:** Today and result-card "Open" pass the canonical opportunity for the ticker from the scan being shown (`consolidate_scanner_results` → `hsf_stock_opp`), so Stock Intelligence matches the screen the user came from.
2. **D2:** Market Brief and the Scanner header snapshot read the latest `cron` / `US_MARKET` run via `ui/market_scans`. This ends cross-user leakage and makes Brief, Today and the Scanner agree. The scheduler's digest loader stays as is.
3. **ST1:** a "Back to latest market scan" control after your own scan.
4. **D4/D5:** the holiday-aware session label everywhere, and a distinct "market data unavailable" state.
5. **Regression tests** for each, plus an AppTest of `app.py` `main()` with auth mocked covering market default → lens → card → Stock Intelligence.

**Why this outranks the alternatives:**

- **It's the only P0 class, and it attacks trust directly.** HSF's positioning is *transparent, point-in-time, evidence-first*. A stock that is "Top setup · HSF 68" on Today and "Not currently ranked" one tap later contradicts that promise on the most-used path. D2 can also show one user's scan to others.
- **Monetization fixes (pricing copy, packaging)** matter, but they're mostly copy and decisions, and there's little point asking people to pay while screens contradict each other. The copy corrections (T1/T2) are small and can ride along in Run 70 or 71.
- **Performance (PF1)** is a real latency tax but not a correctness failure; it's a good Run 71.
- **Security (SEC1)** should be scheduled promptly, but bumping the lock touches the environment the certified scanner runs in, so it deserves its own run with re-certification rather than being bundled.
- **Mobile and UX simplification (U1–U3, M1–M4)** are polish. The layouts already work at phone width.

**Suggested order after Run 70:**

| Run | Focus |
|---|---|
| 71 | Performance: per-user caching, returning-user block to Today |
| 72 | Pricing copy and plan visibility |
| 73 | Dependency security bump + re-certification |
| 74 | Onboarding and scan-entry simplification, and dead-code cleanup |
