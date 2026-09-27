# Run 85F — Full Product Entitlement Audit

## Executive Summary

Run 85F audited the authenticated product and background operations against the
single entitlement source in `ui/app_session.py`. The audit found and fixed
alternate render/export/worker paths that exposed paid evidence without changing
HSF Score, ranking, scanner results, model inference, or research capture.

The principal defects were Premium PreBreakout evidence in general-purpose views,
Pro historical outcomes on alternate pages, ungated watchlist export, lower-tier
paper-trading presentation, a real-time alert worker that did not reapply plan
limits after downgrade, and dynamically named Premium AI session entries that
could survive an account or tier transition.

## Baseline

- Branch: `dev`
- Starting HEAD: `b263862a528fb0186c215caf38e6ae924ad3c0be`
- Python: 3.12.6
- Canonical resolver: `ui.app_session.compute_entitlements`
- Capability map: `ui.app_session.FEATURE_MIN_TIER`
- Alert limits: `ui.app_session.ALERT_LIMIT_BY_TIER`
- Internal `basic` is displayed to customers as **Free**.

## Canonical Entitlement Matrix

| Capability | Free | Pro | Premium | Admin |
|---|---:|---:|---:|---:|
| Core Scanner / S&P 500 | Yes | Yes | Yes | Yes |
| HSF Score / basic Stock Intelligence | Yes | Yes | Yes | Yes |
| Watchlists / in-app alerts | Yes | Yes | Yes | Yes |
| Alert limit | 1 | 5 | 25 | 25 |
| Nasdaq, premarket, after-hours scans | No | Yes | Yes | Yes |
| Unusual volume / earnings | No | Yes | Yes | Yes |
| Interactive history / historical research / track record | No | Yes | Yes | Yes |
| CSV export | No | Yes | Yes | Yes |
| Email alerts | No | Yes | Yes | Yes |
| AI notes and summaries | No | No | Yes | Yes |
| PreBreakout / early-breakout evidence | No | No | Yes | Yes |
| Full universe / paper-trading workflow | No | No | Yes | Yes |
| Diagnostics / admin panel | No | No | No | Yes |

Admin override is explicit in `compute_entitlements`; Admin is an operational
role, not a customer plan. Labs/Kalshi is currently an authenticated experimental
surface with no canonical paid capability. Run 85F did not invent a gate for it.

## Entitlement Check Inventory

| Surface/operation | Capability | Canonical enforcement |
|---|---|---|
| Scanner universes/session modes | scan capabilities | `flags` from `compute_entitlements` |
| Historical Research / outcome stats | `can_track_record` | tabs and every audited alternate renderer |
| Scan history | `can_scan_history` | Scanner tabs |
| PreBreakout tab/evidence/lens | `can_early_breakout` | tab plus shared display redaction |
| AI summary/chat/narrative | `can_ai_notes` | gate before provider calls |
| CSV downloads | `can_export_csv` | operation path; Premium columns separately redacted for Pro |
| Paper trade workflow | `can_paper_trade` | Settings, Journal, paper actions |
| Alert creation count | tier alert limit | UI, scheduled runner, real-time worker |
| Email alert/digest | `can_email_alerts` / tier check | scheduler and recipient-specific rendering |
| Billing plan actions | explicit target-plan checks | billing authorization and Stripe flow |

Direct plan comparisons remaining in billing and background email code express
plan purchase transitions or recipient policy, not page-local substitutes for
the canonical capability resolver.

## Surface Audit

### Today

HSF Score and core setup ranking remain available to all plans. Top-setup display
now removes Premium PreBreakout evidence for Free and Pro without recalculating
or changing the underlying score.

### Market Brief

Run 85D gates remain intact. Premium candidate lists and AI text use
`can_early_breakout` and `can_ai_notes`; historical outcomes use
`can_track_record`. Opportunity, watchlist-match, movement, and watch-next
renderers now receive a redacted display copy for lower tiers. Snapshot freezing
continues to use canonical, unmodified observations.

### Scanner

HSF Score and ranking are unchanged. Premium model fields are removed from cards,
opportunity views, Model details, smart-alert suggestions, CSV exports, and the
Early breakout lens for lower tiers. Pro still receives CSV export but does not
receive Premium PreBreakout columns. Historical and Early Breakout tabs use their
canonical capabilities.

### Day Trader

Audited as an authenticated core capability. It has no paid capability in the
canonical map and no new gate was invented.

### Stock Intelligence

Core intelligence remains available. Premium PreBreakout evidence is redacted
before any section renders. Historical context remains guarded by
`can_track_record` and uses the correct Pro upgrade message.

### How HSF Works

Methodology now clearly describes historical research as Pro/Premium and
PreBreakout/AI-assisted research as Premium. It does not imply a different HSF
Score by plan.

### My Stocks

Watchlist state remains core and account-scoped. Premium PreBreakout evidence is
redacted; Historical Replay requires `can_track_record`; CSV requires
`can_export_csv`.

### Journal

Core journal access remains available. Paper-account activity, connection link,
and paper-trade-specific empty-state copy require `can_paper_trade`.

### Settings

Paper-account controls remain guarded by `can_paper_trade`. No lower-tier setting
was found that could activate a paid operation.

### Billing

Current plan naming, upgrade paths, authorization, restore-token handling, and
the Run 85C shared account card remain intact. Customer copy uses Free, not Basic.

### Labs

Labs/Kalshi requires authentication. No canonical customer-tier entitlement
exists, so it remains an authenticated experimental capability pending an
explicit product packaging decision.

## Background / Non-Page Operations

Scheduled alert evaluation enforces canonical plan caps and Pro+ email delivery.
The real-time worker now also selects newest alerts first and enforces 1/5/25/25
after downgrade. Morning digest recipients require Pro; PreBreakout picks require
Premium. Scan, score, persistence, and research jobs were not modified.

## AI Boundary

Audited provider call sites are gated before execution by `can_ai_notes`, Admin,
or the Premium three-step workflow. Dynamic AI session keys (`_ai_summary_*`,
`_ai_ticker_*`, `_ai_chat_*`, `brief_narrative_*`, `opp_ai_*`, and
`aic_explain_*`) are now cleared on downgrade, logout, and account transition.
Rule-based HSF output remains distinct from AI-generated text.

## Historical Research Boundary

The canonical gate is `can_track_record` (Pro+). Market Brief, Scanner,
Stock Intelligence, My Stocks replay, and Alerts outcome scorecards now agree.

## PreBreakout Boundary

The canonical gate is `can_early_breakout` (Premium+). A shared presentation
adapter removes Premium fields, signals, explanations, setup labels, lifecycle
signal names, and model probability from lower-tier copies. It never mutates the
source observation or HSF Score.

## Alerts

Creation limits are Free 1, Pro 5, Premium 25, Admin 25. UI, scheduled evaluation,
and real-time evaluation are covered. Email remains Pro+. Smart alert suggestions
cannot use PreBreakout evidence below Premium.

## Exports

Scanner and watchlist CSV generation require `can_export_csv`. Scanner export for
Pro additionally strips Premium model columns; Premium/Admin receive them.

## Email

Free receives no customer email. Pro can receive alerts/digest content without
Premium picks. Premium/Admin can receive PreBreakout picks. Rendering is selected
per recipient tier rather than inherited from a page session.

## Shared / Deep Links

Shared ticker links preserve their destination through authentication, but the
destination renderer applies current entitlements. Navigation hiding is not the
security boundary. Existing B4 first-render context tests remain green.

## Caching

Public market-data caches remain shared. Account-owned state remains covered by
the Run 83 identity boundary. Run 85F adds prefix-aware removal for dynamically
named entitlement-sensitive AI output, preserving only browser convenience
preferences such as Scanner view/lenses.

## Admin Behavior

Admin intentionally receives all capability flags through the canonical resolver
and has the same alert cap as Premium. Customer upgrade prompts are suppressed by
the existing shared plan component. Admin-only diagnostics remain separate.

## Findings

| ID | Severity | Surface | Capability | Expected Tier | Actual Before | Fix | Status |
|---|---|---|---|---|---|---|---|
| F-01 | High | Scanner/cards/intelligence | PreBreakout | Premium | Visible to lower tiers | Shared render redaction | Fixed |
| F-02 | High | Market Brief/Today/Stock Intelligence/My Stocks | PreBreakout | Premium | Alternate render paths leaked evidence | Redact before rendering | Fixed |
| F-03 | High | Scanner lens/smart alerts | PreBreakout | Premium | Lower tier could infer candidates | Gate lens and suggestion construction | Fixed |
| F-04 | High | Scanner CSV | Premium model evidence | Premium | Pro export contained fields | Redact Pro export | Fixed |
| F-05 | Medium | My Stocks CSV | Export | Pro | Free download path existed | Gate generation and control | Fixed |
| F-06 | High | My Stocks/Alerts | Historical research | Pro | Alternate replay/scorecards visible | `can_track_record` gates | Fixed |
| F-07 | Medium | Journal | Paper workflow | Premium | Lower-tier workflow copy/activity | `can_paper_trade` gate | Fixed |
| F-08 | High | Real-time worker | Alert cap | Tier-specific | Saved over-limit alerts still evaluated | Newest-first plan cap | Fixed |
| F-09 | Critical | Session cache | AI content | Premium/account owner | Dynamic keys survived transitions | Prefix-aware purge | Fixed |

## Files Changed

- `app.py`
- `billing_service/realtime_alerts.py`
- `pages/journal.py`
- `ui/alerts.py`
- `ui/app_session.py`
- `ui/discover.py`
- `ui/entitlement_view.py` (new)
- `ui/headline_score.py`
- `ui/market_brief.py`
- `ui/methodology.py`
- `ui/result_cards.py`
- `ui/results.py`
- `ui/results_intelligence.py`
- `ui/smart_alerts.py`
- `ui/stock_intelligence.py`
- `ui/today.py`
- `ui/watchlist_intelligence_feed.py`
- `ui/watchlists.py`
- `tests/test_realtime_alerts.py`
- `tests/test_run85f_entitlement_matrix.py` (new)
- `tests/test_smart_alerts.py`

## Entitlement Matrix Tests

- Run 85F: 14 passed
- Run 85D Market Brief: 9 passed
- Run 85E historical gating: 4 passed
- Combined entitlement/isolation regression: 100 passed
- Run 85C account card + Run 85 AI claims: 33 passed
- B1/B2/B3/B4: 24 passed, 12 skipped (optional integration environment)
- Billing unit module: 26 skipped because optional billing dependencies/services
  are not configured in this local test environment; full-suite billing tests
  covered by the repository's billing CI job.
- Alert suites: 103 passed
- Research/frozen-core suites: 99 passed

## Regression Results

- Full suite: **1,952 passed, 38 skipped, 176 subtests passed** in 85.65s
- Failures: 0
- Xfailed: 0
- Warnings: 0 reported by pytest
- Ruff (`E9,F,I`, existing CI policy): passed
- Streamlit startup smoke: passed after allowing a local loopback port
- `git diff --check`: passed

## Dependency Audit

The exact strict audit could not complete on macOS because `requirements.lock`
contains Linux-only `nvidia-nccl-cu12==2.30.7`; pip-audit attempted platform
resolution even with the pinned/no-dependency option. No dependency files were
changed. The strict Ubuntu CI audit completed successfully in Smoke Checks run
`36302605193`, confirming zero reported known vulnerabilities in the locked set.

## Frozen-Core Verification

No HSF Score formula, Breakout Score, scanner engine, ranking, scheduled scan,
model training/inference, Gate U, research capture, maturation, cohort, or
Autonomous Research Mode file was modified. Redaction is presentation/export
only; canonical observations and scores remain identical across tiers.

## Remaining Entitlement Risk

- Labs is intentionally authenticated but not assigned a paid capability; product
  policy should remain explicit if packaging changes.
- Definitive dependency status requires the Ubuntu CI audit.
- Live Free/Pro/Premium/Admin account checks, Stripe entitlement propagation, and
  actual recipient email verification belong to Run 86.

## Final Decision

**ENTITLEMENT MODEL VERIFIED**, subject to final live-account verification. The
code-level entitlement matrix, dependency audit, and regression suite are
internally consistent, and all verified leaks found in this audit are closed.

Next: **Run 86 — Final Live Launch Check**. Do not add another engineering or
feature audit before the live Free/Pro/Premium/Admin launch verification.
