# Run 86 — Final Launch Gate

## Executive Summary

The HSF release candidate is code-ready. The exact release SHA passes the full
local regression and all GitHub Smoke Checks, including billing authorization,
dependency auditing, deployment doctor, and Streamlit startup. No B1-B4,
entitlement, HSF Score, scanner, or AI trust-boundary regression was found.

The launch decision is **CONDITIONAL GO**, not GO, because production deployment
identity and external integrations cannot be truthfully certified from this
repository environment. Real Free/Pro/Premium/Admin accounts, Stripe checkout and
webhooks, email delivery, and physical/mobile browser behavior require live tests.

## Release Candidate

- Branch: `dev` (also current `origin/main`)
- HEAD: `96678747f112814ea78828e2d2442059f0f6fd72`
- Commit: `Document Run 85F CI verification`
- Run 85F implementation: `c7c1242696061b2a381da3e54a70855c2f2260cb`
- Python: local 3.12.6; CI/deploy validation 3.13
- Streamlit: locked deployment/CI 1.54.0; existing local venv 1.49.1
- Deployment dependencies: `requirements.txt` constrained by `requirements.lock`
- Streamlit config: minimal toolbar and curated navigation; default page list hidden
- Exact-SHA CI: Smoke Checks run `36302841590`, success

No commits follow Run 85F. `origin/main` and `dev` both resolve to this release
candidate. Repository branch equality does not prove which SHA Streamlit Cloud is
currently serving.

## Automated Test Results

- Collected: 1,990 tests
- Passed: 1,952
- Failed: 0
- Skipped: 38
- Xfailed: 0
- Subtests passed: 176
- Warnings: 0 reported by pytest
- Duration: 85.86 seconds

Focused B1/B2/B3/B4 and Run 85/85B/85C/85D/85E/85F regression:
95 passed, 12 skipped in 5.93 seconds. Skips are optional integration tests in the
local environment; the isolated CI billing contract passed on the exact SHA.

Exact-SHA GitHub jobs all passed:

- smoke (compile, lint, imports, unit smoke)
- core dependency imports and dependency-backed tests
- deployment readiness doctor
- live Streamlit startup smoke
- full application dependency imports
- billing contract
- dependency audit

## Security

The strict Ubuntu `pip-audit` result for the exact SHA is: **No known
vulnerabilities found**. Locked remediated versions remain:

- tornado 6.5.8
- cryptography 50.0.0
- soupsieve 2.9
- GitPython 3.1.60

B1 billing authorization, B2 account isolation, B4 first-render account context,
and single-use restore tokens remain covered. Run 85F found no remaining obvious
entitlement bypass after closing alternate PreBreakout, historical research,
export, alert-worker, and dynamic AI-cache paths.

## Entitlement Matrix

`ui.app_session.FEATURE_MIN_TIER`, `compute_entitlements`, and
`ALERT_LIMIT_BY_TIER` remain authoritative.

| Capability | Free | Pro | Premium | Admin | Gate status |
|---|---:|---:|---:|---:|---|
| Core Scanner / S&P 500 | Yes | Yes | Yes | Yes | PASS |
| HSF Score | Yes | Yes | Yes | Yes | PASS |
| Basic Stock Intelligence | Yes | Yes | Yes | Yes | PASS |
| Market Brief core | Yes | Yes | Yes | Yes | PASS |
| My Stocks / watchlists | Yes | Yes | Yes | Yes | PASS |
| In-app alerts | 1 | 5 | 25 | 25 | PASS |
| Email alerts | No | Yes | Yes | Yes | LIVE TEST REQUIRED for delivery |
| CSV export | No | Yes | Yes | Yes | PASS |
| Scan history / historical research / track record | No | Yes | Yes | Yes | PASS |
| Nasdaq, premarket, after-hours, unusual volume | No | Yes | Yes | Yes | PASS |
| PreBreakout / early-breakout evidence | No | No | Yes | Yes | PASS |
| AI notes/summaries | No | No | Yes | Yes | PASS for gate; LIVE TEST REQUIRED for provider |
| Full-universe / paper-trading workflow | No | No | Yes | Yes | PASS for gate |
| Diagnostics/admin panel | No | No | No | Yes | PASS |

Paid tiers unlock monitoring, investigation, research, and workflow. They do not
receive a different HSF Score.

## Account / Plan State

Internal `basic` renders as **Free**. Customer-facing names remain Free, Pro,
Premium, and Admin. The shared account card is rendered through shared navigation
or the Scanner sidebar on Today, Market Brief, Scanner, Day Trader, Stock
Intelligence, How HSF Works, My Stocks, Journal, Settings, and Billing.

Automated expectations remain:

- Free: Upgrade to Pro
- Pro: Upgrade to Premium
- Premium: no inappropriate upsell
- Admin: no customer upsell

B4 establishes identity, plan, entitlements, navigation, and CTA before the first
authenticated page renders. B2 clears account-owned and dynamically named Premium
state on logout/account transition while preserving browser-only preferences.
Live four-account first-render and transition checks remain required.

## Scanner

B3 passes: a populated watchlist cannot replace canonical Scanner results. The
Scanner loads the latest full-market results, adds the canonical HSF Score, ranks
by that headline score, and only then applies presentation lenses. Run 85F keeps
Premium evidence out of lower-tier cards, model details, CSV, smart alerts, and
the Early breakout lens without mutating results or ranking.

## Market Brief

Run 85D passes. Free cannot render Premium PreBreakout/AI or Pro historical
research. Pro receives historical capabilities without Premium candidate/AI
content. Premium/Admin receive their canonical capabilities. AI entitlement is
checked before provider execution, not only before display.

## Stock Intelligence

Run 85E passes. Core intelligence is available to Free. Historical research uses
`can_track_record` (Pro+), and PreBreakout/AI evidence follows Premium gates. A
direct ticker/deep link reaches the same gated renderer and cannot bypass it.

## HSF Score Invariant

**PASS.** Run 85D renders identical HSF Scores across Free, Pro, Premium, and
Admin for the same observations. Run 85F redacts only presentation evidence and
explicitly preserves score fields. No score, ranking, scanner, or model file was
changed in Runs 85F/86.

## AI Trust Boundary

**PASS in code/tests; provider invocation needs a live Premium check.** Genuine AI
surfaces are labeled Claude/AI and gated before calls. HSF, not AI, owns the score
and canonical ordering. Prompts and regression tests prohibit direct buy, sell,
hold, entry, exit, sizing, or price-target instructions. Rule/template output is
not relabeled as AI.

## Stripe

| Operation | Status | Evidence |
|---|---|---|
| Checkout authorization and contract | PASS | B1 and isolated billing CI |
| Free to Pro checkout | LIVE TEST REQUIRED | Real Stripe/browser interaction unavailable |
| Webhook to entitlement propagation | LIVE TEST REQUIRED | Mock contract is not live delivery |
| Pro to Premium upgrade | LIVE TEST REQUIRED | Requires real customer/subscription |
| Billing portal ownership | LIVE TEST REQUIRED | Authorization passes; live Stripe customer untested |
| Cancellation/downgrade | LIVE TEST REQUIRED | Code/tests cover state; live event untested |
| Restore token replay protection | PASS | Single-use atomic-consume regression |

## Email

**LIVE TEST REQUIRED.** Construction, tier-specific content, alert preference, and
delivery-state behavior are tested. No real message was delivered from this
environment, so provider configuration, inbox receipt, and production sender
reputation are unverified.

## Mobile

**LIVE TEST REQUIRED.** Run 80 source/regression coverage remains green, but code
and CSS inspection are not device verification. Test Today, Scanner cards/lenses,
Market Brief, Stock Intelligence, My Stocks, mobile navigation, authentication,
and Billing at approximately 375, 390, and 430 pixels on a real browser/device.

## US_MARKET

Status: **FRESH for the latest trading day** (current date is Sunday,
2026-09-27).

- Workflow: Scheduled Market Scans run `36175904943`, success
- Session: regular
- Universe: US_MARKET
- Started: 2026-09-25 18:50:49 UTC
- Completed/exported: 2026-09-25 18:55:19 UTC
- Requested: 11,497 symbols
- Processed: 10,844
- Skipped: 652
- Candidates: 100
- Errors: 0
- Snapshot/export status: success
- Scanner commit: `5c2e08517a0e908305bbd6478c7534fc31a45680`

The snapshot diagnostic reports clustered calibrated PreBreakout probabilities
(68 of 100 at the most common value) while raw PreBreakout scores are 100/100
unique. This is a visible diagnostic, not a Run 86 scoring change or launch
blocker.

Latest scheduled non-dry-run maturation (`36202017693`) succeeded on 2026-09-25:
4,233 observations scanned, 18 outcomes attached, 1,213 symbols deferred, and an
estimated five-run backlog. The later capacity dry run processed the full ready
set without provider rate limits. This is an evidence-freshness limitation to
monitor, not failure of current scanning or user state.

## Deployment

`main` and `dev` both point to the release candidate, and exact-SHA CI is green.
No connected deployment API or signed-in production admin surface proves the SHA
currently served by Streamlit Cloud.

**DEPLOYMENT SHA UNVERIFIED**

Production must show `96678747f112814ea78828e2d2442059f0f6fd72` in the admin
build/health surface after redeploy.

## Release Matrix

| Area | Status | Evidence |
|---|---|---|
| CI | PASS | Exact-SHA Smoke Checks `36302841590` |
| Dependencies | PASS | Strict audit: no known vulnerabilities |
| B1 Billing security | PASS | Focused tests + isolated billing CI |
| B2 Account isolation | PASS | Identity boundary and dynamic AI-cache tests |
| B3 Scanner behavior | PASS | Canonical-results regression |
| B4 First-login state | PASS | Post-login context regression |
| Plan naming | PASS | Free/Pro/Premium/Admin tests |
| Shared account card | PASS | Run 85C tests |
| Market Brief gating | PASS | Run 85D tests |
| Historical gating | PASS | Run 85E tests |
| Full entitlement model | PASS | Run 85F matrix and operation tests |
| AI trust boundary | PASS | Run 85 claim tests; live provider check still required |
| HSF Score invariant | PASS | Cross-tier rendering regression |
| Stripe Checkout | LIVE TEST REQUIRED | No live Stripe transaction |
| Stripe webhook | LIVE TEST REQUIRED | No live event delivery |
| Billing portal | LIVE TEST REQUIRED | No live customer portal session |
| Cancellation | LIVE TEST REQUIRED | No live subscription event |
| Email delivery | LIVE TEST REQUIRED | No inbox/provider delivery evidence |
| Mobile | LIVE TEST REQUIRED | No real device/browser pass in Run 86 |
| US_MARKET freshness | PASS | Latest trading-day artifact inspected |
| Production deployment | LIVE TEST REQUIRED | Production SHA not independently observed |

## Live Tests Required

- [ ] Free login: first render shows Free and correct CTA
- [ ] Pro login: first render shows Pro and correct CTA
- [ ] Premium login: first render shows Premium and no inappropriate upsell
- [ ] Admin login: first render shows Admin and no customer upsell
- [ ] Navigate every page as Free, Pro, Premium, and Admin
- [ ] Free logout to Premium; Premium logout to Free; verify zero account-state leak
- [ ] Scanner loads canonical latest US_MARKET results with populated watchlists
- [ ] Market Brief gates Free/Pro/Premium content correctly
- [ ] Stock Intelligence gates history, PreBreakout, and AI correctly
- [ ] Stripe Free to Pro checkout completes
- [ ] Stripe webhook updates entitlement without a manual refresh workaround
- [ ] Pro to Premium upgrade completes
- [ ] Billing portal opens the correct Stripe customer
- [ ] Cancellation/downgrade updates gates and clears Premium session content
- [ ] Create/fire an alert and verify plan limits
- [ ] Deliver a real email and verify tier-appropriate content
- [ ] Test mobile navigation and Today at 375/390/430px
- [ ] Test mobile Scanner cards/lenses at 375/390/430px
- [ ] Test mobile Market Brief and Stock Intelligence at 375/390/430px
- [ ] Verify latest US_MARKET snapshot appears in the deployed app
- [ ] Verify production startup/redeploy and admin SHA equals the release candidate

To convert this verdict to GO, Howard must complete exactly these launch-critical
groups:

1. Verify all four live accounts on first render, page navigation, entitlement
   boundaries, and both cross-account transitions.
2. Complete the live Stripe checkout, webhook propagation, upgrade, portal, and
   cancellation/downgrade round trips.
3. Confirm one real alert/email delivery and Premium AI provider response with
   tier-appropriate content.
4. Complete the 375/390/430px mobile flow for Today, Scanner, Market Brief, Stock
   Intelligence, My Stocks, authentication, and Billing.
5. Redeploy and verify the production admin build SHA, production secrets/health,
   and current US_MARKET snapshot.

## Remaining Product Backlog

### LAUNCH BLOCKERS

NONE found in code. The live checks above are release conditions, not reproduced
code defects.

### POST-LAUNCH P1

- Accumulate and publish honest forward effectiveness evidence; current predictive
  effectiveness remains unvalidated/early.
- Validate multi-user Neon/load behavior before materially scaling paid access.
- Confirm production secret rotation, legal review, and security testing before a
  broad public launch, as already documented in `docs/LAUNCH_CHECKLIST.md`.

### POST-LAUNCH P2

- Monitor and reduce the documented maturation backlog/capacity warning.
- Reconcile the stale Streamlit 1.49.1 pin in `pyproject.toml` with the authoritative
  1.54.0 deployment lock to avoid developer-environment ambiguity.
- Continue documenting provider-failure/failover and Neon restore operations.

### OPTIONAL / FUTURE

NONE. No speculative feature backlog is introduced by Run 86.

## Launch Blockers

No verified code, security, scoring, entitlement, scanner, or latest-trading-day
operational blocker exists. Live external integration and deployment verification
remain mandatory before changing the verdict to GO.

## Final Verdict

# CONDITIONAL GO
