# Run 73 - Paid Tier Differentiation

## Executive Summary

Run 73 clarifies HSF's value ladder without changing scoring, ranking, scanning,
research, or model behavior:

- **Free - Discover:** see what HSF sees now.
- **Pro - Monitor & investigate:** personalize monitoring and work efficiently
  with interactive results, exports, alerts, advanced scans, and history.
- **Premium - Research & workflow:** add AI-assisted research, Early Breakout
  research, custom full-market scanning, and paper-trading workflow.

Pro is the primary paid plan. Free retains HSF Score, Today, the latest scheduled
full-market opportunities, basic Stock Intelligence, watchlists, saved screens,
and basic market discovery.

## Before

### Actual capability matrix before Run 73

| Capability | Free/Basic | Pro | Premium | Admin |
|---|---:|---:|---:|---:|
| Today / latest scheduled market opportunities | Yes | Yes | Yes | Yes |
| HSF Score | Yes | Yes | Yes | Yes |
| Basic Stock Intelligence | Yes | Yes | Yes | Yes |
| Watchlists and saved screens | Yes | Yes | Yes | Yes |
| Own S&P 500 scan | Yes | Yes | Yes | Yes |
| In-app alerts | 1 | 5 | 25 | 25 |
| Email alert delivery | No | Yes | Yes | Yes |
| Interactive results and CSV export | No | Yes | Yes | Yes |
| Nasdaq / combined scans | No | Yes | Yes | Yes |
| Premarket, after-hours, unusual-volume filters | No | Yes | Yes | Yes |
| Earnings calendar and filters | No | Yes | Yes | Yes |
| Scan history / historical research | No | Yes | Yes | Yes |
| AI notes, summaries and results chat | No | No | Yes | Yes |
| Early Breakout research | No | No | Yes | Yes |
| Custom full-market scan workflow | No | No | Yes | Yes |
| Alpaca paper-trading workflow | No | No | Yes | Yes |
| Diagnostics / admin panel | No | No | No | Yes |

### Audit findings

1. The signed-up Free experience was incorrectly described as including
   "interactive charts" even though interactive row selection/charts are Pro.
2. The Free sidebar said "limited scan," contradicting the intentional free
   access to the current scheduled full-market opportunities.
3. Pro and Premium had equal CTA weight in the sidebar, obscuring Pro as the
   expected paid choice.
4. Upgrade prompts often named a tier but did not explain the attempted workflow
   or user benefit.
5. Premarket, after-hours, and unusual-volume capabilities were enforced by Tier
   attributes but absent from the centralized entitlement map.
6. Legacy `TIERS_CONFIG` plan names/monthly prices disagreed with the Run-72
   customer pricing source (`Basic $19`, `Pro $25`, `Premium $49` versus
   `Free`, `$19`, `$39`). The customer pages were already using Run 72, but the
   dormant metadata was a drift risk.
7. Watchlists and browser-local saved screens have no enforced tier limit. They
   must not be advertised as increased paid capacity.
8. The journal itself is available broadly. Premium's actual differentiator is
   the Alpaca paper-trading workflow that feeds paper activity into the journal,
   not a paywalled journal.

## Decision

The entitlement boundaries were already broadly aligned with the packaging
decision, so Run 73 did not move essential features between plans.

Changes were limited to:

- centralizing three existing Pro scan capabilities in `FEATURE_MIN_TIER`;
- correcting inaccurate Free copy;
- making Pro the featured/default paid upgrade;
- replacing generic lock messages with capability-specific explanations;
- aligning dormant tier labels/monthly metadata with customer-visible plans;
- improving labels so they describe workflows rather than implementation details.

## After

The enforced capability matrix remains the same as the table above. The change is
that customer-facing pricing, signup, Billing, sidebar, alert-limit, export,
earnings, Market Brief, AI, paper-trading, and scan-filter messages now describe
that matrix consistently.

The comparison table and landing cards continue to derive inclusion from
`ui.app_session.FEATURE_MIN_TIER` and `ALERT_LIMIT_BY_TIER`. Admin-only flags are
excluded from purchasable benefits.

## Upgrade Reasons

### Free to Pro

1. **Automated monitoring:** five alert slots plus email delivery when a saved
   condition fires.
2. **Faster investigation:** interactive result selection, charts, CSV export,
   earnings context, and historical scan research.
3. **Broader personal discovery:** own Nasdaq/combined scans plus premarket,
   after-hours, unusual-volume, and advanced filters.

### Pro to Premium

1. **AI-assisted research workflow:** setup notes, scan summaries, scan-change
   analysis, watchlist digest, natural-language screening, and results chat.
2. **Advanced setup research:** Early Breakout candidates and the custom
   full-market scanning workflow.
3. **Practice workflow:** Alpaca paper-account connection, confirmed paper orders,
   and paper activity integrated with the journal.

Premium also raises the alert limit from 5 to 25.

## Premium Verdict

**ADEQUATE**

A rational advanced user can distinguish Premium from Pro through four enforced,
user-facing capabilities: AI research, Early Breakout research, custom full-market
scanning, and paper trading. Those are real and meaningful, but they serve a
narrower advanced workflow than Pro's broadly useful monitoring/export/history
bundle. Premium is defensible, not yet as obviously compelling as Pro, and should
not be strengthened by moving essential Pro capabilities upward.

The strongest future categories, if supported by real implementation, would be
deeper validated research tooling, richer journal/workflow automation, and more
advanced alert orchestration. No speculative capability was added in Run 73.

## Upgrade Moments

| Attempt | Prompt behavior |
|---|---|
| Export / interactive results | Pro explains selection, comparison, and CSV workflow |
| Email a brief / email alert | Pro explains automatic inbox delivery |
| First alert limit | Pro identifies 5 slots plus email |
| Pro alert limit | Premium identifies the increase from 5 to 25 |
| Earnings context | Pro explains event-risk investigation |
| Nasdaq / combined scan | Pro explains broader personal discovery |
| Historical research | Pro explains comparison with prior observations |
| AI research | Premium explains notes, summaries, and results chat |
| Paper trading | Premium explains practice and journaling without real-money orders |

The core Free discovery flow has no interruptive paywall.

## Conversion Journey

1. A signed-out visitor sees the same shared Free/Pro/Premium plan cards used by
   Billing. Pro is visually marked "Most popular."
2. Signup creates a `basic` database tier, presented to users as **Free**, and
   accurately lists current-market discovery and basic intelligence.
3. Free users see Today, scheduled market opportunities, HSF Score, basic Stock
   Intelligence, watchlists, saved screens, one in-app alert, and an S&P scan.
4. A paid capability presents a specific explanation and correct required plan.
5. Billing uses the shared comparison source and offers Pro and Premium Stripe
   checkout. Pro is presented as the primary paid choice.
6. The billing service accepts only `pro` or `premium`, validates the app user,
   chooses the matching Stripe price, and uses signed webhook events to update
   the database tier idempotently.
7. Checkout returns with a restorable app session. The app polls/refetches the
   database tier, clears stale entitlement state, and rerenders newly available
   capabilities.

The code path is verified structurally and by existing billing/auth tests. A live
Stripe purchase was not performed in this development environment.

## Test Results

Run 73 adds coverage for:

- Free core discovery, HSF Score, latest opportunities, and basic Stock Intelligence;
- Pro and Premium capability sets;
- displayed alert limits;
- admin-only exclusion;
- shared signed-out/Billing pricing;
- capability-specific upgrade tier selection;
- entitlement-derived pricing;
- Pro featured-plan treatment;
- prohibited/misleading claims.

Final command results are recorded in the commit report. Live Stripe checkout and
webhook delivery remain environment-dependent and were not executed locally.
