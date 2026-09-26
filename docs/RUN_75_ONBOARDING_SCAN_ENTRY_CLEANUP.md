# Run 75 - Onboarding and Scan Entry Simplification

## Outcome

Run 75 simplifies navigation and presentation without changing scanner inputs,
scoring, ranking, entitlements, or persistence:

- **Today is the authenticated landing page.** The redirect occurs once per
  user session. An explicit shared Stock Intelligence redirect still wins.
- **The guided tour is the sole onboarding UI.** The duplicate first-run panel
  and session-only page orientation cards were retired.
- **Custom scan is one entry point.** Existing filters, fixed-universe scans,
  watchlist/single-ticker actions, and the Premium Market/Strategy/Profile
  workflow now live in one collapsed Scanner section.
- **The latest full-market scan remains the default Scanner view.** Users are
  told that a custom scan is optional, and the existing return-to-market action
  is unchanged.

## Safety Boundaries

No changes were made to:

- scan execution, universe resolution, filters, scoring, models, or ranking;
- scheduled full-market scans;
- HSF Score, statuses, alerts, or intelligence semantics;
- tier gates or custom-scan capabilities;
- Run 28's activation definition (watchlist plus personal-intelligence view).

The shared-link redirect is evaluated before the Today redirect. The landing
marker is normalized by user and cleared at logout so browser reuse cannot carry
navigation state between accounts.

## Cleanup

Removed modules with no production callers:

- `ui/components.py` - legacy run-button/table helpers;
- `ui/watchlist_alerts.py` - unused duplicate watchlist digest path;
- `ui/history.py` - legacy scan-history expander no longer rendered by the app.

Compatibility modules `ui/pages.py` and `ui/pages_main.py` remain because the
package facade and scheduler compatibility tests still reference them.
`ui/heat_strip.py` remains because its color helper still has a direct contract
test; migrating that helper is outside this cleanup.

## Verification

Tests cover once-per-user Today routing, shared-link precedence, the single
Custom scan entry, removal of redundant onboarding renderers, logout cleanup,
and dead-module removal. Existing scanner layout and rerun-read tests remain in
the regression set.
