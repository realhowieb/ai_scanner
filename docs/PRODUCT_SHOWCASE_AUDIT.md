# HSF Product Showcase Audit

## Scope and guardrails

Audited the current Streamlit implementation on `dev`: Market Brief, Scanner
results, Day Trader, PreBreakout, My Watchlist, Alerts, Stock Intelligence, and
Historical Replay. `HSF_SCREENSHOT_MODE` changes rendering only. It supplies no
fixtures, performs no scan, changes no DataFrame values, and writes no research
or production state.

## Page audit

| View | Primary purpose | Strongest visual | Current friction | Showcase treatment |
| --- | --- | --- | --- | --- |
| Market Brief | Explain what matters now | Market context plus ranked opportunities | Orientation/help and secondary sections can push opportunities below the fold | Shared width/spacing; collapsed sidebar; retain freshness and real context |
| Scanner / Results | Discover and rank setups | HSF Opportunities table and selected-name explanation | Full raw table, filters, diagnostics, and exports compete with the ranked view | Make the intelligence table the showcase endpoint; hide the secondary raw output in screenshot mode |
| Day Trader | Monitor live intraday structure | Dense live table with Direction, ADX, VWAP, RVOL, SuperTrend, EWO | Refresh/watch controls and wide secondary columns dilute the table | Keep source selector; hide session notification controls; display the decision-useful column subset |
| PreBreakout | Surface early setup candidates | Probability-ranked candidate table | Model administration and filters read like an internal console | Hide model training/status and controls; preserve the same 60% filter and descending ranking |
| My Watchlist | Personalized attention workflow | Attention-first intelligence cards | Management tools and stable names compete with urgent changes | Use the attention view for captures; management remains below current intelligence |
| Alerts | Show meaningful state changes | Chronological intelligence feed | Preferences/static-alert management can dominate | Capture the recent intelligence section; crop before configuration |
| Stock Intelligence | Explain one setup deeply | Canonical IntelligenceView/current state | Ticker input and lower historical sections can distract | Use as the second hero candidate; current state, reasons, caution, and provenance remain visible |
| Historical Replay | Explain what HSF knew over time | Timeline plus point-in-time inspector | Requires accumulated production history and can look sparse | Use only when a symbol has a complete, legible session |

## Five strongest views

1. **Scanner / HSF Opportunities** — the clearest statement of broad discovery,
   prioritization, status, movement, setup, model context, and drill-down.
2. **Stock Intelligence** — proves that ranked names have traceable reasoning,
   cautions, separate scores, lifecycle, and freshness.
3. **Market Brief** — communicates market context and what deserves attention.
4. **Day Trader** — shows the depth of real-time technical context without
   suggesting a new score or trade instruction.
5. **My Watchlist** — demonstrates personalized monitoring and change detection.

PreBreakout is a strong supporting screenshot, but less self-explanatory than
Watchlist without adjacent copy. Alerts is valuable in a product tour but a
weak hero. Historical Replay should wait for a visually complete session.

## Consistency findings

- Existing shared page headers, HSF score language, status labels, freshness,
  and canonical IntelligenceView provide a sound visual foundation.
- The largest inconsistency was density: controls, model administration, and
  full raw tables appeared with the same weight as primary intelligence.
- Screenshot mode uses one maximum content width, tighter spacing, quiet metric
  and table borders, collapsed sidebar, and reduced Streamlit chrome.
- Unknown values remain `—` or existing unavailable copy. Screenshot mode never
  substitutes zero, Neutral, or False.
- Current timestamps and market-state captions remain visible so a capture does
  not imply live data when the market is closed or data is stale.

## Remaining risks

- Capture quality still depends on current production data and the user's
  entitlements/session. No demo data is generated.
- Streamlit authentication makes unattended capture inappropriate without a
  dedicated test account and approved credential handling.
- Tables may still require horizontal movement below 1440 px. Use the listed
  desktop viewports and avoid browser zoom below 90%.
- Historical Replay and Alerts can be sparse outside active accumulation.

