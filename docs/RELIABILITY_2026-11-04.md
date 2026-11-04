# Automation reliability and recovery — 2026-11-04

## Premium automations

- An empty Brief narrative releases its claim so the next ops pass can retry it.
- Failed stock-note generation or watchlist discovery releases the scan claim. Saved notes remain reusable, so a later pass can recover the failed notes.
- Journal email eligibility now compares the current closed-trade fingerprint with the last successfully emailed coaching note. If SMTP fails after coaching is generated, the next pass reuses the saved note and retries delivery. An unchanged, already emailed journal is skipped.
- If the stop/target pair cannot be completed (for example, only one alert slot remains), remove the alert just created and avoid recording a misleading pair in the journal. Cleanup failures are logged.
- Regression checks cover retry behavior, account isolation in saved answers, consent and verification before email work, Premium eligibility, alert limits, and scheduled-scan AI summaries.

Read-only production verification at approximately 10 AM Pacific: three saved Brief narratives, five stock notes; the latest watchlist pass at 17:46 UTC scored 188 of 198 requested stocks in 33.1 seconds. No customer emails or trades were triggered for this verification. Friday/Sunday live email delivery remains subject to its scheduled window; delivery retries were tested with simulated SMTP failures.

## Watchlist scores

A watchlist no longer combines the latest market scan with automatic scores written before that scan. Missing scores keep the quote-only presentation until a current pass arrives. Scanner and Watchlists show the auto-score timestamp; the Scanner Watchlist tab also shows the API's delayed-scan status.

## Database recovery

- Enabled and verified protection on `live` in `HSF_Stock_Finder`.
- Existing plan: Launch; verified retention: 86,400 seconds (one day).
- Seven-day retention (`604800`) is prepared, awaiting approval for history storage costs. No plan upgrade is required; old discarded history cannot be recovered by extending the setting.
- Owner-approved [restore drill](https://github.com/realhowieb/ai_scanner/actions/runs/41882862990): **87 tables, 28 MB, dump 27 seconds, restore 4 seconds, total 70 seconds; passed**. Three large research tables were checked as schema only. Production was read only; the temporary copy was deleted when the workflow ended.
- Recorded the earlier October 29 drill too, corrected retired deployment references, and marked the completed recovery tasks in the launch checklists.

## Dependency updates

Validated together: eslint-config-next 16.4.2, ESLint 10.13.0, jsdom 30.1.3, and @testing-library/user-event 14.6.8. These cover Dependabot PRs #173, #175, #174, and #172 respectively; those PRs can be closed as superseded after this change merges.

ESLint 10 installs and runs successfully with the updated Next.js configuration. npm still warns that three upstream plugins declare peer support only through ESLint 9 (import, jsx-a11y, react). No force or legacy-peer-deps setting was added. A clean-install dry run succeeds. Keep this warning visible to reviewers rather than silencing it.

## Validation

- Full Python suite: 3,406 passed, 68 skipped, 165 subtests passed. Subsequent focused checks cover the final alert-limit and email-consent regressions.
- Web suite: 571 passed before the new freshness regression; the focused Scanner Watchlist suite covers the additional test.
- Web lint, typecheck, production build, and production dependency audit passed (zero vulnerabilities). Local Node version: 26.10.0; GitHub CI uses Node 22.
- Python lint and OpenAPI contract checks cover the changed API paths. No response-schema changes are required.

Code changes target `dev`; production release remains the separate `dev` → `main` promote.
