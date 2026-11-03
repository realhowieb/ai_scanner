# Premium automations (2026-11-03)

Six things Premium members used to press a button for now happen on their own. All are
behind the Premium flag `can_auto_research` (domain/plans.py) and listed on the pricing card.

| # | What | When | Where |
|---|------|------|-------|
| 1 | The Brief's AI summary is written once per market scan and shows on the Brief without a click | Trading days 08:00-18:00 ET, after each scan | `services/premium_auto.run_brief`, `GET /v1/ai/brief-narrative?saved=true` |
| 2 | Sunday email: an AI rundown of each watchlist (up to 5); the Watchlists page shows the last rundown on load | Sunday from 21:00 UTC | `run_watchlist_email`, `GET /v1/ai/watchlists/{id}/note` |
| 3 | AI notes for stocks on Premium watchlists that newly ranked; the stock page shows them on load | After each market scan, at most 15 per scan | `run_notes`, `GET /v1/ai/notes/{ticker}` |
| 4 | Friday email: AI journal coaching, only when the journal changed since the last note; the Journal shows the last note on load | Friday from 21:30 UTC | `run_journal_email`, `GET /v1/ai/journal-coach` |
| 5 | A short AI read of the top results in scheduled custom scan emails | With each scheduled scan | `services/scheduled_scans.ai_summary` |
| 6 | Logging a trade (or a paper fill) adds price alerts at the Trade plan stop and first target (1.5R); they're removed when the last open trade in that stock closes or is deleted | On log / close / delete | `api/trade_alerts.py`, journal rows show `auto_alerts` |

Saved answers live in `hsf_premium_auto` (db/premium_auto.py; removed with the account).
The GET endpoints above never call the AI, so opening a page costs nothing.

## Cost and load

- Jobs run on the API's existing 5-minute ops pass (api/ops_jobs.py), in small batches, with
  no new service. The heavy inputs (market observation, brief) are the cached ones the app
  already builds, so memory stays flat.
- AI calls: 1 per scan for the Brief (shared), up to 15 per scan for stock notes (shared by
  everyone), 1 per watchlist per member per week, 1 per member per week for coaching (only
  when trades changed), 1 per scheduled scan. Weekly emails and scheduled-scan summaries
  count toward the member's own daily AI limit; Brief and stock notes don't.

## Switches (default on; set to 0 to stop)

`HSF_PREMIUM_AUTO_ENABLED` (all of 1-4), `HSF_AUTO_BRIEF`, `HSF_AUTO_NOTES`,
`HSF_AUTO_WATCHLIST_EMAIL`, `HSF_AUTO_JOURNAL_EMAIL`, `HSF_SCHEDULED_SCAN_AI`,
`HSF_AUTO_TRADE_ALERTS`. Tuning: `HSF_AUTO_NOTES_PER_SCAN` (15), `HSF_AUTO_MAX_USERS` (300).
AI is also off whenever `AI_ENABLED` is off. Weekly emails respect the "Weekly recap" email
preference.
