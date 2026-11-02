# Premium value pack (2026-11-02)

## Pricing card
`ui/pricing.py` ROWS now lists the Premium features that were already live but hidden
behind one AI line: Ask HSF, "Describe a scan" (AI scan builder), AI watchlist rundowns
(app + morning email), the AI "why" line on alerts, the AI journal coach, Early Breakout
alerts. The web card (`PlansSection` in `web/src/features/Landing.tsx`) shows 25 alerts,
Ask HSF, Scheduled custom scans, AI scan summaries and Early Breakout first; the rest sit
under "All plan features".

## Scheduled custom scans (`can_scheduled_scans`, Premium)
- Custom scan page: "Scheduled scans" card saves the current form (name + before the open /
  midday / after the close). Up to 3 per account. Pro and Starter see a locked teaser.
- API: `GET/POST /v1/scheduled-scans`, `DELETE /v1/scheduled-scans/{id}` (delete works on
  any plan). Plan rules are checked on save and again before each run.
- Runner: `services/scheduled_scans.py`, called by `api/ops_jobs.py` every 5 minutes on
  trading days. Windows (ET): open 08:05-09:25, midday 12:00-13:30, close 16:20-18:00.
  Each schedule is claimed once per day and slot (`db/scheduled_scans.py`, table
  `hsf_scheduled_scans`), queued on the API scan pool (`api.scan_jobs`, so the one-scan
  memory bound holds), and the top 10 results are emailed (needs a verified address and
  the "alerts" email preference). A lapsed plan pauses the schedule and says so.
- Env: `HSF_SCHEDULED_SCANS_ENABLED=0` stops it; `HSF_SCHEDULED_SCANS_PER_PASS` (default 3).

## CSV export of your research (`can_export_research`, Premium)
"Download CSV" on a watchlist (Tickers card), the Journal (every page), Track record
(summary + the chosen horizon's daily series) and a saved scan in Scan history. Built in
the browser from data already on the page (`web/src/features/ResearchCsv.tsx`). Other plans
see a "CSV · Premium" link to the plans. Scanner CSV stays Pro.
