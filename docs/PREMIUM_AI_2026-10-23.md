# Premium AI tools (2026-10-23)

All five are Premium (`can_ai_notes`) and go through `ui.ai.ask_claude` / `ask_claude_chat`,
so `AI_ENABLED=0` turns them all off. Code: `api/ai_premium.py`.

| Feature | Where | Endpoint | Cost guard |
|---|---|---|---|
| Ask HSF | `/ask` (More menu) | `POST /v1/ai/ask` | counts toward `AI_DAILY_LIMIT` (25/day) |
| Journal coach | Journal page card | `POST /v1/ai/journal-coach` | daily limit; cached per journal state; needs 5 closed trades |
| Scan builder | Custom scan, "Describe your scan" | `POST /v1/ai/scan-builder` | daily limit; fills the form only, nothing runs |
| "Why" line on alerts | alert-rule events (in-app, email, push, chat) | alert worker on hsf-api-v1 | own cap `HSF_ALERT_AI_WHY_DAILY` (default 10/account/day), 20 per pass; does not use the 25/day budget |
| AI rundown in the morning email | digest card for Premium/admin | runs in the scheduled-scans workflow | shares the watchlist note cache; daily limit |

Kill switches: `HSF_ALERT_AI_WHY=0` (alert line), `HSF_DIGEST_AI_RUNDOWN=0` (email card).

The morning digest runs on GitHub Actions, so the email card needs the
`ANTHROPIC_API_KEY` repository secret (GitHub → Settings → Secrets and variables →
Actions). Without it the card is skipped and the email goes out as before.

Rough cost on the default model (Opus 4.8, $5 / $25 per million tokens): Ask HSF 3-8¢ a
question, journal coach ~3¢, scan builder ~1¢, alert line ~1¢, email rundown ~2¢ per
Premium user per day.
