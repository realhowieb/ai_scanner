# Six integrations — October 22, 2026

Every one of these is off or harmless until its settings are added, so this change can
be promoted before any keys exist. Never commit keys or paste them into chat.

| # | Integration | Where | Settings | Until set |
| --- | --- | --- | --- | --- |
| 1 | Alpaca News | `hsf-api-v1` | none (uses `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY`); `HSF_NEWS_ENABLED=0` turns it off | works now |
| 2 | Cloudflare Turnstile | `hsf-web-v1` + `hsf-api-v1` | web: `NEXT_PUBLIC_TURNSTILE_SITE_KEY` (rebuild); API: `TURNSTILE_SECRET_KEY` | no bot check |
| 3 | SEC EDGAR | `hsf-api-v1` | `HSF_SEC_USER_AGENT="HSFinest.AI you@yourdomain"` (or a valid `SMTP_FROM`); `HSF_FILINGS_ENABLED=0` turns it off | card hidden |
| 4 | Telegram / Discord alerts | `hsf-api-v1` | Telegram: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME`, `TELEGRAM_WEBHOOK_SECRET`; Discord: none. `HSF_CHAT_ALERTS_ENABLED=0` turns both off | Discord works now; Telegram hidden |
| 5 | Render Key Value | `hsf-api-v1` | `REDIS_URL` = the Key Value's **Internal** URL | each process counts in memory (as before) |
| 6 | Sentry source maps | `hsf-web-v1` | `SENTRY_AUTH_TOKEN` (org token, project:releases); optional `SENTRY_ORG` (default `hs-finest-ai`), `SENTRY_PROJECT` (default `python`) | web errors stay minified |

## 1. Alpaca News

- `GET /v1/stocks/{ticker}/news`: up to 10 headlines from the last 14 days (Benzinga feed).
  Shown as the **News** card on the stock page.
- `GET /v1/news/flags?symbols=`: tickers with a story in the last 24 hours. The Scanner
  shows a **News** marker next to those tickers (current scans only), linking to the card.
- Shared for 5 minutes through Render Key Value (or memory). All plans.

## 2. Cloudflare Turnstile

Create a widget at Cloudflare → Turnstile (domains: `hsf-web-v1.onrender.com`, plus
`app.hsfinest.com` once it exists). Mode "Managed". Set **both** keys together: with only
the secret on the API, web sign-ups would be refused. Sign-up and password reset send the
widget's token; the API checks it with Cloudflare. If Cloudflare can't be reached the
request is allowed (rate limits still apply). Sign-in is unchanged, so the daily live
acceptance journey keeps working.

## 3. SEC EDGAR

- `GET /v1/stocks/{ticker}/filings`: open-market insider buys/sales (Form 4, codes P/S)
  and 8-K events (item labels such as "Earnings results") from the last 90 days. Shown
  as the **SEC filings** card. All plans. Cached 3 hours (Form 4s for 7 days).
- New alert rule **INSIDER_BUY** ("Insider buys at least ($)", 0 = any), Pro
  (`can_insider_alerts`). The first pass records the newest buy as a baseline, so a new
  rule never fires for older filings. Checked after each market scan.
- EDGAR requires a contact in the User-Agent; without one, filings stay off.

## 4. Telegram and Discord alerts (Pro)

Alerts → Delivery → **Telegram and Discord**. Every alert that fires (price alerts and
alert rules) also posts to connected chats, except during quiet hours.

- Discord: the user pastes a channel webhook URL; a test message is posted first. The URL
  is never returned by the API.
- Telegram setup (once): message @BotFather → `/newbot` → copy the token. Set the three
  `TELEGRAM_*` variables on `hsf-api-v1` (`TELEGRAM_WEBHOOK_SECRET` = 32+ random letters
  and digits), then run
  `python scripts/telegram_set_webhook.py https://hsf-api-v1.onrender.com` with the same
  token and secret. Users tap **Connect Telegram**, which opens the bot with a one-time
  code (15 minutes); `/stop` in the chat disconnects.
- Tables `hsf_chat_destinations` and `hsf_chat_link_codes` are created on first use.

## 5. Render Key Value

Create a Key Value instance in **Virginia** (same region as `hsf-api-v1`). The free plan
is enough: it holds rate-limit windows and short caches, and losing them on a restart is
fine. Copy its **Internal URL** into `REDIS_URL` on `hsf-api-v1`. Then:

- Per-IP and per-account rate limits are shared by every API instance and survive deploys.
- News and EDGAR lookups are shared across instances.
- `/readyz` shows `key_value: {configured, ok, ping_ms}` (informational, never a 503).
- If Key Value stops answering, the API falls back to memory for 30 seconds at a time.

## 6. Sentry source maps

Browser crash reports already reach Sentry through `POST /v1/client-errors`. They now carry
the web build's commit (`RENDER_GIT_COMMIT`), and reports with a stack are sent as JavaScript
exceptions tagged `release: web@<commit>`. With `SENTRY_AUTH_TOKEN` set on `hsf-web-v1`, the
build makes source maps, uploads them to that release (`web/scripts/sentry-sourcemaps.mjs`)
and deletes them from the public output. An upload failure never fails the build.

## Verification

- Backend: `tests/test_integrations_2026_10_22.py` (22 tests, no network); full suite
  2908 passed. The 6 failures in `test_billing_preflight.py` / `test_freshness_operations.py`
  also fail on `dev` without this change.
- Web: 401 tests, typecheck, lint and production build pass; a build with a fake Sentry token
  failed the upload gracefully and left no `.map` files in `.next/static`.
- Not verified against the live providers from the sandbox (its proxy blocks sec.gov and
  Alpaca); check the stock page cards after deploy.
