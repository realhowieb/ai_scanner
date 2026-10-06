# 🚀 Go-Live Checklist

The **code is production-ready** and `main` == `dev`. This is the config +
verification checklist for flipping the switch to real users.

---

## 1. Streamlit Cloud secrets (Settings → Secrets)

```toml
# --- Database (required) ---
NEON_DATABASE_URL = "postgresql://..."
AI_SCANNER_SQLITE_FALLBACK = "false"        # never silently fall back to sqlite in prod

# --- Sessions / auth ---
COOKIE_PASSWORD = "..."                      # required for session restore after Stripe redirect
APP_BASE_URL = "https://hsfinestai.streamlit.app"   # production app; the BETA app (hsf-beta) sets "https://hsf-beta.streamlit.app"

# --- Email (password reset, verification, digests; Resend over SMTP) ---
SMTP_HOST = "smtp.resend.com"
SMTP_PORT = "587"
SMTP_USER = "resend"
SMTP_PASS = "re_..."                         # Resend API key
SMTP_FROM = "alerts@ai.hsfinest.com"
# SMTP_FROM_NAME = "HSF Alerts"              # optional display name (default "HSF Alerts")

# --- Billing service ---
BILLING_API_BASE = "https://ai-scanner-h2c8.onrender.com"

# --- AI features ---
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-haiku-4-5"         # recommended: ~5x cheaper; unset = claude-opus-4-8 (config.py default)
AI_ENABLED = "1"                             # set "0" to kill all AI instantly, no redeploy
AI_DAILY_LIMIT = "25"                        # per-user AI calls / 24h (0 = unlimited)

# --- Alerting (optional) ---
SLACK_WEBHOOK_URL = "..."                    # scan-error spike alerts
ALERT_EMAIL = "..."                          # email fallback if Slack absent

# --- Watchlist digests (optional; off by default) ---
# WATCHLIST_ALERTS_ENABLED = "1"             # enable scheduled per-user email digests
```

## 2. Render (billing service) env vars

```
STRIPE_SECRET_KEY            # LIVE key (sk_live_...) at launch
STRIPE_WEBHOOK_SECRET        # LIVE webhook signing secret
STRIPE_PRICE_PRO            # LIVE price id, $25 / month
STRIPE_PRICE_PREMIUM        # LIVE price id, $40 / month
STRIPE_PRICE_PRO_YEARLY     # LIVE price id, $250 / year (set both yearly ids or neither)
STRIPE_PRICE_PREMIUM_YEARLY # LIVE price id, $400 / year
STRIPE_PRICE_PRO_LEGACY     # optional: older LIVE price ids still billed (comma-separated); none at first launch
STRIPE_PRICE_PREMIUM_LEGACY # optional, same
DATABASE_URL                # same Neon DB
APP_SUCCESS_URL  = https://hsfinestai.streamlit.app
APP_CANCEL_URL   = https://hsfinestai.streamlit.app
APP_PORTAL_RETURN_URL = https://hsfinestai.streamlit.app/billing

# Real-time price-alert worker (runs inside the billing service)
REALTIME_ALERTS_ENABLED = 1  # "0" or unset = worker off
REALTIME_POLL_SECONDS = 60   # optional, default 60
ALPACA_API_KEY_ID           # same keys as Streamlit / GitHub
ALPACA_API_SECRET_KEY
SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / SMTP_FROM   # same Resend values as section 1
SMTP_FROM_NAME              # optional
```

The GitHub **Realtime Alerts** job (cron-job.org, every 5 minutes on weekdays)
checks the same alerts with GitHub's own secrets, so alerts keep firing when the
Render service sleeps. Both mark `user_alerts.last_fired_at`, so an alert never
fires twice.

## 2b. Render (HSF API service) env vars

Separate Render web service (`hsf-api`), set up per [docs/API.md](docs/API.md):

```
DATABASE_URL       # same Neon DB
API_JWT_SECRET     # 32+ random characters; only on this service, never in chat or git
API_CORS_ORIGINS   # comma-separated exact https origins of the web frontend (no wildcard/path); empty = no browser access
SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / SMTP_FROM   # same Resend values; sign-up and reset emails
ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY                   # same keys as Streamlit; needed for custom scans (POST /v1/scans)
API_SCAN_WORKERS   # optional, default 1: custom scans run at once (memory-bound)
```

- [ ] Health check path `/healthz`
- [ ] Move off the free plan (it sleeps after 15 idle minutes) before the app has real users

## 3. Stripe dashboard

- [ ] **Switch from Test mode → Live mode** ⚠️ (the #1 launch step)
- [ ] Use **live** publishable/secret keys and **live** price IDs
- [ ] Webhook endpoint → `https://ai-scanner-h2c8.onrender.com/webhook` (the service has no `/stripe/webhook` route; that path would return 404 and paid users would not be upgraded)
- [ ] Webhook events subscribed:
  - `checkout.session.completed`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
- [ ] Settings → Billing → **Customer portal → Cancellations**: "At end of billing period" (P2-49; set it in live mode too)
- [ ] Set a **monthly spend alert** in the Anthropic console as an AI-cost backstop

## 3b. Billing pre-flight (after the live env vars are saved on Render)

1. Render → Environment: add `BILLING_DEBUG_STATUS = 1` and let it redeploy.
2. Open `https://ai-scanner-h2c8.onrender.com/debug/status` and look at `billing_preflight`:
   - `stripe_mode` must say `"live"`.
   - `ok` must be `true` and `problems` empty. It flags any price that is missing, in the
     wrong mode (a test price with the live key), archived, the wrong amount or interval,
     a yearly pair with only one id set, and a webhook endpoint that isn't enabled at
     `/webhook` or misses one of the three events.
   - A `warnings` line about webhooks means the key can't list endpoints; check them by hand.
3. Remove `BILLING_DEBUG_STATUS` again (the page is hidden without it).

The page shows only whether values are set and what is wrong; it never shows keys or price ids.

---

## 4. Final smoke test (do it on the live app, ~5 min)

- [ ] Sign up with a **real** email → verification email arrives
- [ ] Log in → run a scan → "✨ Generate AI summary" returns a result (confirms ANTHROPIC_API_KEY)
- [ ] Do a **real** small upgrade with a live card → land back as Pro, **no re-login**
- [ ] `curl https://ai-scanner-h2c8.onrender.com/health` → shows `"features": [...]`
- [ ] Render → Settings → Health Check Path = `/healthz` (liveness only; `/health` checks the database and would keep Neon awake 24/7)
- [ ] Trigger a password reset → reset email arrives and works

---

## 5. Emergency levers (know these before launch)

| Situation | Lever |
|---|---|
| AI cost spike / key leak | Set `AI_ENABLED = "0"` in Streamlit secrets → all AI off, no redeploy |
| Per-user AI abuse | Lower `AI_DAILY_LIMIT` |
| Roll back the app | Redeploy a previous commit from the Streamlit Cloud dashboard |
| Billing broken | Render dashboard → check `/health` and webhook delivery logs |
| Disable digests | `WATCHLIST_ALERTS_ENABLED = "0"` (or leave unset) |

---

## Post-launch (first 2 weeks)

- [ ] Watch **Admin → Diagnostics → AI feature usage** to see what users actually use
- [ ] Watch scan-error Slack alerts
- [ ] Consider adding error monitoring (Sentry) — the one real ops gap
- [ ] Once stable, `pip freeze` from the deploy env → exact-version lockfile

🤖 Generated with [Claude Code](https://claude.com/claude-code)
