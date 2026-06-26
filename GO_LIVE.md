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
APP_BASE_URL = "https://hsf-beta.streamlit.app"

# --- Email (password reset, verification, digests) ---
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "587"
SMTP_USER = "..."
SMTP_PASS = "..."
SMTP_FROM = "..."

# --- Billing service ---
BILLING_API_BASE = "https://ai-scanner-h2c8.onrender.com"

# --- AI features ---
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-haiku-4-5"         # ~5x cheaper than Opus for this use
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
STRIPE_PRICE_PRO            # LIVE price id
STRIPE_PRICE_PREMIUM        # LIVE price id
DATABASE_URL                # same Neon DB
APP_SUCCESS_URL  = https://hsf-beta.streamlit.app
APP_CANCEL_URL   = https://hsf-beta.streamlit.app
APP_PORTAL_RETURN_URL = https://hsf-beta.streamlit.app
```

## 3. Stripe dashboard

- [ ] **Switch from Test mode → Live mode** ⚠️ (the #1 launch step)
- [ ] Use **live** publishable/secret keys and **live** price IDs
- [ ] Webhook endpoint → `https://ai-scanner-h2c8.onrender.com/stripe/webhook`
- [ ] Webhook events subscribed:
  - `checkout.session.completed`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
- [ ] Set a **monthly spend alert** in the Anthropic console as an AI-cost backstop

---

## 4. Final smoke test (do it on the live app, ~5 min)

- [ ] Sign up with a **real** email → verification email arrives
- [ ] Log in → run a scan → "✨ Generate AI summary" returns a result (confirms ANTHROPIC_API_KEY)
- [ ] Do a **real** small upgrade with a live card → land back as Pro, **no re-login**
- [ ] `curl https://ai-scanner-h2c8.onrender.com/health` → shows `"features": [...]`
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
