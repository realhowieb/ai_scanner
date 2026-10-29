# Go-live checklist

How production is set up today, and what to check before and after a release. Updated
2026-10-29 (the Streamlit app and the separate billing service are gone; older steps that
pointed at them were removed). Only setting names are listed here, never values.

Unchecked boxes are still open.

---

## 1. What runs where

| Piece | Where | Deploys from |
|---|---|---|
| Web app (Next.js), app.hsfinest.com | Render `hsf-web-v1` (Virginia, Starter) | `main` |
| API (FastAPI), billing embedded, alert worker | Render `hsf-api-v1` (Virginia, Starter) | `main` |
| Database | Neon, branch `live` (AWS us-east-1) | — |
| Scans, digests, maturation, health jobs | GitHub Actions on `main`, started by cron-job.org | `main` |
| Payments | Stripe, live mode since 2026-10-28 | — |

Work goes into `dev`; a promote PR (`dev` → `main`) releases it. The **Live acceptance**
workflow runs the API checks and the browser journey after each promote.

## 2. Settings by service (names only)

**hsf-api-v1:** `DATABASE_URL`, `API_JWT_SECRET`, `APP_ENCRYPTION_KEY`, `HSF_PROXY_SECRET`,
`APP_BASE_URL` (= https://app.hsfinest.com), `APP_RETURN_HOSTS`, `API_CORS_ORIGINS`,
`ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY`, `ANTHROPIC_API_KEY`, `SMTP_HOST` / `SMTP_PORT` /
`SMTP_USER` / `SMTP_PASS` / `SMTP_FROM`, `BILLING_MODE=embedded`, `STRIPE_SECRET_KEY` (live),
`STRIPE_WEBHOOK_SECRET` (live), `STRIPE_PRICE_PRO` / `STRIPE_PRICE_PREMIUM` and their `_YEARLY`
ids, `REALTIME_ALERTS_ENABLED=1`, `SENTRY_DSN`, `TURNSTILE_SECRET_KEY`, `TELEGRAM_*`,
`HSF_SUPPORT_EMAIL`, `HSF_GOVERNING_LAW`, `REDIS_URL`.

**hsf-web-v1:** `HSF_API_BASE_URL`, `HSF_PROXY_SECRET` (same value as the API),
`NEXT_PUBLIC_POSTHOG_KEY` / `NEXT_PUBLIC_POSTHOG_HOST`, `NEXT_PUBLIC_TURNSTILE_SITE_KEY`,
`NODE_VERSION`, `SENTRY_AUTH_TOKEN`.

**GitHub Actions secrets:** `DATABASE_URL`, Alpaca keys, SMTP values, `ANTHROPIC_API_KEY`
(digest AI rundown), plus the test-account secrets the Live acceptance journey uses.

A custom-domain change must update `APP_BASE_URL`, `API_CORS_ORIGINS` and
`APP_RETURN_HOSTS` together.

## 3. Stripe (live)

- [x] Live keys and the four live price ids on hsf-api-v1 (Pro $25/mo, $250/yr; Premium $40/mo, $400/yr)
- [x] Webhook `https://hsf-api-v1.onrender.com/billing/webhook` with `checkout.session.completed`,
      `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.paid`,
      `invoice.payment_failed`, `charge.refunded`, `refund.failed`
- [x] Customer portal cancels at the end of the billing period
- [x] Real purchase verified end to end (2026-10-28); Admin → Health → Billing diagnostics shows mode live, sync errors 0
- [ ] Optional: add `consent_collection` / Terms URL in Stripe Checkout settings once the Terms of Service are approved

## 4. Legal pages

- [x] Review `legal/terms.v1.txt` (Terms of Service) and `legal/privacy.v1.txt`; when approved,
      change `status: proposed` to `approved` and set `effective_date` (a small PR)
- [ ] Then set `HSF_LEGAL_PAGES_ENABLED=1` on hsf-api-v1. This publishes /terms and /privacy,
      links them from the sign-up box and upgrade buttons, and records each acceptance
      (sign-up and every new monthly/yearly checkout) in `user_policy_acceptances`
- [ ] Founding 100 pages and checkout have their own switches (`FOUNDING_*`); see the founding thread / `docs/` notes

## 5. Cost and safety backstops

- [ ] Anthropic console: set a monthly spend limit and alert (AI is also capped per user by `AI_DAILY_LIMIT`)
- [ ] Neon: the restore window is 1 day on the current plan. Before paid users grow, raise it
      (paid plan) so a problem noticed after a weekend can still be undone
- [ ] Neon: mark branch `live` as protected
- [ ] Run the **Database restore drill** workflow once (Actions → Database restore drill → Run) and
      note the timings in `docs/PROVIDER_FAILOVER_AND_RESTORE.md`; repeat monthly

## 6. After each promote

The Live acceptance workflow does most of this. By hand, when it matters:

- [ ] `https://hsf-api-v1.onrender.com/readyz?strict=true` is `ok` (cron-job.org pings it every 15 minutes)
- [ ] `https://hsf-api-v1.onrender.com/billing/health` is `ok`
- [ ] Sign up with a real address → the verification email arrives; a password reset works
- [ ] Sentry shows no new issues for the release

## 7. Emergency levers

| Situation | Lever (hsf-api-v1 env, then redeploy unless noted) |
|---|---|
| AI cost spike or key leak | `AI_ENABLED=0` turns all AI off; lower `AI_DAILY_LIMIT` for abuse |
| Rule alerts misfiring | `HSF_ALERT_RULES_ENABLED=0` |
| Real-time alert worker | `REALTIME_ALERTS_ENABLED=0` |
| Day Trader warmers / live enrich using too much memory | `HSF_WARM_DAY_TRADER=0`, `PREBREAKOUT_LIVE_ENRICH=0` |
| Lifecycle emails | `HSF_LIFECYCLE_EMAILS_ENABLED=0` |
| Scan watchdog | `HSF_SCAN_WATCHDOG_ENABLED=0` |
| Roll back a release | Render → service → Deploys → redeploy the previous `main` commit (both services) |
| Database damaged | `docs/PROVIDER_FAILOVER_AND_RESTORE.md` → "Neon: restoring the live database" |
| Billing broken | `/billing/health`, Stripe → Developers → Webhooks delivery log, Admin → Health → Billing diagnostics |

Other provider outages (Alpaca, email, cron-job.org, GitHub, Stripe, Anthropic):
`docs/PROVIDER_FAILOVER_AND_RESTORE.md`.
