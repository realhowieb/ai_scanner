# Secret rotation runbook (P1-45)

How to replace every credential HSF uses, where each one lives, and how to check
that the new one works. Secrets are kept in **four** places, and a rotation is
only finished when every place that holds the secret has the new value:

| Store | Where | Who reads it |
|---|---|---|
| **Streamlit** | share.streamlit.io → app → Settings → Secrets | the web app |
| **GitHub** | repo → Settings → Secrets and variables → Actions | scheduled scans and other jobs |
| **Render** | dashboard → billing service → Environment | the billing service (deploys from `dev`) |
| **cron-job.org** | each job → Advanced → Headers | the dispatch calls that start the scheduled jobs |

## When to rotate

- **Right away** if a secret may have leaked: it showed up in a log, a screenshot,
  a chat, a commit, or someone who had it no longer should. Rotate first, then
  look into what happened.
- **On a schedule:** every 6 months for the database password and the Stripe,
  Alpaca and SMTP keys. Once a year for the GitHub token used by cron-job.org
  (or make it expire after a year when you create it).
- **When people change:** remove the person from each provider (Neon, Stripe,
  Render, Streamlit, GitHub, cron-job.org, Resend), then rotate anything they
  could have seen.

Keep a dated note of each rotation (which secret, where it was changed, and
whether it checked out). Never paste secret values into that note, into chat, or
into an issue.

## Inventory

| Secret | Streamlit | GitHub | Render | cron-job.org | Notes |
|---|:-:|:-:|:-:|:-:|---|
| `DATABASE_URL` (Neon) | ✓ | ✓ | ✓ | | Same database, three copies. Workflows pass it on as `NEON_DATABASE_URL` too; there is no separate secret. |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | ✓ | ✓ | ✓ | | Market data. `ALPACA_DATA_URL` / `ALPACA_FEED` are settings, not secrets. |
| `STRIPE_SECRET_KEY` | | | ✓ | | Billing API key. |
| `STRIPE_WEBHOOK_SECRET` | | | ✓ | | Signs Stripe → billing webhooks. |
| `SMTP_USER`, `SMTP_PASS` (Resend) | ✓ | ✓ | ✓ | | Email. `SMTP_HOST/PORT/FROM` are settings. |
| `COOKIE_PASSWORD` | ✓ | | | | Encrypts the sign-in cookie. **See the warning below.** |
| `APP_ENCRYPTION_KEY` | ✓ | | | | Encrypts users' saved Alpaca paper keys. **See the warning below.** |
| `FMP_API_KEY`, `FINNHUB_API_KEY` | ✓ | ✓ | | | Earnings calendar. |
| `SENTRY_DSN` | ✓ | ✓ | | | Low risk (it only lets someone send events), rotate if abused. |
| GitHub token for dispatch | | | | ✓ | A fine-grained personal access token, limited to this repo and to Actions: read and write. |
| `LOADTEST_DATABASE_URL` | | ✓ | | | Neon **branch** only (P1-44). Delete the branch and the secret when you're done. |

`GITHUB_TOKEN` / `github.token` inside workflows is created per run by GitHub
and needs no rotation.

## How to rotate each one

### Neon database password

1. Neon console → project → **Roles** → the app's role → **Reset password**.
   This ends the old password at once, so have the four places open first.
2. Copy the new connection string. Use the **pooled** one (host contains
   `-pooler`); it allows far more connections than the direct one.
3. Update `DATABASE_URL` in Streamlit, GitHub and Render.
4. Check: the app signs in and shows your watchlist; Render's `/health` is OK;
   run **Scheduled Market Scans** by hand from the Actions tab and it succeeds.

Expect a minute or two of "Database temporarily unavailable" between step 1 and
step 3. Do it outside market hours.

### Alpaca

1. Alpaca dashboard → API keys → generate a new key pair.
2. Update both values in Streamlit, GitHub and Render.
3. Check: a scheduled scan run by hand returns prices; then delete the old key
   in Alpaca.

### Stripe

1. **API key:** Stripe dashboard → Developers → API keys → roll the secret key.
   Stripe lets the old key keep working for a period you choose (pick 1 hour).
   Update `STRIPE_SECRET_KEY` in Render.
2. **Webhook secret:** Developers → Webhooks → the billing endpoint → roll the
   signing secret. Update `STRIPE_WEBHOOK_SECRET` in Render.
3. Check: in the Stripe dashboard, resend a recent webhook event and confirm it
   returns 200; start a checkout in test mode and confirm it opens.

### SMTP (Resend)

1. Resend → API keys → create a new key; it is the `SMTP_PASS`.
2. Update Streamlit, GitHub and Render.
3. Check: request a password reset for your own account and receive it; then
   delete the old key in Resend.

### FMP, Finnhub

Create a new key in each provider's dashboard, update Streamlit and GitHub, and
check that the next scan's earnings column is filled. Then revoke the old key.

### GitHub token for cron-job.org

1. GitHub → Settings → Developer settings → Fine-grained tokens → generate a new
   token: this repository only, **Actions: Read and write**, expiry 1 year.
2. In cron-job.org, replace the `Authorization: Bearer …` header in every job
   (the scan slots and the BTC logger).
3. Check: each job's next run shows a new run in the Actions tab. Then delete
   the old token.

### `COOKIE_PASSWORD` and `APP_ENCRYPTION_KEY` (read first)

- Changing `COOKIE_PASSWORD` **signs everyone out** once. That's acceptable after
  a suspected leak, so do it after hours.
- Users' saved Alpaca paper keys are encrypted with `APP_ENCRYPTION_KEY`, **or
  with `COOKIE_PASSWORD` if `APP_ENCRYPTION_KEY` is not set**. Changing the key
  they were encrypted with makes them unreadable, and those users must enter
  their paper keys again.
- So before rotating `COOKIE_PASSWORD`, make sure `APP_ENCRYPTION_KEY` is set
  (to a separate random value). If it isn't set yet, setting it now has the
  same effect as rotating: saved paper keys need to be entered again.
- Generate values with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

## After any rotation

- The Security Scan's secret check runs on every push. If a secret ever lands in
  a commit, rotate it. Removing it from the code is not enough, because it
  stays in git history.
- Session cookies are stored only as hashes (P2-58), and account tokens are
  hashed too, so a database copy does not hand out working sessions or links.
  Rotating the database password is still the step that cuts off access to the
  data itself.
