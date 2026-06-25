# Deployment Notes

## Runtime dependency shape

`requirements.txt` is the Streamlit Cloud install file for the full app. It now
delegates to smaller dependency groups:

- `requirements-core.txt` for the Streamlit scanner, yfinance, DB, scheduler,
  auth, charting, and direct Yahoo endpoint parsing runtime
- `requirements-ml.txt` for ML-assisted scanner features
- `requirements-extended.txt` for optional integrations

The full app profile still includes heavier optional feature paths:

- `alpaca-py` for extended-hours/market-data paths
- `xgboost`, `scikit-learn`, and `joblib` for ML-assisted features
- `psycopg[binary]` for Neon/Postgres history storage

Because those packages affect deploy reliability, CI runs two dependency import
smoke jobs:

- core dependency import smoke: installs only `requirements-core.txt`
- full dependency import smoke: installs `requirements.txt`

Package metadata mirrors this split: base `pip install .` uses the core runtime,
while ML and extended integrations live under optional dependency groups.

`pyppeteer` and `yahoo_fin` are intentionally excluded from the deploy profiles:
they are not imported by the app, and their transitive dependencies can downgrade
`websockets` below the version required by modern `yfinance`.

## Recommended next split

Keep `requirements.txt` as the production Streamlit file. As features become
more modular, the next step is to make optional UI paths degrade cleanly when
their optional dependency group is not installed.

- `requirements-core.txt`: Streamlit, pandas, numpy, yfinance, requests, DB
- `requirements-ml.txt`: xgboost, scikit-learn, joblib
- `requirements-extended.txt`: alpaca-py
- `billing_service/requirements.txt`: billing service dependencies
- future `requirements-dev.txt`: test/lint tooling

## Scheduled scan runtime

The scheduled GitHub Actions scan job installs `requirements-core.txt`, not the
full deployment profile. Scheduled scans are headless scanner jobs, so they do
not need optional ML or extended integration packages to start.

Production scheduled scans should use Neon/Postgres and fail visibly if the
database is unavailable:

- `NEON_DATABASE_URL` or `DATABASE_URL`
- `AI_SCANNER_SQLITE_FALLBACK=false`

Alpaca market data credentials must use the same names read by the scanner:

- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`
- optional `ALPACA_DATA_URL`

## Secrets required for production (Streamlit Cloud)

Set these in **Settings → Secrets** for the Streamlit Cloud app. The app reads
them via `st.secrets` first, then falls back to environment variables.

### Database (required)
- `NEON_DATABASE_URL` — Neon/Postgres connection string
- `AI_SCANNER_SQLITE_FALLBACK=false` — prevent silent fallback to SQLite

### Authentication & admin
- `ADMIN_USERS` — comma-separated list of admin usernames, e.g. `howard,admin`
- `SESSION_TTL_DAYS` — session cookie lifetime in days (default: 14)
- `LOGIN_RATE_LIMIT_WINDOW_SEC` — lockout window in seconds (default: 600)
- `LOGIN_RATE_LIMIT_MAX_ATTEMPTS` — max failures before lockout (default: 10)

### Password reset (required for self-service password reset)
- `SMTP_HOST` — e.g. `smtp.sendgrid.net` or `smtp.gmail.com`
- `SMTP_PORT` — default 587 (TLS)
- `SMTP_USER` — SMTP login username
- `SMTP_PASS` — SMTP password or API key
- `SMTP_FROM` — From address, e.g. `noreply@yourdomain.com`
- `APP_BASE_URL` — full URL of the app, e.g. `https://hsf-beta.streamlit.app`
- `RESET_TOKEN_TTL_MINUTES` — how long reset links are valid (default: 30)

### Billing service
The billing service is a **separate FastAPI process** deployed on Render (or
similar). It must be running before users can subscribe.

- `BILLING_API_BASE` — URL of the billing service, e.g. `https://ai-scanner-h2c8.onrender.com`
- `BILLING_HEALTH_TIMEOUT` — health check timeout in seconds (default: 3)

The billing service itself requires its own env vars (set in Render dashboard):
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_PRO`
- `STRIPE_PRICE_PREMIUM`
- `DATABASE_URL` — same Neon connection string as the main app
- `APP_SUCCESS_URL` / `APP_CANCEL_URL` / `APP_PORTAL_RETURN_URL`

### Alerting (optional but recommended)
- `SLACK_WEBHOOK_URL` — Incoming Webhook URL for a Slack channel; alerts fire when scan errors exceed threshold
- `ALERT_EMAIL` — fallback alert email if Slack is not configured
- `SCAN_ERROR_ALERT_THRESHOLD` — errors in window before alerting (default: 5)
- `SCAN_ERROR_ALERT_WINDOW_MINUTES` — lookback window in minutes (default: 15)

### AI features (Premium)
- `ANTHROPIC_API_KEY` — Anthropic API key; enables AI summary, ticker deep-dive, and NL screener
- `ANTHROPIC_MODEL` — model id (default: `claude-opus-4-8`; use `claude-haiku-4-5` to cut cost ~5x)
- `AI_ENABLED` — set to `0` to instantly disable all AI features without a redeploy (default: on)
- `AI_DAILY_LIMIT` — max AI calls per user per rolling 24h (default: 25; `0` = unlimited)
- `AI_REQUEST_TIMEOUT_SECONDS` — per-request timeout so the UI never hangs (default: 30)

### Optional
- `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` — extended market data
- `SHOW_DIAGNOSTICS=1` — enable scan diagnostics panel by default
- `SESSION_TTL_DAYS` — override default 14-day session lifetime
