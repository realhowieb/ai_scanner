# AI Scanner

AI Scanner is a Streamlit stock scanner with scheduled market scans, optional Neon/Postgres storage, and Stripe-backed billing helpers.

User guide:

- [HS Finest AI App Tutorial](APP_TUTORIAL.md)

## Run the Streamlit App

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Developer checks:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python scripts/streamlit_smoke.py --timeout 60
```

Optional real browser smoke flow:

```bash
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
python scripts/streamlit_browser_flow.py --timeout 60
```

The app defaults to local development settings when no secrets are configured. Production deployments should provide database, market-data, cookie, and billing secrets through the host environment or Streamlit secrets.

Required production auth setting:

- `COOKIE_PASSWORD`

Optional demo-user settings for local/test deployments:

- `ENABLE_DEMO_USERS=1`
- `DEMO_BASIC_PASSWORD`
- `DEMO_PRO_PASSWORD`
- `DEMO_PREMIUM_PASSWORD`
- `DEMO_ADMIN_PASSWORD`

## Scheduled Scans

Scheduled scans run through:

```bash
python -m scheduler.cron_runner
```

The GitHub Actions workflow uses this module path. The runner reads local universe files such as `sp500.txt` and `nasdaq.txt`, then saves scan results through the existing run storage helpers.

Useful optional environment variables:

- `NEON_DATABASE_URL` or `DATABASE_URL`
- `AI_SCANNER_SQLITE_FALLBACK=false` for production/scheduled jobs
- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`
- `ALPACA_DATA_URL`
- `CRON_NASDAQ_LIMIT`
- `CRON_MIN_GAP`
- `CRON_MIN_PRICE`
- `CRON_MAX_PRICE`
- `CRON_TOP_N`
- `CRON_PROFILE`

## Billing Service

The billing service has its own dependency file:

```bash
python -m pip install -r billing_service/requirements.txt
uvicorn billing_service.main:app --host 0.0.0.0 --port 8000
```

Required billing service environment variables:

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_PRO`
- `STRIPE_PRICE_PREMIUM`
- `DATABASE_URL`
- `APP_SUCCESS_URL`
- `APP_CANCEL_URL`

Required Streamlit app billing setting:

- `BILLING_API_BASE`

## Database Notes

Schema helpers create or upgrade the app tables at runtime. The user schema includes Stripe customer/subscription fields used by the billing service.
