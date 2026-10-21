# AI Scanner

AI Scanner (HSFinest.AI) is a stock scanner: a FastAPI service (`api/`, Render
hsf-api-v1), a Next.js web app (`web/`, Render hsf-web-v1), scheduled market scans
and emails on GitHub Actions, Neon/Postgres storage and Stripe billing. The old
Streamlit app was retired on 2026-10-21.

User guide:

- [HS Finest AI App Tutorial](APP_TUTORIAL.md)

## Local development

Set up the local environment (Python 3.13, the same pinned versions production
runs from `requirements.lock`, plus test tools):

```bash
bash scripts/setup_local_env.sh
```

Add `--fresh` to delete and rebuild `.venv` (for example after a Homebrew Python
upgrade). Settings come from environment variables. For a local database, point
`DATABASE_URL` at the Neon branch `dev-local` (a copy of live; refresh it with
`bash scripts/reset_dev_branch.sh`), never at `live` itself.

Outages and database restore: [docs/PROVIDER_FAILOVER_AND_RESTORE.md](docs/PROVIDER_FAILOVER_AND_RESTORE.md).

Run the API and the web app: [docs/API.md](docs/API.md) and [web/README.md](web/README.md).

Developer checks:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -m pytest tests/
```

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

Optional: `APP_RETURN_HOSTS` (default `hsf-web-v1.onrender.com`), comma-separated
hosts of other apps whose https return links checkout and the portal accept, so
upgrades started in the web app come back to it.

Billing service base URL used by the API and jobs: `BILLING_API_BASE`
(default `https://hsf-api-v1.onrender.com/billing`).

## Database Notes

Schema helpers create or upgrade the app tables at runtime. The user schema includes Stripe customer/subscription fields used by the billing service.
