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
