# Deployment Notes

## Runtime dependency shape

`requirements.txt` is currently the Streamlit Cloud install file for the full
app. It includes core scanner dependencies plus heavier optional feature paths:

- `alpaca-py` for extended-hours/market-data paths
- `xgboost`, `scikit-learn`, and `joblib` for ML-assisted features
- `pyppeteer` and charting packages for UI/reporting features
- `psycopg[binary]` for Neon/Postgres history storage

Because those packages affect deploy reliability, CI runs a dependency import
smoke job that installs `requirements.txt` and imports the main scanner modules.

## Recommended next split

Keep `requirements.txt` as the production Streamlit file until deployment is
stable, then split optional groups into separate files or extras:

- `requirements-core.txt`: Streamlit, pandas, numpy, yfinance, requests, DB
- `requirements-ml.txt`: xgboost, scikit-learn, joblib
- `requirements-billing.txt`: billing service dependencies
- `requirements-dev.txt`: test/lint tooling
