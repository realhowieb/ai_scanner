"""Quick Alpaca connectivity test.

Run locally with your keys to confirm Alpaca is configured and returning data:

    ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=... python -m scripts.test_alpaca

Or, if your keys are in the environment / .env already, just:

    python -m scripts.test_alpaca

Exits 0 on success, 1 on failure. Prints a clear pass/fail per step.
"""
from __future__ import annotations

import sys

TEST_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]


def main() -> int:
    print("=== Alpaca connectivity test ===\n")

    # 1. Config present?
    try:
        from data.price_alpaca import get_alpaca_config
    except Exception as e:
        print(f"❌ Could not import Alpaca module: {e}")
        return 1

    cfg = get_alpaca_config()
    if cfg is None:
        print("❌ Alpaca is NOT configured.")
        print("   Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (env or st.secrets).")
        return 1
    print(f"✅ Config loaded. data_url={cfg.get('data_url')}, "
          f"key=…{str(cfg.get('api_key'))[-4:]}")

    # 2. Live data call?
    try:
        from data.price_alpaca import download_multi_alpaca
        data = download_multi_alpaca(
            TEST_TICKERS, period="1mo", interval="1d", prepost=False, timeout_s=20.0
        )
    except Exception as e:
        print(f"❌ Alpaca API call failed: {type(e).__name__}: {e}")
        print("   Common causes: invalid keys, no market-data subscription, network.")
        return 1

    got = {t: (df is not None and len(df) > 0) for t, df in (data or {}).items()}
    ok_count = sum(1 for v in got.values() if v)
    print(f"\n✅ API call succeeded. {ok_count}/{len(TEST_TICKERS)} tickers returned bars.")
    for t in TEST_TICKERS:
        df = (data or {}).get(t)
        if df is not None and len(df) > 0:
            last_close = df["Close"].iloc[-1] if "Close" in df.columns else "?"
            print(f"   {t}: {len(df)} bars, last close ≈ {last_close}")
        else:
            print(f"   {t}: ⚠️ no data")

    if ok_count == 0:
        print("\n❌ Connected but got NO data for any ticker — check your data subscription.")
        return 1

    print("\n🎉 Alpaca is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
