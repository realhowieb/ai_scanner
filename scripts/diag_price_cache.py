"""Diagnostic: verify the Neon OHLCV price cache round-trips against prod.

Writes a synthetic frame for a reserved self-test symbol, reads it back, and
checks the data + DatetimeIndex survive the DB round-trip. Also checks staleness
gating. Touches only the reserved symbol row in price_data_cache (harmless — it's
a cache). Run headless via the diagnostics workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SELFTEST = "__CACHE_SELFTEST__"


def main() -> None:
    import pandas as pd

    from db import prices as p

    idx = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"])
    df = pd.DataFrame(
        {"Open": [10.0, 11, 12], "High": [11, 12, 13], "Low": [9, 10, 11],
         "Close": [10.5, 11.5, 12.5], "Volume": [1000, 2000, 3000]},
        index=idx,
    )

    print("=" * 70)
    print("Neon OHLCV price-cache round-trip")
    print("=" * 70)

    p.upsert_price_data_snapshot({SELFTEST: df})
    print("wrote self-test frame")

    cached, stale = p.get_price_data_snapshot([SELFTEST], max_age_minutes=60)
    got = cached.get(SELFTEST)
    if got is None:
        print("FAIL: frame not returned from cache")
        return
    ok_cols = list(got.columns) == ["Open", "High", "Low", "Close", "Volume"]
    ok_close = [round(x, 2) for x in got["Close"].tolist()] == [10.5, 11.5, 12.5]
    ok_index = isinstance(got.index, pd.DatetimeIndex)
    print(f"columns_ok={ok_cols}  close_ok={ok_close}  datetime_index={ok_index}")
    print(f"rows={len(got)}  index0={got.index[0]}")

    # Staleness: a 0-minute window must treat the just-written row as stale.
    _c2, stale0 = p.get_price_data_snapshot([SELFTEST], max_age_minutes=0)
    print(f"staleness_gate_ok={SELFTEST in stale0}")

    verdict = ok_cols and ok_close and ok_index
    print("\nRESULT:", "PASS ✅" if verdict else "FAIL ❌")


if __name__ == "__main__":
    main()
