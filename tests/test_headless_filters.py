"""build_filtered_price_data must pass args correctly to the price/volume filters.

Regression: it passed the price_data dict as `tickers` and the threshold float
as the `lookup`, so every per-symbol lookup failed and the filters returned
nothing — every headless (premarket/postmarket) scan produced 0 results.
"""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_PANDAS, "filters read DataFrame Close/Volume")
class BuildFilteredPriceDataTests(unittest.TestCase):
    def _frame(self, price, vol):
        import pandas as pd

        n = 30
        return pd.DataFrame({
            "Close": [price] * n, "High": [price] * n, "Low": [price] * n,
            "Open": [price] * n, "Volume": [vol] * n,
        })

    def test_keeps_in_range_liquid_names_and_drops_the_rest(self):
        from scan.headless_common import build_filtered_price_data

        data = {
            "AAPL": self._frame(230, 50_000_000),   # keep
            "MSFT": self._frame(400, 20_000_000),   # keep
            "AVGO": self._frame(1700, 5_000_000),   # drop: price > max
            "PENNY": self._frame(2.0, 1_000_000),   # drop: price < min
            "THIN": self._frame(50, 100),           # drop: dollar vol < 2M
        }
        out = build_filtered_price_data(
            data, min_price=5.0, max_price=1000.0, min_dollar_vol=2_000_000,
        )
        self.assertEqual(sorted(out.keys()), ["AAPL", "MSFT"])
        # Frames are returned intact (same objects), not just the keys.
        self.assertIs(out["AAPL"], data["AAPL"])

    def test_mega_caps_are_not_all_filtered_out(self):
        from scan.headless_common import build_filtered_price_data

        data = {s: self._frame(p, 30_000_000) for s, p in
                [("AAPL", 230), ("MSFT", 400), ("NVDA", 200), ("AMD", 160)]}
        out = build_filtered_price_data(
            data, min_price=5.0, max_price=1000.0, min_dollar_vol=2_000_000,
        )
        self.assertEqual(len(out), 4)  # the bug returned 0


if __name__ == "__main__":
    unittest.main()
