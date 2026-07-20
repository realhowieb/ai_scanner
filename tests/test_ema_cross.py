"""Shared EMA 9/21 cross helper + scanner-column display mapping."""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS = importlib.util.find_spec("pandas") is not None
_STREAMLIT = importlib.util.find_spec("streamlit") is not None


@unittest.skipUnless(_PANDAS, "EMA cross needs pandas")
class EmaCrossDetailTests(unittest.TestCase):
    def _frame(self, closes):
        import pandas as pd

        return pd.DataFrame({"Close": closes})

    def test_fresh_golden_cross(self):
        from scan.indicators import ema_cross_detail

        d = ema_cross_detail(self._frame([100.0] * 24 + [200.0]))
        self.assertIsNotNone(d)
        self.assertEqual(d["direction"], "bullish")
        self.assertGreater(d["ema_fast"], d["ema_slow"])

    def test_fresh_death_cross(self):
        from scan.indicators import ema_cross_detail

        d = ema_cross_detail(self._frame([100.0] * 24 + [40.0]))
        self.assertEqual(d["direction"], "bearish")

    def test_no_fresh_cross_returns_none(self):
        from scan.indicators import ema_cross_detail

        self.assertIsNone(ema_cross_detail(self._frame([100.0 + i for i in range(30)])))

    def test_too_few_bars(self):
        from scan.indicators import ema_cross_detail

        self.assertIsNone(ema_cross_detail(self._frame([1, 2, 3])))

    def test_accepts_bare_series(self):
        import pandas as pd

        from scan.indicators import ema_cross_detail

        d = ema_cross_detail(pd.Series([100.0] * 24 + [200.0]))
        self.assertEqual(d["direction"], "bullish")

    @unittest.skipUnless(_STREAMLIT, "market_data imports streamlit")
    def test_callers_delegate_and_preserve_shapes(self):
        from market_data import ema_cross_label
        from scheduler.alert_runner import _ema_cross_signal

        golden = self._frame([100.0] * 24 + [200.0])
        self.assertEqual(ema_cross_label(golden), "Golden")
        sig = _ema_cross_signal(golden)
        self.assertEqual(sig["direction"], "bullish")
        self.assertIn("ema9", sig)
        self.assertIn("ema21", sig)


@unittest.skipUnless(_PANDAS and _STREAMLIT, "display mapping needs pandas + streamlit")
class EmaCrossColumnDisplayTests(unittest.TestCase):
    def test_maps_values_and_dashes_the_rest(self):
        import pandas as pd

        from ui.result_helpers import format_ema_cross_column

        out = format_ema_cross_column(
            pd.DataFrame({"Ticker": ["A", "B", "C"], "EMACross": ["Golden", "Death", None]})
        )
        self.assertEqual(list(out["EMACross"]), ["🟢 Golden", "🔴 Death", "—"])

    def test_absent_column_is_noop(self):
        import pandas as pd

        from ui.result_helpers import format_ema_cross_column

        df = pd.DataFrame({"Ticker": ["A"]})
        self.assertIs(format_ema_cross_column(df), df)


if __name__ == "__main__":
    unittest.main()
