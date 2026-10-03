"""Postmarket scans annotate results with after-hours price/change (display only)."""

import math
import unittest
from unittest import mock

import pandas as pd

from scan import headless_common as hc


def _results():
    return pd.DataFrame(
        {"Ticker": ["NVDA", "AMD", "EA"], "BreakoutScore": [90.0, 80.0, 70.0], "Last": [100.0, 50.0, 0.0]}
    )


class AddAfterHoursColumnsTests(unittest.TestCase):
    def _run(self, quotes):
        with mock.patch("ui.scan_providers.get_alpaca_extended_last_prices", return_value=quotes):
            return hc.add_after_hours_columns(_results())

    def test_adds_ah_last_and_pct_change_vs_close(self):
        out = self._run({"NVDA": 105.0, "AMD": 48.0})
        self.assertEqual(out["AHLast"].tolist()[:2], [105.0, 48.0])
        self.assertEqual(out["AHPctChange"].tolist()[:2], [5.0, -4.0])

    def test_scores_and_order_unchanged(self):
        out = self._run({"NVDA": 105.0, "AMD": 48.0})
        self.assertEqual(out["Ticker"].tolist(), ["NVDA", "AMD", "EA"])
        self.assertEqual(out["BreakoutScore"].tolist(), [90.0, 80.0, 70.0])
        self.assertEqual(out["Last"].tolist(), [100.0, 50.0, 0.0])

    def test_missing_quote_or_zero_close_leaves_blank(self):
        out = self._run({"NVDA": 105.0, "EA": 12.0})
        self.assertTrue(math.isnan(out.loc[1, "AHLast"]))  # AMD: no quote
        self.assertTrue(math.isnan(out.loc[1, "AHPctChange"]))
        self.assertTrue(math.isnan(out.loc[2, "AHPctChange"]))  # EA: close 0

    def test_provider_error_fails_open(self):
        with mock.patch("ui.scan_providers.get_alpaca_extended_last_prices", side_effect=OSError("down")):
            out = hc.add_after_hours_columns(_results())
        self.assertEqual(out["Ticker"].tolist(), ["NVDA", "AMD", "EA"])
        self.assertTrue(out["AHLast"].isna().all())

    def test_empty_frame_passes_through(self):
        empty = pd.DataFrame()
        self.assertIs(hc.add_after_hours_columns(empty), empty)


class PipelineSessionTests(unittest.TestCase):
    def _pipeline(self, run_type, session_label=None):
        with (
            mock.patch("data.tradability.filter_tradable_tickers", side_effect=lambda s: s),
            mock.patch.object(hc, "fetch_headless_prices", return_value=({}, [], 0.0)),
            mock.patch.object(hc, "build_filtered_price_data", return_value={}),
            mock.patch.object(hc, "maybe_run_gap_filter"),
            mock.patch.object(hc, "run_headless_breakout", return_value=_results()),
            mock.patch("ui.scan_providers.get_alpaca_extended_last_prices", return_value={"NVDA": 101.0}) as q,
            mock.patch("ui.scan_providers.get_alpaca_premarket_quotes", return_value={}),
        ):
            df, _meta = hc.run_headless_pipeline(
                run_type,
                ["NVDA", "AMD", "EA"],
                min_price=5,
                max_price=1000,
                min_dollar_vol=1,
                use_parallel=False,
                parallel_workers=1,
                parallel_chunk=10,
                apply_gap_filter=False,
                top_n=10,
                session_label=session_label,
            )
        return df, q

    def test_postmarket_session_gets_ah_columns(self):
        df, q = self._pipeline("postmarket", session_label="postmarket")
        q.assert_called_once()
        self.assertEqual(df.loc[0, "AHPctChange"], 1.0)

    def test_regular_and_premarket_sessions_untouched(self):
        for run_type in ("regular", "premarket"):
            df, q = self._pipeline(run_type, session_label=run_type)
            q.assert_not_called()
            self.assertNotIn("AHLast", df.columns)


if __name__ == "__main__":
    unittest.main()
