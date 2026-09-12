import datetime as dt
import unittest
from unittest import mock

import pandas as pd

from analytics.signal_outcomes import score_signal
from db import signal_outcomes as so


class SignalOutcomeTests(unittest.TestCase):
    def test_summarize_recent_outcomes_computes_hit_rate(self):
        cur = mock.MagicMock()
        # aggregate row, then best-performer row
        cur.fetchone.side_effect = [(10, 3, 6, 7, 3, 0.062, -0.028), ("AMD", 0.13)]
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(so, "get_neon_conn", return_value=conn):
            s = so.summarize_recent_outcomes(days_back=7)
        self.assertEqual(s["completed"], 10)
        self.assertEqual(s["pending"], 3)
        self.assertEqual(s["hits"], 6)
        self.assertAlmostEqual(s["hit_rate"], 0.6)
        self.assertAlmostEqual(s["avg_winner"], 0.062)
        self.assertEqual(s["best_ticker"], "AMD")
        self.assertAlmostEqual(s["best_return"], 0.13)

    def test_summarize_recent_outcomes_zeroed_without_db(self):
        with mock.patch.object(so, "get_neon_conn", return_value=None):
            s = so.summarize_recent_outcomes()
        self.assertEqual(s["completed"], 0)
        self.assertIsNone(s["hit_rate"])

    def test_score_signal_returns_forward_windows_and_excursions(self):
        bars = pd.DataFrame(
            {
                "Close": [100.0, 102.0, 99.0, 105.0, 104.0, 110.0],
                "High": [101.0, 103.0, 100.0, 106.0, 105.0, 112.0],
                "Low": [99.0, 101.0, 97.0, 103.0, 102.0, 108.0],
            },
            index=pd.date_range("2026-01-05", periods=6, freq="B"),
        )

        outcome = score_signal(bars, dt.date(2026, 1, 5))

        self.assertEqual(outcome["return_1d"], 0.02)
        self.assertEqual(outcome["return_3d"], 0.05)
        self.assertEqual(outcome["return_5d"], 0.10)
        self.assertEqual(outcome["mfe_5d"], 0.12)
        self.assertEqual(outcome["mae_5d"], -0.03)


if __name__ == "__main__":
    unittest.main()
