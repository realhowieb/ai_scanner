import datetime as dt
import unittest

import pandas as pd

from analytics.signal_outcomes import score_signal


class SignalOutcomeTests(unittest.TestCase):
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
