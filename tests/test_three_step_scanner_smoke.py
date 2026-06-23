import importlib.util
import unittest
from unittest.mock import patch


PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None

if PANDAS_AVAILABLE:
    import pandas as pd

    from ui.three_step_scanner import run_scan_engine


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas is required for three-step scanner smoke tests")
class ThreeStepScannerSmokeTests(unittest.TestCase):
    def test_run_scan_engine_uses_fake_provider_path_and_strategy_filter(self):
        session_state = {
            "is_admin": True,
            "max_nasdaq_scan": 100,
            "max_combo_scan": 100,
            "top_n": 2,
            "min_price": 1.0,
            "max_price": 100.0,
            "min_gap_pct": 0.0,
            "min_gap_pct_aggressive": 0.0,
            "unusual_vol_aggressive": False,
            "price_snapshot_id": "fake-snapshot",
        }
        seen = {}

        def fake_breakout_scan(**kwargs):
            seen.update(kwargs)
            return pd.DataFrame(
                [
                    {"Ticker": "LOW", "GapPct": 0.5, "VolRel20": 1.0, "Close": 10.0},
                    {"Ticker": "HIGH", "GapPct": 3.0, "VolRel20": 2.5, "Close": 20.0},
                    {"Ticker": "DOWN", "GapPct": -2.0, "VolRel20": 3.0, "Close": 30.0},
                ]
            )

        with (
            patch("ui.three_step_scanner.st.session_state", session_state),
            patch("ui.three_step_scanner.resolve_scan_universe", return_value=["LOW", "HIGH", "DOWN"]),
            patch("ui.three_step_scanner.run_breakout_scan", side_effect=fake_breakout_scan),
            patch("ui.three_step_scanner.score_prebreakout", side_effect=lambda frame: frame),
        ):
            result = run_scan_engine("SP500", "gap_up", "aggressive", live_mode=True)

        self.assertEqual(list(result["Ticker"]), ["HIGH", "LOW"])
        self.assertEqual(seen["tickers"], ["LOW", "HIGH", "DOWN"])
        self.assertEqual(seen["snapshot_id"], "fake-snapshot")
        self.assertFalse(seen["use_cache"])
        self.assertEqual(seen["profile"], "aggressive")


if __name__ == "__main__":
    unittest.main()
