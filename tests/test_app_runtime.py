import datetime as dt
import unittest

try:
    import pandas as pd

    from ui.app_runtime import get_market_session, normalize_results_to_df
except Exception:  # pragma: no cover - optional in minimal envs
    pd = None
    get_market_session = None
    normalize_results_to_df = None


@unittest.skipIf(pd is None, "pandas and streamlit are required for app runtime tests")
class AppRuntimeTests(unittest.TestCase):
    def test_market_session_uses_eastern_session_windows(self):
        eastern = dt.timezone(dt.timedelta(hours=-4))

        self.assertEqual(
            get_market_session(dt.datetime(2026, 6, 22, 8, 0, tzinfo=eastern)),
            "premarket",
        )
        self.assertEqual(
            get_market_session(dt.datetime(2026, 6, 22, 10, 0, tzinfo=eastern)),
            "regular",
        )
        self.assertEqual(
            get_market_session(dt.datetime(2026, 6, 22, 17, 0, tzinfo=eastern)),
            "afterhours",
        )
        self.assertEqual(
            get_market_session(dt.datetime(2026, 6, 21, 10, 0, tzinfo=eastern)),
            "closed",
        )

    def test_normalize_results_to_df_handles_json_records(self):
        df = normalize_results_to_df('[{"Ticker": "AAPL", "Score": 91}]')

        self.assertIsNotNone(df)
        self.assertEqual(df.loc[0, "Ticker"], "AAPL")

    def test_normalize_results_to_df_rejects_invalid_json(self):
        self.assertIsNone(normalize_results_to_df("{not json"))


if __name__ == "__main__":
    unittest.main()
