import unittest

try:
    import pandas as pd
    import streamlit as st
except Exception:  # pragma: no cover - optional in minimal envs
    pd = None
    st = None

if pd is not None and st is not None:
    import db.earnings as db_earnings
    from ui.earnings_results import prepare_results_with_earnings
    from ui.results import _format_earnings_for_display


@unittest.skipIf(pd is None or st is None, "pandas and streamlit are required for earnings results tests")
class EarningsResultsTests(unittest.TestCase):
    def setUp(self):
        for key in list(st.session_state.keys()):
            if str(key).startswith("earn") or str(key).startswith("results_") or str(key).startswith("scan_"):
                st.session_state.pop(key, None)

    def test_basic_users_do_not_run_earnings_enrichment(self):
        calls = []
        st.session_state["enable_earnings_enrichment"] = True
        df = pd.DataFrame({"Ticker": ["AAPL"]})

        def enrich(_df):
            calls.append(_df)
            raise AssertionError("Basic users should not enrich earnings")

        result, scan_ran_at = prepare_results_with_earnings(
            df,
            flags={"can_earnings": False},
            earn_col_days="earnings_in_days",
            add_earnings_days_column=enrich,
        )

        self.assertEqual(calls, [])
        self.assertIs(result, df)
        self.assertIsNotNone(scan_ran_at)

    def test_enabled_pro_user_gets_earnings_columns(self):
        st.session_state["enable_earnings_enrichment"] = True
        df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})

        def enrich(work):
            out = work.copy()
            out["earnings_in_days"] = [2, None]
            return out

        result, _scan_ran_at = prepare_results_with_earnings(
            df,
            flags={"can_earnings": True},
            earn_col_days="earnings_in_days",
            add_earnings_days_column=enrich,
        )

        self.assertIn("earnings_in_days", result.columns)
        self.assertIn("Earnings", result.columns)
        self.assertEqual(result["earnings_in_days"].iloc[0], 2)

    def test_cached_enriched_results_are_reused(self):
        st.session_state["enable_earnings_enrichment"] = True
        df = pd.DataFrame({"Ticker": ["AAPL"]})

        first, _scan_ran_at = prepare_results_with_earnings(
            df,
            flags={"can_earnings": True},
            earn_col_days="earnings_in_days",
            add_earnings_days_column=lambda work: work.assign(earnings_in_days=[4]),
        )

        def fail_if_called(_work):
            raise AssertionError("Cached earnings results should be reused")

        second, _scan_ran_at = prepare_results_with_earnings(
            pd.DataFrame({"Ticker": ["AAPL"]}),
            flags={"can_earnings": True},
            earn_col_days="earnings_in_days",
            add_earnings_days_column=fail_if_called,
        )

        self.assertEqual(first["earnings_in_days"].iloc[0], 4)
        self.assertEqual(second["earnings_in_days"].iloc[0], 4)

    def test_db_earnings_join_uses_ticker_when_date_column_missing(self):
        original_loader = db_earnings.load_earnings_details_map
        try:
            db_earnings.load_earnings_details_map = lambda _symbols: {
                "AAPL": (pd.Timestamp.today().date(), "BMO")
            }
            df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})

            result = db_earnings.add_earnings_days_column(df.copy())

            self.assertIn("earnings_date", result.columns)
            self.assertIn("earnings_time", result.columns)
            self.assertIn("earnings_in_days", result.columns)
            self.assertEqual(result["earnings_in_days"].iloc[0], 0)
        finally:
            db_earnings.load_earnings_details_map = original_loader

    def test_earnings_display_uses_dash_for_unknown_values(self):
        df = pd.DataFrame({"Ticker": ["AAPL"], "earnings_in_days": [None], "Earnings": [None]})

        result = _format_earnings_for_display(df)

        self.assertEqual(result["earnings_in_days"].iloc[0], "—")
        self.assertEqual(result["Earnings"].iloc[0], "—")


if __name__ == "__main__":
    unittest.main()
