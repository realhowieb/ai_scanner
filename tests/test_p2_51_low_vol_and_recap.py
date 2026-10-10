"""P2-51 (owner: both): bond/cash ETFs don't fire market-wide breakout alerts;
the recap lists entered/left names by HSF Score with a floor. Display only."""
import datetime as dt
import unittest

import pandas as pd

from scheduler.alert_runner import MIN_ALERT_VOLATILITY_PCT, _evaluate
from ui import recap

DF = pd.DataFrame([
    {"Ticker": "IOVA", "BreakoutScore": 118.9, "Volatility20D%": 6.2, "Last": 3.0},
    {"Ticker": "MINT", "BreakoutScore": 44.9, "Volatility20D%": 0.05, "Last": 100.0},
    {"Ticker": "SPHY", "BreakoutScore": 41.9, "Volatility20D%": 0.3, "Last": 23.0},
    {"Ticker": "AIG", "BreakoutScore": 40.8, "Volatility20D%": 1.2, "Last": 80.0},
    {"Ticker": "NEW", "BreakoutScore": 50.0, "Volatility20D%": None, "Last": 5.0},
])


def tickers(lines):
    return [ln.split(":")[0] for ln in lines]


class AlertTests(unittest.TestCase):
    def test_low_volatility_names_skip_market_wide_breakout_alerts(self):
        self.assertEqual(MIN_ALERT_VOLATILITY_PCT, 1.0)
        got = tickers(_evaluate({"alert_type": "breakout", "threshold": 30}, DF, set()))
        self.assertEqual(got, ["IOVA", "NEW", "AIG"])        # missing volatility is kept; strongest first

    def test_watchlist_only_alert_keeps_what_the_user_watches(self):
        got = tickers(_evaluate({"alert_type": "breakout", "threshold": 30, "watchlist_only": True},
                                DF, {"MINT", "IOVA"}))
        self.assertEqual(got, ["IOVA", "MINT"])

    def test_watchlist_alert_unchanged(self):
        got = tickers(_evaluate({"alert_type": "watchlist"}, DF, {"MINT"}))
        self.assertEqual(got, ["MINT"])

    def test_no_volatility_column_means_no_filter(self):
        got = tickers(_evaluate({"alert_type": "breakout", "threshold": 30}, DF.drop(columns=["Volatility20D%"]), set()))
        self.assertIn("MINT", got)


class RecapTests(unittest.TestCase):
    def test_by_score_orders_and_floors(self):
        scores = {"WRBY": 69, "AIG": 47, "MINT": 28, "JPST": 38}
        from unittest import mock

        with mock.patch("ui.headline_score.hsf_scores_by_ticker", return_value=scores):
            got = recap._by_score(["MINT", "JPST", "AIG", "WRBY", "URI"], pd.DataFrame({"Ticker": []}))
        self.assertEqual(got, ["WRBY (69)", "AIG (47)"])

    def test_lines_state_the_floor(self):
        r = {"scans": 3, "entered": ["WRBY (69)"], "left": [], "standouts": []}
        text = "\n".join(recap.recap_lines(r))
        self.assertIn("Entered the ranked list (HSF 40+): WRBY (69).", text)
        self.assertIn("No names scoring HSF 40+ left the ranked list.", text)


if __name__ == "__main__":
    unittest.main()
