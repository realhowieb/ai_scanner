"""Relative strength vs SPY in the API: Scanner rows + Beating SPY filter, Day Trader vs SPY."""
from __future__ import annotations

import unittest
from unittest import mock

from ui.results_intelligence import consolidate_scanner_results


def _latest(**kw):
    from api import scans

    opps = [
        {"ticker": "AAA", "score": 90, "signals": ["breakout"], "rs_10d": 4.0, "rs_20d": 8.5, "rs_line_high": True},
        {"ticker": "BBB", "score": 85, "signals": [], "rs_10d": -1.0, "rs_20d": -2.0, "rs_line_high": False},
        {"ticker": "CCC", "score": 80, "signals": [], "rs_20d": None, "rs_line_high": None},
        {"ticker": "DDD", "score": 70, "signals": [], "rs_10d": 1.0, "rs_20d": 12.0, "rs_line_high": False},
    ]
    run = {"id": 1, "created_at": None}
    with mock.patch.object(scans, "market_runs", return_value=[run]), \
            mock.patch.object(scans, "run_opportunities", return_value=opps), \
            mock.patch.object(scans, "scan_freshness", return_value={"stale": False}), \
            mock.patch.object(scans, "run_freshness", return_value={}), \
            mock.patch.object(scans, "_iso", return_value=None):
        return scans.latest_scan({}, "premium", limit=50, offset=0, **kw)


class ScannerRelativeStrengthTest(unittest.TestCase):
    def test_scan_rows_carry_rs(self):
        rows = {r["ticker"]: r for r in _latest()["setups"]}
        self.assertEqual((rows["AAA"]["rs_10d"], rows["AAA"]["rs_20d"], rows["AAA"]["rs_line_high"]), (4.0, 8.5, True))
        self.assertIs(rows["BBB"]["rs_line_high"], False)
        self.assertIsNone(rows["CCC"]["rs_20d"])
        self.assertIsNone(rows["CCC"]["rs_line_high"])

    def test_beating_spy_keeps_only_positive_rs(self):
        r = _latest(beating_spy=True)
        self.assertEqual([s["ticker"] for s in r["setups"]], ["AAA", "DDD"])
        self.assertEqual((r["total"], r["scan_total"]), (2, 4))

    def test_sort_by_rs_puts_missing_last(self):
        self.assertEqual([s["ticker"] for s in _latest(sort="rs_20d")["setups"]], ["DDD", "AAA", "BBB", "CCC"])

    def test_consolidate_reads_scan_columns_and_score_ignores_them(self):
        base = {"Ticker": "XYZ", "BreakoutScore": 40.0, "IsBreakout": True, "EMACross": "Golden", "PctChange": 3.0}
        with_rs = consolidate_scanner_results([{**base, "RS10D%": 2.5, "RS20D%": 6.0, "RSLineHigh": True}])[0]
        without = consolidate_scanner_results([base])[0]
        self.assertEqual((with_rs["rs_10d"], with_rs["rs_20d"], with_rs["rs_line_high"]), (2.5, 6.0, True))
        self.assertIsNone(without["rs_20d"])
        self.assertEqual(with_rs["score"], without["score"])  # display only: HSF score unchanged


class DayTraderVsSpyTest(unittest.TestCase):
    def _run(self, symbols, raw):
        from api import market
        from api import today as api_today
        loads = []

        def cached(key, loader, ttl_s=0, stale_s=0):
            if key == "dt_state":
                return "open"
            if isinstance(key, tuple) and key[0] == "dt_rows":
                return raw
            return loader()

        with mock.patch.object(api_today, "_cached", side_effect=cached), \
                mock.patch.object(market, "_dt_symbols", return_value=symbols), \
                mock.patch.object(market, "_dt_earnings", return_value={}), \
                mock.patch("analytics.day_trade_display.enrich_row", side_effect=lambda r, s, n: dict(r)), \
                mock.patch("analytics.day_trade_display.rank_key", side_effect=lambda r: r["ticker"]):
            return market.day_trader("custom"), loads

    def test_vs_spy_and_spy_row_hidden(self):
        raw = [{"ticker": "NVDA", "chg_pct": 3.5}, {"ticker": "AMD", "chg_pct": -0.4}, {"ticker": "SPY", "chg_pct": 1.0},
               {"ticker": "ZZZ", "chg_pct": None}]
        out, _ = self._run(["NVDA", "AMD", "ZZZ"], raw)
        rows = {r["ticker"]: r for r in out["rows"]}
        self.assertNotIn("SPY", rows)
        self.assertEqual(rows["NVDA"]["vs_spy_pct"], 2.5)
        self.assertEqual(rows["AMD"]["vs_spy_pct"], -1.4)
        self.assertIsNone(rows["ZZZ"]["vs_spy_pct"])
        self.assertEqual(out["spy_chg_pct"], 1.0)
        self.assertEqual(out["missing"], 0)

    def test_spy_asked_for_stays_and_no_spy_quote_gives_none(self):
        out, _ = self._run(["SPY", "NVDA"], [{"ticker": "SPY", "chg_pct": 1.0}, {"ticker": "NVDA", "chg_pct": 2.0}])
        self.assertEqual({r["ticker"] for r in out["rows"]}, {"SPY", "NVDA"})
        out, _ = self._run(["NVDA"], [{"ticker": "NVDA", "chg_pct": 2.0}])
        self.assertIsNone(out["rows"][0]["vs_spy_pct"])
        self.assertIsNone(out["spy_chg_pct"])

    def test_loader_adds_spy_once(self):
        from api import market
        from api import today as api_today
        seen = []

        def cached(key, loader, ttl_s=0, stale_s=0):
            if key == "dt_state":
                return "open"
            return loader()

        with mock.patch.object(api_today, "_cached", side_effect=cached), \
                mock.patch.object(market, "_dt_symbols", return_value=["NVDA"]), \
                mock.patch.object(market, "_dt_earnings", return_value={}), \
                mock.patch("market_data.build_day_trader_metrics", side_effect=lambda s: seen.append(list(s)) or []), \
                mock.patch("analytics.day_trade_display.enrich_row", side_effect=lambda r, s, n: dict(r)), \
                mock.patch("analytics.day_trade_display.rank_key", side_effect=lambda r: r["ticker"]):
            market.day_trader("custom")
        self.assertEqual(seen, [["NVDA", "SPY"]])


if __name__ == "__main__":
    unittest.main()
