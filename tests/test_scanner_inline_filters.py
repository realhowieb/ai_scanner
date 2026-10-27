"""Scanner price and relative volume filters on /v1/scans/latest (api/scans.latest_scan)."""
from __future__ import annotations

import unittest
from unittest import mock


def _scan(**kw):
    from api import scans

    opps = [
        {"ticker": "AAA", "score": 90, "signals": ["breakout"], "last": 3.2, "rvol": 4.0},
        {"ticker": "BBB", "score": 85, "signals": ["breakout"], "last": 12.0, "rvol": 1.2},
        {"ticker": "CCC", "score": 80, "signals": [], "last": 55.0, "rvol": 2.5},
        {"ticker": "DDD", "score": 70, "signals": [], "last": None, "rvol": None},
        {"ticker": "EEE", "score": 60, "signals": [], "last": 150.0, "rvol": 3.1},
    ]
    run = {"id": 1, "created_at": None}
    with mock.patch.object(scans, "market_runs", return_value=[run]), \
            mock.patch.object(scans, "run_opportunities", return_value=opps), \
            mock.patch.object(scans, "scan_freshness", return_value={"stale": False}), \
            mock.patch.object(scans, "run_freshness", return_value={}), \
            mock.patch.object(scans, "_iso", return_value=None):
        return scans.latest_scan({}, "premium", limit=50, offset=0, **kw)


def tickers(r):
    return [s["ticker"] for s in r["setups"]]


class InlineFilterTests(unittest.TestCase):
    def test_no_filters_keeps_every_row(self):
        r = _scan()
        self.assertEqual(tickers(r), ["AAA", "BBB", "CCC", "DDD", "EEE"])

    def test_price_band_is_inclusive_and_skips_rows_without_a_price(self):
        r = _scan(min_price=5, max_price=55)
        self.assertEqual(tickers(r), ["BBB", "CCC"])
        self.assertEqual((r["total"], r["scan_total"]), (2, 5))
        self.assertEqual(tickers(_scan(max_price=5)), ["AAA"])
        self.assertEqual(tickers(_scan(min_price=100)), ["EEE"])

    def test_min_rvol_combines_with_score_and_signal(self):
        self.assertEqual(tickers(_scan(min_rvol=2)), ["AAA", "CCC", "EEE"])
        self.assertEqual(tickers(_scan(min_rvol=2, min_score=75)), ["AAA", "CCC"])
        self.assertEqual(tickers(_scan(min_rvol=2, signal="breakout")), ["AAA"])


if __name__ == "__main__":
    unittest.main()
