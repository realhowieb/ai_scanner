"""Kalshi BTC market data: normalization, classification, nearest-the-money."""
from __future__ import annotations

import unittest
from unittest import mock


def _raw(ticker, subtitle, floor, yes_bid, yes_ask, close="2026-07-31T04:00:00Z"):
    return {
        "ticker": ticker,
        "title": "Bitcoin price on Jul 31, 2026?",
        "subtitle": subtitle,
        "floor_strike": floor,
        "yes_bid_dollars": yes_bid,
        "yes_ask_dollars": yes_ask,
        "last_price_dollars": yes_ask,
        "close_time": close,
    }


class NormalizeTests(unittest.TestCase):
    def test_parses_dollars_and_classifies_kind(self):
        from data.kalshi_markets import _normalize

        above = _normalize(_raw("KXBTC-x-T70000", "$70,000 or above", "70000", "0.44", "0.46"))
        self.assertEqual(above["kind"], "above")
        self.assertEqual(above["yes_bid"], 0.44)
        self.assertEqual(above["yes_ask"], 0.46)
        self.assertEqual(above["yes_prob_pct"], 45.0)      # mid → implied prob %
        self.assertEqual(above["spread_cents"], 2.0)
        self.assertEqual(above["series"], "KXBTC")

        below = _normalize(_raw("KXBTC-x-T60000", "$60,000 or below", "60000", "0.30", "0.32"))
        self.assertEqual(below["kind"], "below")
        rng = _normalize(_raw("KXBTC-x-B65000", "$65,000 to 65,099.99", "65000", "0.10", "0.12"))
        self.assertEqual(rng["kind"], "range")

    def test_missing_ticker_dropped(self):
        from data.kalshi_markets import _normalize

        self.assertIsNone(_normalize({"subtitle": "x"}))


class NearestTheMoneyTests(unittest.TestCase):
    def test_picks_closest_directional_skipping_ranges(self):
        from data.kalshi_markets import _normalize, nearest_the_money

        markets = [
            _normalize(_raw("KXBTC-1", "$70,000 or above", "70000", "0.4", "0.5")),
            _normalize(_raw("KXBTC-2", "$64,500 to 64,599.99", "64500", "0.4", "0.5")),  # range, closest
            _normalize(_raw("KXBTC-3", "$64,000 or above", "64000", "0.4", "0.5")),
        ]
        atm = nearest_the_money(markets, 64300)
        # closest strike is the range bucket, but ranges are skipped → picks 64000
        self.assertEqual(atm["floor_strike"], 64000.0)
        self.assertEqual(atm["kind"], "above")


class FetchTests(unittest.TestCase):
    def test_fetch_merges_series_and_sorts_by_close(self):
        import data.kalshi_markets as km

        def fake_fetch(series, limit):
            if series == "KXBTC":
                return [_raw("KXBTC-late", "$70,000 or above", "70000", "0.4", "0.5",
                             close="2026-07-31T20:00:00Z")]
            if series == "KXBTCD":
                return [_raw("KXBTCD-soon", "$60,000 or above", "60000", "0.4", "0.5",
                             close="2026-07-31T04:00:00Z")]
            return []

        with mock.patch.object(km, "_fetch_series", side_effect=fake_fetch):
            out = km.fetch_btc_markets()
        self.assertEqual([m["ticker"] for m in out], ["KXBTCD-soon", "KXBTC-late"])  # soonest first


if __name__ == "__main__":
    unittest.main()
