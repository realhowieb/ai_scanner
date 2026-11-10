"""Scanner rank moves (rank / previous_rank) and sectors on /v1/scans/latest."""
from __future__ import annotations

import unittest
from unittest import mock

CURRENT = [
    {"ticker": "AAA", "score": 90, "signals": ["breakout"]},
    {"ticker": "BBB", "score": 85, "signals": []},
    {"ticker": "NEW", "score": 80, "signals": ["breakout"]},
    {"ticker": "CCC", "score": 70, "signals": []},
]
PREVIOUS = [
    {"ticker": "CCC", "score": 88, "signals": []},
    {"ticker": "AAA", "score": 84, "signals": []},
    {"ticker": "BBB", "score": 80, "signals": []},
]
SECTORS = {"AAA": "Technology", "BBB": "Finance", "CCC": "Technology"}


def _scan(runs=2, previous=PREVIOUS, **kw):
    from api import scans

    rows = [{"id": 2, "created_at": "now"}, {"id": 1, "created_at": "before"}][:runs]
    by_id = {2: CURRENT, 1: previous}
    with mock.patch.object(scans, "market_runs", return_value=rows), \
            mock.patch.object(scans, "run_opportunities", side_effect=lambda i: by_id[i]), \
            mock.patch.object(scans, "scan_freshness", return_value={"stale": False}), \
            mock.patch.object(scans, "run_freshness", return_value={}), \
            mock.patch.object(scans, "_iso", side_effect=lambda v: v), \
            mock.patch("api.sectors.sector_map", return_value=SECTORS):
        return scans.latest_scan({}, "premium", limit=50, offset=0, **kw)


def col(r, key):
    return {s["ticker"]: s[key] for s in r["setups"]}


class RankMoveTests(unittest.TestCase):
    def test_rows_carry_their_rank_now_and_in_the_scan_before(self):
        r = _scan()
        self.assertEqual(col(r, "rank"), {"AAA": 1, "BBB": 2, "NEW": 3, "CCC": 4})
        self.assertEqual(col(r, "previous_rank"), {"AAA": 2, "BBB": 3, "NEW": None, "CCC": 1})
        self.assertEqual(r["previous_scan_at"], "before")

    def test_rank_is_the_whole_scan_position_under_a_filter(self):
        r = _scan(signal="breakout")
        self.assertEqual(col(r, "rank"), {"AAA": 1, "NEW": 3})

    def test_no_earlier_scan_means_nothing_is_compared(self):
        for r in (_scan(runs=1), _scan(previous=[])):
            self.assertIsNone(r["previous_scan_at"])
            self.assertEqual(set(col(r, "previous_rank").values()), {None})


class SectorTests(unittest.TestCase):
    def test_rows_carry_a_sector_and_the_choices_count_them(self):
        r = _scan()
        self.assertEqual(col(r, "sector"), {"AAA": "Technology", "BBB": "Finance", "NEW": None, "CCC": "Technology"})
        self.assertEqual(r["sectors"], [{"sector": "Technology", "count": 2}, {"sector": "Finance", "count": 1}])

    def test_sector_filter_keeps_the_choices_and_counts_the_match(self):
        r = _scan(sector="Technology")
        self.assertEqual([s["ticker"] for s in r["setups"]], ["AAA", "CCC"])
        self.assertEqual(r["total"], 2)
        self.assertEqual(len(r["sectors"]), 2)

    def test_choices_follow_the_other_filters(self):
        r = _scan(signal="breakout")
        self.assertEqual(r["sectors"], [{"sector": "Technology", "count": 1}])


class SectorFileTests(unittest.TestCase):
    def test_lookup_ignores_separators_and_unknowns(self):
        from api import sectors

        with mock.patch.object(sectors, "sector_map", return_value={"BRKB": "Finance"}):
            self.assertEqual(sectors.sector_for("BRK-B"), "Finance")
            self.assertEqual(sectors.sector_for("brk.b"), "Finance")
            self.assertIsNone(sectors.sector_for("ZZZZ"))

    def test_screener_parse(self):
        from scripts.refresh_sectors import parse_screener, to_csv

        got = parse_screener({"data": {"rows": [
            {"symbol": "BRK/B", "sector": "Finance", "industry": "Insurance", "marketCap": "1,000,000,000,000.00"},
            {"symbol": "AAPL ", "sector": "Technology", "marketCap": "NA"},
            {"symbol": "XYZ", "sector": ""}]}})
        self.assertEqual(got, {"AAPL": ("Technology", "", ""), "BRK-B": ("Finance", "Insurance", "1000000000000")})
        self.assertEqual(to_csv(got), "ticker,sector,industry,market_cap\nAAPL,Technology,,\n"
                                      "BRK-B,Finance,Insurance,1000000000000\n")


if __name__ == "__main__":
    unittest.main()
