"""AI watchlist note (Premium): POST /v1/ai/watchlists/{id}/note and its facts block."""
import unittest
from unittest import mock

from tests.test_api_paid_features import PaidApiTestCase

INTEL = {"watchlist_id": 3, "name": "Core", "scan_available": True, "last_scan_at": "2026-10-20T14:35:00+00:00",
         "previous_scan_at": "2026-10-20T13:35:00+00:00", "items": [
             {"ticker": "AAA", "in_latest_scan": True, "ranked": True, "hsf_score": 82, "previous_hsf_score": 70,
              "score_change": 12, "rank": 4, "setup": "Breakout", "status": "STRONG", "price_change_pct": 3.25,
              "rvol": 2.4},
             {"ticker": "BBB", "in_latest_scan": True, "ranked": False, "previous_rank": 30, "score_change": None},
             {"ticker": "CCC", "in_latest_scan": False},
         ]}


class FactsTests(unittest.TestCase):
    def test_facts_lines_biggest_change_first(self):
        from api.ai import watchlist_facts

        facts = watchlist_facts(INTEL, {"CCC": {"earnings_days": 1}})
        lines = facts.splitlines()
        self.assertTrue(lines[0].startswith("- AAA: HSF Score 82; change +12"))
        self.assertIn("relative volume 2.4x", lines[0])
        self.assertIn("was ranked #30", facts)
        self.assertIn("CCC: not in the latest scan; reports earnings tomorrow", facts)

    def test_empty_list_or_no_scan(self):
        from api.ai import watchlist_facts

        self.assertEqual(watchlist_facts({**INTEL, "items": []}, {}), "")
        self.assertEqual(watchlist_facts({**INTEL, "scan_available": False}, {}), "")


class WatchlistNoteApiTests(PaidApiTestCase):
    def setUp(self):
        super().setUp()
        from api.today import _cache

        _cache.clear()
        self.addCleanup(_cache.clear)
        p = mock.patch
        p("api.user_data.get_watchlist", return_value={"id": 3, "name": "Core", "items": []}).start()
        self.intel = p("api.watchlist_intel.intelligence", return_value=INTEL).start()
        p("api.alert_rules.earnings_lookup", return_value={"AAA": {"earnings_days": 0}, "BBB": {"earnings_days": 30}}).start()
        self.ask = p("ui.ai.ask_claude", return_value=("AAA jumped 12 points.", None)).start()
        self.addCleanup(mock.patch.stopall)

    def post(self, email):
        return self.client.post("/v1/ai/watchlists/3/note", headers=self.h(email))

    def test_premium_only(self):
        self.assertEqual(self.post("pro@example.com").status_code, 403)
        self.ask.assert_not_called()

    def test_note_is_written_from_facts_and_cached_per_scan(self):
        r = self.post("prem@example.com").json()
        self.assertEqual((r["text"], r["snapshot_time"]), ("AAA jumped 12 points.", INTEL["last_scan_at"]))
        kw = self.ask.call_args.kwargs
        self.assertEqual(kw["feature"], "watchlist_note")
        self.assertIn("AAA: HSF Score 82", kw["user"])
        self.assertIn("reports earnings today", kw["user"])
        self.assertNotIn("in 30 days", kw["user"])  # only earnings within a week
        self.post("prem@example.com")
        self.assertEqual(self.ask.call_count, 1)

    def test_empty_watchlist_returns_no_text(self):
        self.intel.return_value = {**INTEL, "items": []}
        self.assertIsNone(self.post("prem@example.com").json()["text"])
        self.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
