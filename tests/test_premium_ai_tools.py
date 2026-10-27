"""Premium AI tools (2026-10-23): Ask HSF, journal coach, scan builder, the alert "Why"
line and the morning email rundown."""
import json
import unittest
from unittest import mock

from tests.test_api_paid_features import PaidApiTestCase


def _obs():
    ranked = {
        "AAA": {"score": 82, "rank": 1, "primary_setup": "Breakout", "status": "STRONG", "last": 10.5,
                "chg_pct": 3.2, "rvol": 2.4, "signals": ["breakout"]},
        "BBB": {"score": 61, "rank": 2, "primary_setup": "Gapper", "last": 20.0, "chg_pct": 1.0, "rvol": 1.1,
                "signals": ["gapper"]},
    }
    return {"observation_id": "obs-1", "scan_at": "2026-10-23T14:35:00+00:00", "ranked": ranked,
            "raw": {"CCC": {"last": 5.0, "chg_pct": -1.0, "rvol": 0.8}}, "previous_ranked": {"AAA": {"score": 70, "rank": 3}}}


class TickerMentionTests(unittest.TestCase):
    def test_dollar_tickers_and_known_upper_case_words(self):
        from api.ai_premium import mentioned_tickers

        got = mentioned_tickers("How do $nvda and AAA look? IS THE market OK, and what about msft?", {"AAA", "MSFT"})
        self.assertEqual(got, ["NVDA", "AAA"])  # lower-case msft isn't a ticker unless written $msft

    def test_capped(self):
        from api.ai_premium import MAX_MENTIONS, mentioned_tickers

        self.assertEqual(len(mentioned_tickers(" ".join(f"${c}" for c in "BCDEFGHJ"), set())), MAX_MENTIONS)


class ScanBuilderParseTests(unittest.TestCase):
    def parse(self, obj, ids=(4, 9)):
        from api.ai_premium import parse_builder

        return parse_builder("Sure:\n" + json.dumps(obj), list(ids))

    def test_valid_settings_pass_through(self):
        out = self.parse({"universe": "nasdaq", "session": "premarket", "top_n": 50, "min_price": 5,
                          "unusual": True, "min_gap": 3, "explain": "NASDAQ gappers over $5."})
        v = out["view"]
        self.assertEqual((v["universe"], v["session"], v["top_n"], v["min_price"]), ("nasdaq", "premarket", 50, 5.0))
        self.assertTrue(v["unusual"] and v["gap"])
        self.assertEqual(v["min_gap"], 3.0)
        self.assertEqual(out["explanation"], "NASDAQ gappers over $5.")

    def test_bad_values_are_dropped(self):
        v = self.parse({"universe": "mars", "session": "night", "top_n": 70, "min_price": -1, "max_price": "x",
                        "rm_rf": True, "score_all": True})["view"]
        self.assertEqual((v["universe"], v["session"], v["top_n"]), ("sp500", "regular", 50))
        self.assertNotIn("min_price", v)
        self.assertNotIn("max_price", v)
        self.assertNotIn("rm_rf", v)
        self.assertFalse(v["score_all"])  # only for ticker / watchlist / earnings

    def test_watchlist_must_be_the_users(self):
        self.assertEqual(self.parse({"universe": "watchlist", "watchlist_id": 9})["view"]["watchlist_id"], 9)
        self.assertEqual(self.parse({"universe": "watchlist", "watchlist_id": 77})["view"]["watchlist_id"], 4)
        self.assertEqual(self.parse({"universe": "watchlist"}, ids=())["view"]["universe"], "sp500")

    def test_ticker_and_price_order(self):
        v = self.parse({"universe": "ticker", "ticker": "$tsla", "min_price": 50, "max_price": 10})["view"]
        self.assertEqual((v["ticker"], v["min_price"], v["max_price"]), ("TSLA", 10.0, 50.0))

    def test_no_json_raises(self):
        from api.ai_premium import parse_builder

        with self.assertRaises(ValueError):
            parse_builder("I can't do that.", [])


class PremiumAiApiTests(PaidApiTestCase):
    def setUp(self):
        super().setUp()
        from api.today import _cache

        _cache.clear()
        self.addCleanup(_cache.clear)
        p = mock.patch
        p("api.watchlist_intel.market_observation", return_value=_obs()).start()
        p("api.watchlist_intel._active_alert_counts", return_value=None).start()
        p("api.user_data.list_watchlists", return_value=[{"id": 4, "name": "Core", "is_default": True, "symbol_count": 2}]).start()
        p("api.user_data.get_watchlist", return_value={"id": 4, "name": "Core", "items": [{"ticker": "AAA"}, {"ticker": "CCC"}]}).start()
        p("api.alert_rules.earnings_lookup", return_value={"CCC": {"earnings_days": 1, "earnings_time": "amc"}}).start()
        p("api.news.stock_news", return_value={"items": [{"headline": "AAA wins contract", "published_at": "2026-10-23T12:00:00Z"}]}).start()
        p("api.outcomes.ai_evidence", return_value=None).start()
        p("api.ai_premium._brief_facts", return_value="- SPY: +0.40%").start()
        self.chat = p("ui.ai.ask_claude_chat", return_value=("AAA leads your list.", None)).start()
        self.ask = p("ui.ai.ask_claude", return_value=("- You do better on 80+ scores.", None)).start()
        self.addCleanup(mock.patch.stopall)

    def post(self, email, path, body=None):
        return self.client.post(path, headers=self.h(email), json=body)

    def test_ask_is_premium_only(self):
        for email in ("free@example.com", "pro@example.com"):
            self.assertEqual(self.post(email, "/v1/ai/ask", {"messages": [{"role": "user", "content": "hi"}]}).status_code, 403)
        self.chat.assert_not_called()

    def test_ask_answers_from_hsf_facts(self):
        r = self.post("prem@example.com", "/v1/ai/ask", {"messages": [{"role": "user", "content": "How is AAA doing?"}]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["answer"], "AAA leads your list.")
        kw = self.chat.call_args.kwargs
        self.assertEqual(kw["feature"], "ask_hsf")
        context = kw["messages"][0]["content"]
        self.assertIn('"Core" (default): AAA, CCC', context)
        self.assertIn("AAA: HSF Score 82; rank #1; change +12 since previous scan", context)
        self.assertIn("CCC: scanned, not a ranked HSF setup", context)
        self.assertIn("reports earnings tomorrow after the close", context)
        self.assertIn("Top 2 ranked setups", context)
        self.assertIn("Stocks named in the question", context)
        self.assertIn("AAA wins contract", context)
        self.assertIn("SPY: +0.40%", context)
        self.assertEqual(kw["messages"][-1], {"role": "user", "content": "How is AAA doing?"})

    def test_ask_needs_a_user_question_last(self):
        r = self.post("prem@example.com", "/v1/ai/ask", {"messages": [{"role": "assistant", "content": "hi"}]})
        self.assertEqual(r.status_code, 422)

    def test_journal_coach_needs_enough_trades(self):
        with mock.patch("api.trading.insights", return_value={"closed": 2}):
            r = self.post("prem@example.com", "/v1/ai/journal-coach")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"closed": 2, "min_trades": 5, "text": None})
        self.ask.assert_not_called()

    def test_journal_coach_writes_and_caches(self):
        ins = {"closed": 6, "summary": {"wins": 4, "losses": 2, "win_rate": 66.7, "avg_return_pct": 2.1,
                                        "avg_win_pct": 4.0, "avg_loss_pct": -1.9, "profit_factor": 2.5,
                                        "total_pnl": 420.0, "avg_hold_days": 3.2},
               "by_score": [{"label": "80+", "count": 4, "wins": 4, "win_rate": 100.0, "avg_return_pct": 4.0, "pnl": 500.0}],
               "by_hold": [], "by_source": []}
        trades = [{"ticker": "AAA", "entry_price": 10, "exit_price": 11, "shares": 10, "source": "scan",
                   "entered_at": "2026-10-01T14:00:00+00:00", "closed_at": "2026-10-03T14:00:00+00:00", "hsf_score": 84}]
        with mock.patch("api.trading.insights", return_value=ins), mock.patch("db.trades.closed_trades", return_value=trades):
            r = self.post("prem@example.com", "/v1/ai/journal-coach").json()
            self.post("prem@example.com", "/v1/ai/journal-coach")
        self.assertEqual(r["text"], "- You do better on 80+ scores.")
        self.assertEqual(self.ask.call_count, 1)
        kw = self.ask.call_args.kwargs
        self.assertEqual(kw["feature"], "journal_coach")
        self.assertIn("win rate 66.7%", kw["user"])
        self.assertIn("- 80+: 4 trades", kw["user"])
        self.assertIn("AAA: +10.0%, held 1-5 days, from a scan, HSF Score when taken 84", kw["user"])

    def test_scan_builder_fills_settings(self):
        self.ask.return_value = ('{"universe": "nasdaq", "min_price": 5, "gap": true, "explain": "Gappers."}', None)
        r = self.post("prem@example.com", "/v1/ai/scan-builder", {"text": "nasdaq gappers over $5"})
        self.assertEqual(r.status_code, 200, r.text)
        v = r.json()["view"]
        self.assertEqual((v["universe"], v["min_price"], v["gap"], v["name"], v["ticker"]), ("nasdaq", 5.0, True, "AI scan", ""))
        self.assertIn("id 4", self.ask.call_args.kwargs["user"])
        self.assertEqual(self.ask.call_args.kwargs["feature"], "scan_builder")

    def test_scan_builder_unparseable_is_502(self):
        self.ask.return_value = ("Sorry, no.", None)
        r = self.post("prem@example.com", "/v1/ai/scan-builder", {"text": "anything good"})
        self.assertEqual(r.status_code, 502)

    def test_scan_builder_is_premium_only(self):
        self.assertEqual(self.post("pro@example.com", "/v1/ai/scan-builder", {"text": "anything"}).status_code, 403)


class AlertWhyTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch
        p("ui.ai.is_configured", return_value=True).start()
        self.used = p("api.ai_premium._why_used_today", return_value=0).start()
        p("db.ai_usage.record_ai_call").start()
        self.ask = p("ui.ai.ask_claude", return_value=("AAA jumped 12 points on 2.4x volume.", None)).start()
        self.addCleanup(mock.patch.stopall)

    def ev(self, tier, user="u@example.com"):
        return {"user_id": user, "ticker": "AAA", "message": "AAA HSF Score 82 is at or above 80", "_tier": tier}

    def test_premium_events_get_a_why_line(self):
        from api.ai_premium import annotate_alerts

        events = [self.ev("premium"), self.ev("pro"), self.ev("admin")]
        self.assertEqual(annotate_alerts(events, _obs()), 2)
        self.assertEqual(events[0]["message"], "AAA HSF Score 82 is at or above 80\nWhy: AAA jumped 12 points on 2.4x volume.")
        self.assertEqual(events[1]["message"], "AAA HSF Score 82 is at or above 80")
        kw = self.ask.call_args.kwargs
        self.assertEqual((kw["feature"], kw["username"]), ("alert_why", None))  # own cap, not the daily AI budget
        self.assertIn("AAA: HSF Score 82", kw["user"])

    def test_daily_cap_and_kill_switch(self):
        from api.ai_premium import annotate_alerts

        self.used.return_value = 10
        events = [self.ev("premium")]
        self.assertEqual(annotate_alerts(events, _obs()), 0)
        self.used.return_value = 0
        with mock.patch.dict("os.environ", {"HSF_ALERT_AI_WHY": "0"}):
            self.assertEqual(annotate_alerts(events, _obs()), 0)
        self.assertNotIn("Why:", events[0]["message"])

    def test_failed_ai_keeps_the_message(self):
        from api.ai_premium import annotate_alerts

        self.ask.return_value = (None, "AI rate-limited.")
        events = [self.ev("premium")]
        self.assertEqual(annotate_alerts(events, _obs()), 0)
        self.assertEqual(events[0]["message"], "AAA HSF Score 82 is at or above 80")


class DigestRundownTests(unittest.TestCase):
    def test_compose_shows_the_ai_rundown_card(self):
        from scheduler.morning_digest import _compose

        with mock.patch("scheduler.morning_digest._track_record_line", return_value=("", "")):
            html, text = _compose("u@example.com", [], [], [], [], ai_rundown="AAA gained 12 points <b>.")
        self.assertIn("AI rundown", html)
        self.assertIn("AAA gained 12 points &lt;b&gt;.", html)  # escaped
        self.assertIn("AI rundown (Premium):\nAAA gained 12 points <b>.", text)
        html, _ = _compose("u@example.com", [], [], [], [])
        self.assertNotIn("AI rundown", html)

    def test_rundown_uses_the_default_list_and_never_raises(self):
        from api import ai_premium

        with mock.patch("api.user_data.list_watchlists", return_value=[{"id": 1, "name": "A", "is_default": False},
                                                                       {"id": 2, "name": "B", "is_default": True}]), \
                mock.patch("api.user_data.get_watchlist", return_value={"id": 2, "items": [{"ticker": "AAA"}]}) as gw, \
                mock.patch("api.ai.watchlist_note", return_value={"text": "Quiet morning."}) as note:
            self.assertEqual(ai_premium.digest_rundown("u@example.com"), "Quiet morning.")
        gw.assert_called_with("u@example.com", 2)
        self.assertTrue(note.call_args.args[2]["can_ai_notes"])
        with mock.patch("api.user_data.list_watchlists", side_effect=RuntimeError("db down")):
            self.assertIsNone(ai_premium.digest_rundown("u@example.com"))
        with mock.patch.dict("os.environ", {"HSF_DIGEST_AI_RUNDOWN": "0"}):
            self.assertIsNone(ai_premium.digest_rundown("u@example.com"))


if __name__ == "__main__":
    unittest.main()
