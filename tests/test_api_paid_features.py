"""Paid-feature parity APIs (P1-67..P1-72): server-side plan gates and shapes."""
import json
import os
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase, _hash

PG_URL = os.environ.get("HSF_TEST_PG_URL")


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class PaidApiTestCase(ApiTestCase):
    def setUp(self):
        super().setUp()
        for email, tier in (("free@example.com", "basic"), ("prem@example.com", "premium")):
            self.accounts[email] = {"username": email, "full_name": email.split("@")[0], "tier": tier,
                                    "password": _hash("right pw"), "is_admin": False, "is_active": True}

    def h(self, email):
        return self.auth(self.login(email).json()["access_token"])

    def get(self, email, path, **kw):
        return self.client.get(path, headers=self.h(email), **kw)


class HistoryTests(PaidApiTestCase):
    RUNS = [{"id": 11, "name": "SP500 | 2 results | 3.0s", "label": "SP500", "username": "pro@example.com",
             "row_count": 2, "duration_sec": 3.0, "is_snapshot": False, "created_at": None}]

    def setUp(self):
        super().setUp()
        import pandas as pd

        df = pd.DataFrame([{"Ticker": "AAA", "BreakoutScore": 80, "IsBreakout": True, "Last": 10.0, "PctChange": 3.0,
                            "VolRel20": 2.0, "PreBreakoutProb%": 77.0},
                           {"Ticker": "BBB", "BreakoutScore": 50, "Last": 20.0, "PctChange": 1.0}])
        p = mock.patch
        p("db.runs.list_runs", side_effect=lambda **k: [r for r in self.RUNS if r["username"] == k.get("username")]).start()
        p("api.history._owned_run", side_effect=lambda u, i: next((r for r in self.RUNS if r["id"] == i and r["username"] == u), None)).start()
        p("api.today.run_df", return_value=df).start()

    def test_free_is_refused(self):
        for path in ("/v1/runs", "/v1/runs/11", "/v1/track-record", "/v1/track-record/daily"):
            with self.subTest(path=path):
                r = self.get("free@example.com", path)
                self.assertEqual(r.status_code, 403)
                self.assertIn("Pro", r.json()["detail"])

    def test_pro_lists_and_opens_own_runs(self):
        r = self.get("pro@example.com", "/v1/runs")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual([x["id"] for x in r.json()], [11])
        d = self.get("pro@example.com", "/v1/runs/11").json()
        self.assertEqual(d["label"], "SP500")
        self.assertEqual([s["ticker"] for s in d["setups"]][:1], ["AAA"])
        self.assertTrue(all(s["prob"] is None for s in d["setups"]))          # PreBreakout redacted for Pro
        self.assertEqual(d["max_results"], 100)

    def test_someone_elses_run_is_404(self):
        self.assertEqual(self.get("prem@example.com", "/v1/runs/11").status_code, 404)
        self.assertEqual(self.get("prem@example.com", "/v1/runs").json(), [])

    def test_track_record(self):
        rows = {(5, "breakout"): {"horizon_days": 5, "avg_return": 0.012, "median_return": 0.01, "win_rate": 0.55,
                                  "sample_size": 40, "runs_used": 20, "computed_at": None, "benchmark": "SPY",
                                  "top_n": 10, "ranking": "breakout"}}
        with mock.patch("db.track_record.load_latest_track_record",
                        side_effect=lambda h, ranking: rows.get((h, ranking))), \
                mock.patch("db.track_record.load_daily_excess", return_value=[("2026-10-01", 0.004)]):
            tr = self.get("pro@example.com", "/v1/track-record").json()
            daily = self.get("pro@example.com", "/v1/track-record/daily?ranking=breakout&horizon=5").json()
            bad = self.get("pro@example.com", "/v1/track-record/daily?horizon=7")
        self.assertEqual(len(tr["summaries"]), 1)
        self.assertTrue(tr["summaries"][0]["sufficient"])
        self.assertIn("not evidence", tr["disclaimer"])
        self.assertEqual(daily, [{"day": "2026-10-01", "avg_excess_return": 0.004}])
        self.assertEqual(bad.status_code, 422)


class StockHistoricalGateTests(PaidApiTestCase):
    def test_historical_research_is_pro(self):
        core = {"intel": {"has_opportunity": True, "hsf_score": 70, "history_summary": {"observations": 3},
                          "historical_context": {"n": 40}, "outcome_cohort": {"available": True}},
                "scan_at": None, "in_latest_scan": True, "quote": {"last": 1.0, "chg_pct": 0.0}}
        with mock.patch("api.scans._stock_core", return_value=core), \
                mock.patch("api.scans.daily_bars", return_value={"bars": [], "as_of": None}), \
                mock.patch("api.user_data.watchlists_with", return_value=[]), \
                mock.patch("api.user_data.alerts_for", return_value=[]):
            free = self.get("free@example.com", "/v1/stocks/AAA").json()
            pro = self.get("pro@example.com", "/v1/stocks/AAA").json()
        self.assertTrue(free["historical_locked"])
        self.assertIsNone(free["historical_context"])
        self.assertIsNone(free["history_summary"])
        self.assertFalse(pro["historical_locked"])
        self.assertEqual(pro["historical_context"], {"n": 40})


class EarningsTests(PaidApiTestCase):
    def test_pro_gets_calendar_free_refused(self):
        import datetime as dt

        today = dt.datetime.now(dt.timezone.utc).date()
        rows = [{"symbol": "MSFT", "earnings_date": today + dt.timedelta(days=3), "earnings_time": "AMC"},
                {"symbol": "BRK-B", "earnings_date": today + dt.timedelta(days=1), "earnings_time": None},
                {"symbol": "XYZ", "earnings_date": None, "earnings_time": None}]
        with mock.patch("db.earnings.fetch_earnings_this_week", return_value=rows) as f:
            self.assertEqual(self.get("free@example.com", "/v1/earnings").status_code, 403)
            r = self.get("pro@example.com", "/v1/earnings?days=14")
            only = self.get("pro@example.com", "/v1/earnings?tickers=msft,brk.b").json()
        self.assertEqual(f.call_args_list[0].kwargs["days_ahead"], 14)
        items = r.json()
        self.assertEqual([i["ticker"] for i in items], ["BRK-B", "MSFT", "XYZ"])     # soonest first, unknown last
        self.assertEqual((items[1]["days_until"], items[1]["time"]), (3, "amc"))
        self.assertEqual({i["ticker"] for i in only}, {"MSFT", "BRK-B"})
        self.assertEqual(self.get("pro@example.com", "/v1/earnings?days=31").status_code, 422)


class BriefTests(PaidApiTestCase):
    DATA = {"gappers": [{"ticker": "MSFT ⚠️E3d", "last": 400.0, "chg_pct": 2.0, "gap_pct": 3.1}],
            "golden": ["AAPL"], "top_setups": [("NVDA", 91.5)],
            "picks": [{"symbol": "AMD ⚠️E1d", "prob": 77.0}], "earnings_today": ["ORCL"],
            "market_close": [("S&P 500 (SPY)", 580.0, 0.4)], "gainers": [("TSLA", 5.0)], "losers": [("INTC", -3.0)],
            "breadth": (300, 200), "sectors": [("Tech", 1.2)], "snapshot_time": "2026-10-06T13:35:00+00:00"}

    def setUp(self):
        super().setUp()
        from api.today import _cache as cache

        cache.clear()
        self.addCleanup(cache.clear)

    def _get(self, email):
        with mock.patch("ui.market_brief._compute_brief", return_value=dict(self.DATA)), \
                mock.patch("ui.market_brief._market_phase", return_value="regular"), \
                mock.patch("ui.opportunities.build_opportunities",
                           return_value=[{"ticker": "NVDA", "score": 91, "prob": 0.8, "signals": ["prebreakout"]}]), \
                mock.patch("ui.opportunities.compare_opportunities", side_effect=lambda o, p: o), \
                mock.patch("db.opportunity_snapshots.load_previous_opportunity_snapshot", return_value=None), \
                mock.patch("db.opportunity_snapshots.save_opportunity_snapshot") as save, \
                mock.patch("db.signal_outcomes.freeze_opportunities") as freeze:
            r = self.get(email, "/v1/brief")
        self.assertFalse(save.called or freeze.called, "the API must not write research/snapshot rows")
        return r

    def test_shape_and_earnings_flags(self):
        r = self._get("prem@example.com")
        self.assertEqual(r.status_code, 200, r.text)
        b = r.json()
        self.assertTrue(b["available"])
        self.assertEqual(b["gappers"][0]["ticker"], "MSFT")
        self.assertEqual(b["gappers"][0]["earnings_days"], 3)
        self.assertEqual(b["prebreakout_picks"], [{"ticker": "AMD", "prob": 77.0, "earnings_days": 1}])
        self.assertEqual(b["breadth"], {"advancers": 300, "decliners": 200})
        self.assertEqual(b["top_breakout_scores"], [{"ticker": "NVDA", "score": 91.5}])
        self.assertEqual(b["market"][0]["label"], "S&P 500 (SPY)")
        self.assertFalse(b["prebreakout_locked"])

    def test_premium_content_redacted_below_premium(self):
        from api.today import _cache as cache

        b = self._get("pro@example.com").json()
        self.assertEqual(b["prebreakout_picks"], [])
        self.assertTrue(b["prebreakout_locked"])
        self.assertTrue(all(o.get("prob") is None and "prebreakout" not in (o.get("signals") or [])
                            for o in b["opportunities"]))
        cache.clear()
        self.assertTrue(self._get("free@example.com").json()["available"])   # the brief itself is every plan

    def test_no_snapshot_yet(self):
        with mock.patch("ui.market_brief._compute_brief", return_value=None):
            b = self.get("free@example.com", "/v1/brief").json()
        self.assertEqual(b["available"], False)


class DayTraderTests(PaidApiTestCase):
    ROWS = [{"ticker": "AAPL", "open": 100.0, "last": 103.0, "chg_pct": 3.0, "gap_pct": 1.0, "vwap": 102.0,
             "vs_vwap_pct": 1.0, "rvol": 2.5, "volume": 1e6, "data_source": "alpaca_iex"}]

    def setUp(self):
        super().setUp()
        from api.today import _cache

        _cache.clear()
        self.addCleanup(_cache.clear)
        p = mock.patch
        self.metrics = p("market_data.build_day_trader_metrics", side_effect=lambda syms, **k: [
            dict(r) for r in self.ROWS if r["ticker"] in syms]).start()
        p("ui.day_trader._fetch_clock_is_open", return_value=None).start()
        p("ui.day_trader.market_state", return_value="open").start()
        p("api.user_data.list_watchlists", side_effect=lambda u: [{"id": 7, "is_default": True}]
          if u == "pro@example.com" else []).start()
        p("api.user_data.get_watchlist", side_effect=self._wl).start()

    def _wl(self, user, wid):
        from api.user_data import NotFound

        if user == "pro@example.com" and wid == 7:
            return {"id": 7, "items": [{"ticker": "AAPL"}, {"ticker": "MSFT"}]}
        raise NotFound("watchlist")

    def test_free_refused(self):
        self.assertEqual(self.get("free@example.com", "/v1/day-trader").status_code, 403)
        self.assertEqual(self.get("free@example.com", "/v1/day-trader/stair-steppers?symbols=AAPL").status_code, 403)

    def test_watchlist_source_and_score(self):
        r = self.get("pro@example.com", "/v1/day-trader")
        self.assertEqual(r.status_code, 200, r.text)
        b = r.json()
        self.assertEqual((b["state"], b["source"], b["symbols"], b["missing"]), ("open", "watchlist", ["AAPL", "MSFT"], 1))
        self.assertGreater(b["rows"][0]["day_trade_score"], 0)
        self.assertEqual(b["rows"][0]["data_source"], "alpaca_iex")      # extra live fields pass through
        self.assertEqual(self.get("pro@example.com", "/v1/day-trader?watchlist_id=99").status_code, 404)

    def test_custom_symbols_are_validated_and_quotes_shared(self):
        b = self.get("pro@example.com", "/v1/day-trader?source=custom&symbols=aapl, $$$,brk.b").json()
        self.assertEqual(b["symbols"], ["AAPL", "BRK.B"])
        self.get("prem@example.com", "/v1/day-trader?source=custom&symbols=AAPL,BRK.B")
        self.assertEqual(self.metrics.call_count, 1)                    # second caller hit the 30 s cache
        self.assertEqual(self.get("pro@example.com", "/v1/day-trader?source=bogus").status_code, 422)

    def test_movers_source(self):
        with mock.patch("ui.day_trader._top_movers_symbols", return_value=["AAPL"]) as movers:
            b = self.get("pro@example.com", "/v1/day-trader?source=movers").json()
            self.get("prem@example.com", "/v1/day-trader?source=movers")
        self.assertEqual(b["symbols"], ["AAPL"])
        self.assertEqual(movers.call_count, 1)

    def test_stair_steppers(self):
        rows = [{"ticker": "AAPL", "status": "ok", "direction": "up", "r2": 0.95, "pullback_pct": 0.2,
                 "trend_pct_per_hour": 1.5}]
        with mock.patch("ui.stair_stepper.fetch_recent_minute_bars", return_value={"AAPL": []}), \
                mock.patch("ui.stair_stepper.build_rows", return_value=rows), \
                mock.patch("analytics.stair_step.is_stair_stepper", return_value=True):
            b = self.get("pro@example.com", "/v1/day-trader/stair-steppers?symbols=aapl").json()
        self.assertEqual(b["checked"], ["AAPL"])
        self.assertEqual(len(b["matches"]), 1)
        self.assertEqual(self.get("pro@example.com", "/v1/day-trader/stair-steppers?symbols=AAPL&window=7").status_code, 422)


class AITests(PaidApiTestCase):
    def setUp(self):
        super().setUp()
        import pandas as pd

        from api.today import _cache

        _cache.clear()
        self.addCleanup(_cache.clear)
        self.accounts["prem2@example.com"] = dict(self.accounts["prem@example.com"], username="prem2@example.com")
        df = pd.DataFrame([{"Ticker": "AAA", "BreakoutScore": 80, "GapPct": 3.0}, {"Ticker": "BBB", "BreakoutScore": 50}])
        p = mock.patch
        p("api.today.market_runs", return_value=[{"id": 5, "created_at": None}]).start()
        p("api.today.run_df", return_value=df).start()
        p("api.history._owned_run", side_effect=lambda u, i: {"id": i} if (u, i) == ("prem@example.com", 9) else None).start()
        self.ask = p("ui.ai.ask_claude", return_value=("**AAA** leads.", None)).start()
        self.chat = p("ui.ai.ask_claude_chat", return_value=("AAA has the higher score.", None)).start()

    def post(self, email, path, body=None):
        return self.client.post(path, headers=self.h(email), json=body or {})

    def test_premium_only(self):
        for path in ("/v1/ai/summary", "/v1/ai/notes/AAA"):
            self.assertEqual(self.post("pro@example.com", path).status_code, 403, path)
        self.assertEqual(self.get("pro@example.com", "/v1/ai/brief-narrative").status_code, 403)
        self.assertEqual(self.post("pro@example.com", "/v1/ai/chat", {"messages": [{"role": "user", "content": "hi"}]}).status_code, 403)
        self.ask.assert_not_called()

    def test_summary_shared_per_scan_and_counted_to_the_caller(self):
        a = self.post("prem@example.com", "/v1/ai/summary").json()
        b = self.post("prem2@example.com", "/v1/ai/summary").json()
        self.assertEqual((a["run_id"], a["text"]), (5, "**AAA** leads."))
        self.assertEqual(b["text"], a["text"])
        self.assertEqual(self.ask.call_count, 1)                                   # one call per scan
        kw = self.ask.call_args.kwargs
        self.assertEqual((kw["username"], kw["feature"]), ("prem@example.com", "scan_summary"))
        self.assertIn("AAA", kw["user"])

    def test_own_saved_scan_only(self):
        self.assertEqual(self.post("prem@example.com", "/v1/ai/summary", {"run_id": 9}).status_code, 200)
        self.assertEqual(self.post("prem2@example.com", "/v1/ai/summary", {"run_id": 9}).status_code, 404)

    def test_limits_and_outages(self):
        self.ask.return_value = (None, "You've reached today's AI usage limit. Try again tomorrow.")
        self.assertEqual(self.post("prem@example.com", "/v1/ai/notes/AAA").status_code, 429)
        self.ask.return_value = (None, "AI features are temporarily disabled.")
        r = self.post("prem@example.com", "/v1/ai/notes/BBB")
        self.assertEqual(r.status_code, 503)
        self.ask.return_value = (None, "AI failed: boom secret detail")
        r = self.post("prem@example.com", "/v1/ai/summary", {"run_id": 9})
        self.assertEqual(r.status_code, 502)
        self.assertNotIn("secret", r.text)                                         # provider errors aren't echoed
        self.assertIsNone(self.post("prem@example.com", "/v1/ai/notes/ZZZZ").json()["text"])   # not in the scan

    def test_chat(self):
        msgs = [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"}
                for i in range(15)]
        r = self.post("prem@example.com", "/v1/ai/chat", {"messages": msgs})
        self.assertEqual(r.json()["answer"], "AAA has the higher score.")
        sent = self.chat.call_args.kwargs["messages"]
        self.assertIn("scan results (CSV)", sent[0]["content"])
        self.assertEqual(sent[-1], {"role": "user", "content": "q14"})
        self.assertLessEqual(len(sent), 2 + 16)
        self.assertEqual(sent[2]["role"], "user")
        bad = self.post("prem@example.com", "/v1/ai/chat", {"messages": [{"role": "assistant", "content": "x"}]})
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(self.post("prem@example.com", "/v1/ai/chat", {"messages": []}).status_code, 422)

    def test_brief_narrative(self):
        with mock.patch("api.market._brief_core", return_value={"data": {"snapshot_time": "t1", "breadth": (3, 2)}}):
            r = self.get("prem@example.com", "/v1/ai/brief-narrative").json()
        self.assertEqual(r["snapshot_time"], "t1")
        self.assertEqual(self.ask.call_args.kwargs["feature"], "market_brief_narrative")


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class OwnedRunPostgresTests(unittest.TestCase):
    def test_owned_run_checks_the_owner(self):
        import psycopg

        os.environ["DATABASE_URL"] = PG_URL
        self.addCleanup(os.environ.pop, "DATABASE_URL", None)
        from db.schema import ensure_neon_runs_schema

        with psycopg.connect(PG_URL) as c:
            ensure_neon_runs_schema.__wrapped__(c)   # schema_once may have run already in this process
        with psycopg.connect(PG_URL) as c:
            rid = c.execute("INSERT INTO runs (name, results_json, label, username, row_count) "
                            "VALUES ('x', %s, 'SP500', 'Pro@Example.com', 0) RETURNING id",
                            (json.dumps([]),)).fetchone()[0]
        from api.history import _owned_run

        self.assertEqual(_owned_run("pro@example.com", rid)["id"], rid)
        self.assertIsNone(_owned_run("other@example.com", rid))


if __name__ == "__main__":
    unittest.main()
