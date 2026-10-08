"""/v1/outcomes/*: plan gate, shapes, explicit filters, validation, caching and
stale/outage behavior. The database is mocked at db.signal_outcomes."""
import datetime as dt
import unittest
from unittest import mock

from tests.test_api_paid_features import DEPS, PaidApiTestCase
from tests.test_outcome_intelligence import T, fixture, row

PATHS = ("/v1/outcomes/summary", "/v1/outcomes/scores", "/v1/outcomes/horizons", "/v1/outcomes/setups",
         "/v1/outcomes/timeseries", "/v1/outcomes/symbols/AAA", "/v1/outcomes/query")


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class OutcomeApiTests(PaidApiTestCase):
    def setUp(self):
        super().setUp()
        from api import outcomes

        outcomes.clear_cache()
        self.addCleanup(outcomes.clear_cache)
        self.rows = fixture()
        self.stamp = ("7", "a", "b")
        self.fetch = mock.patch("db.signal_outcomes.fetch_outcome_rows", side_effect=lambda **k: list(self.rows)).start()
        self.probe = mock.patch("db.signal_outcomes.outcome_dataset_stamp", side_effect=lambda: self.stamp).start()

    def test_free_is_refused_everywhere(self):
        for path in PATHS:
            with self.subTest(path=path):
                r = self.get("free@example.com", path)
                self.assertEqual(r.status_code, 403)
        self.assertEqual(self.client.get("/v1/outcomes/summary").status_code, 401)

    def test_every_endpoint_answers_with_filters_and_sample_sizes(self):
        for path in PATHS:
            with self.subTest(path=path):
                r = self.get("pro@example.com", path)
                self.assertEqual(r.status_code, 200, r.text)
                body = r.json()
                self.assertIn("filters", body)
                self.assertIn("raw_observations", body)
                self.assertIn("disclaimer", body)
                self.assertFalse(body["dataset"]["stale"])

    def test_summary_numbers(self):
        s = self.get("pro@example.com", "/v1/outcomes/summary").json()
        self.assertEqual(s["filters"]["min_score"], None)
        self.assertEqual((s["total_observations"], s["matured_observations"], s["pending_observations"],
                          s["certified_observations"]), (6, 4, 1, 4))
        self.assertAlmostEqual(s["metrics"]["average_return"], 0.02)
        self.assertAlmostEqual(s["metrics"]["median_excess_return"], 0.02)
        self.assertEqual(s["coverage"]["missing_benchmark"], 1)
        self.assertEqual(s["date_range"]["start"][:10], "2026-09-01")
        obs = self.get("pro@example.com", "/v1/outcomes/summary?unit=observation").json()
        self.assertEqual(obs["matured_observations"], 5)

    def test_filters_are_echoed(self):
        s = self.get("pro@example.com", "/v1/outcomes/summary?horizon=3&min_score=80&setup=breakout").json()
        self.assertEqual(s["horizon"], 3)
        self.assertEqual(s["filters"]["min_score"], 80)
        self.assertEqual(s["filters"]["setup"], "breakout")
        self.assertEqual(s["total_observations"], 2)

    def test_validation(self):
        for q in ("horizon=20", "min_score=101", "min_score=90&max_score=80", "start_date=2026-09-05&end_date=2026-09-01",
                  "score_bucket=abc", "unit=best"):
            with self.subTest(q=q):
                self.assertEqual(self.get("pro@example.com", f"/v1/outcomes/summary?{q}").status_code, 422)
        self.assertEqual(self.get("pro@example.com", "/v1/outcomes/scores?buckets=50-70,60-80").status_code, 422)
        self.assertEqual(self.get("pro@example.com", "/v1/outcomes/symbols/AAA?page_size=500").status_code, 422)

    def test_scores_and_symbol(self):
        sc = self.get("pro@example.com", "/v1/outcomes/scores?buckets=40-49,50-59,60-69,70-79,80-89,90-100").json()
        self.assertEqual(len(sc["buckets"]), 6)
        self.assertIn("win_rate", sc["calibration"]["metrics"])
        sym = self.get("pro@example.com", "/v1/outcomes/symbols/aaa?unit=observation&page_size=1").json()
        self.assertEqual(sym["ticker"], "AAA")
        self.assertEqual(sym["observations"]["total"], 2)
        self.assertEqual(len(sym["observations"]["items"]), 1)
        self.assertEqual(len(sym["observations"]["items"][0]["outcomes"]), 3)
        q = self.get("pro@example.com", "/v1/outcomes/query?ticker=bbb&certified_only=true").json()
        self.assertEqual(q["filters"]["ticker"], "BBB")
        self.assertEqual(q["metrics"]["matured_count"], 1)

    def test_cache_reuses_dataset_and_views_until_new_outcomes_mature(self):
        from api import outcomes

        self.get("pro@example.com", "/v1/outcomes/summary")
        self.get("pro@example.com", "/v1/outcomes/summary")
        self.get("pro@example.com", "/v1/outcomes/horizons")
        self.assertEqual(self.fetch.call_count, 1)                   # one dataset load for all
        # new outcomes mature -> stamp changes -> recomputed after the probe TTL
        self.rows = self.rows + [row(70, "NEW", T(8), r1=0.5, r3=0.5, r5=0.5, b5=0.0)]
        self.stamp = ("8", "c", "d")
        outcomes._cache.clear_key("stamp")                          # what STAMP_TTL_S expiry does
        s = self.get("pro@example.com", "/v1/outcomes/summary").json()
        self.assertEqual(self.fetch.call_count, 2)
        self.assertEqual(s["matured_observations"], 5)

    def test_database_outage_serves_last_good_dataset_marked_stale(self):
        from api import outcomes

        self.get("pro@example.com", "/v1/outcomes/summary")
        self.probe.side_effect = RuntimeError("neon down")
        outcomes._cache.clear_key("stamp")
        s = self.get("pro@example.com", "/v1/outcomes/summary").json()
        self.assertTrue(s["dataset"]["stale"])
        self.assertEqual(s["matured_observations"], 4)

    def test_cold_outage_is_503(self):
        self.probe.side_effect = RuntimeError("neon down")
        r = self.get("pro@example.com", "/v1/outcomes/summary")
        self.assertEqual(r.status_code, 503)

    def test_empty_dataset(self):
        self.rows = []
        s = self.get("pro@example.com", "/v1/outcomes/summary").json()
        self.assertEqual(s["total_observations"], 0)
        self.assertEqual(s["metrics"]["evidence_quality"], "INSUFFICIENT")

    def test_ai_evidence_text(self):
        from api import outcomes

        text = outcomes.ai_evidence("AAA")
        self.assertIn("Historical HSF evidence for AAA", text)
        self.assertIn("filters: none", text)
        self.probe.side_effect = RuntimeError("down")
        outcomes.clear_cache()
        self.assertIsNone(outcomes.ai_evidence("AAA"))               # never breaks the AI note

    def test_ai_ticker_note_receives_full_evidence_and_rules(self):
        import pandas as pd

        from api import ai

        seen = {}
        df = pd.DataFrame([{"Ticker": "AAA", "BreakoutScore": 80}])
        with mock.patch("api.ai.scan_df", return_value=(1, df)), \
                mock.patch("api.ai._ask", side_effect=lambda **k: seen.update(k) or "note"):
            ai.ticker_note("pro@example.com", "AAA")
        self.assertIn("Historical HSF evidence for AAA", seen["user"])
        self.assertIn("filters: none", seen["user"])
        self.assertIn("5 trading days", seen["user"])
        self.assertIn("Never present it as a prediction", seen["system"])

    def test_openapi_documents_the_routes(self):
        spec = self.client.get("/openapi.json").json()
        for path in ("/v1/outcomes/summary", "/v1/outcomes/symbols/{ticker}"):
            self.assertIn(path, spec["paths"])
        self.assertIn("OutcomeSummary", spec["components"]["schemas"])


class BenchmarkScoringTests(unittest.TestCase):
    """SPY is scored by the same function as the stock, over the same window."""

    def _bars(self, closes, start=dt.date(2026, 9, 1)):
        import pandas as pd

        idx = pd.to_datetime([start + dt.timedelta(days=i) for i in range(len(closes))])
        return pd.DataFrame({"Close": closes, "High": closes, "Low": closes}, index=idx)

    def test_benchmark_returns_use_score_signal(self):
        from analytics.signal_outcomes import benchmark_returns, score_signal

        bars = self._bars([100, 101, 102, 103, 104, 110, 111])
        rets = benchmark_returns(bars, dt.datetime(2026, 9, 1, 14, tzinfo=dt.timezone.utc))
        full = score_signal(bars, dt.datetime(2026, 9, 1, 14, tzinfo=dt.timezone.utc))
        self.assertEqual(rets, {k: full[k] for k in ("return_1d", "return_3d", "return_5d")})
        self.assertAlmostEqual(rets["return_5d"], 0.10)
        self.assertIsNone(benchmark_returns(self._bars([100, 101]), dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)))
        self.assertIsNone(benchmark_returns(None, dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)))

    def test_scoring_saves_benchmark_only_for_matured_rows(self):
        from analytics import signal_outcomes as so

        fired = dt.datetime(2026, 9, 1, 14, tzinfo=dt.timezone.utc)
        signals = [{"id": 1, "ticker": "AAA", "fired_at": fired}, {"id": 2, "ticker": "BBB", "fired_at": fired}]
        bars = {"AAA": self._bars([10, 11, 12, 13, 14, 15, 16]), "SPY": self._bars([100] * 7)}
        saved_bench = []
        with mock.patch("db.signal_outcomes.list_pending_outcomes", return_value=signals), \
                mock.patch("db.signal_outcomes.save_outcome", return_value=True), \
                mock.patch("db.signal_outcomes.save_benchmark", side_effect=lambda **k: saved_bench.append(k) or True), \
                mock.patch("data.price_alpaca.download_multi_alpaca", return_value=bars) as dl:
            self.assertEqual(so.score_pending_signal_outcomes(), 2)
        self.assertIn("SPY", dl.call_args[0][0])
        self.assertEqual([k["signal_id"] for k in saved_bench], [1])          # BBB had no bars -> no benchmark
        self.assertEqual(saved_bench[0]["return_5d"], 0.0)

    def test_backfill(self):
        from analytics import signal_outcomes as so

        fired = dt.datetime(2026, 9, 1, 14, tzinfo=dt.timezone.utc)
        saved = []
        with mock.patch("db.signal_outcomes.list_benchmark_backfill", return_value=[{"id": 5, "fired_at": fired}]), \
                mock.patch("db.signal_outcomes.save_benchmark", side_effect=lambda **k: saved.append(k) or True), \
                mock.patch("data.price_alpaca.download_multi_alpaca", return_value={"SPY": self._bars([100, 102] * 4)}):
            self.assertEqual(so.backfill_benchmark_returns(), 1)
        self.assertEqual(saved[0]["symbol"], "SPY")
        self.assertAlmostEqual(saved[0]["return_1d"], 0.02)
        with mock.patch("db.signal_outcomes.list_benchmark_backfill", return_value=[]):
            self.assertEqual(so.backfill_benchmark_returns(), 0)


if __name__ == "__main__":
    unittest.main()
