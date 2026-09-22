"""Run 37 — coverage funnel + data-health tests."""
import unittest

from analytics import coverage as cov


class FailureTaxonomyTests(unittest.TestCase):
    def test_provider_reasons_map_to_taxonomy(self):
        self.assertEqual(cov.classify_failure("rate limit 429"), "RATE_LIMIT")
        self.assertEqual(cov.classify_failure("request timed out"), "TIMEOUT")
        self.assertEqual(cov.classify_failure("empty"), "NO_PRICE_DATA")
        self.assertEqual(cov.classify_failure("error_download"), "API_ERROR")
        self.assertEqual(cov.classify_failure("invalid_frame"), "INDICATOR_FAILURE")

    def test_scan_side_reasons_direct(self):
        self.assertEqual(cov.classify_failure("DELISTED"), "DELISTED")
        self.assertEqual(cov.classify_failure("insufficient history"), "INSUFFICIENT_HISTORY")
        self.assertEqual(cov.classify_failure("untradable by policy"), "FILTERED_BY_POLICY")

    def test_unknown_default(self):
        self.assertEqual(cov.classify_failure(""), "UNKNOWN")
        self.assertEqual(cov.classify_failure("weird gibberish"), "UNKNOWN")


class FunnelTests(unittest.TestCase):
    def _funnel(self, **kw):
        base = dict(universe_version="combo-2026-09", expected=8412, eligible=8103,
                    attempted=8103, price_success=7964,
                    skipped=[(f"T{i}", "timeout") for i in range(139)],
                    indicator_complete=7812, results=100, market_session="regular")
        base.update(kw)
        return cov.build_coverage_funnel(**base)

    def test_coverage_accounting_and_percentages(self):
        f = self._funnel()
        c = f["counts"]
        self.assertEqual(c["expected"], 8412)
        self.assertEqual(c["excluded"], 8412 - 8103)
        self.assertEqual(c["price_failure"], 8103 - 7964)
        self.assertAlmostEqual(f["coverage_pct"], round(7964 / 8103, 4))
        self.assertAlmostEqual(f["percentages"]["eligible_pct"], round(8103 / 8412, 4))

    def test_failed_symbols_get_reasons(self):
        f = self._funnel(skipped=[("AAA", "rate limit"), ("BBB", "timeout"),
                                  ("CCC", "empty")])
        self.assertEqual(f["failure_reasons"],
                         {"RATE_LIMIT": 1, "TIMEOUT": 1, "NO_PRICE_DATA": 1})
        self.assertEqual(f["counts"]["failures"], 3)
        self.assertEqual(f["top_failure_categories"][0]["count"], 1)

    def test_duplicate_symbols_counted_each(self):
        f = self._funnel(skipped=[("AAA", "timeout"), ("AAA", "timeout")])
        self.assertEqual(f["failure_reasons"]["TIMEOUT"], 2)

    def test_partial_indicator_optional(self):
        f = self._funnel(indicator_complete=None)
        self.assertIsNone(f["counts"]["indicator_complete"])
        self.assertIsNone(f["percentages"]["indicator_coverage_pct"])

    def test_empty_universe(self):
        f = self._funnel(expected=0, eligible=0, attempted=0, price_success=0, skipped=[])
        self.assertIsNone(f["coverage_pct"])
        self.assertIsNone(f["percentages"]["eligible_pct"])


class HealthTests(unittest.TestCase):
    def _funnel(self, price_success, eligible=1000, skipped=None):
        return cov.build_coverage_funnel(
            expected=eligible, eligible=eligible, attempted=eligible,
            price_success=price_success, skipped=skipped or [])

    def test_healthy(self):
        h = cov.classify_health(self._funnel(980))
        self.assertEqual(h["state"], "HEALTHY")

    def test_degraded_low_coverage(self):
        h = cov.classify_health(self._funnel(850))  # 85% < healthy 95%
        self.assertEqual(h["state"], "DEGRADED")

    def test_failed_zero_coverage(self):
        h = cov.classify_health(self._funnel(0))
        self.assertEqual(h["state"], "FAILED")

    def test_failed_empty_universe(self):
        h = cov.classify_health(self._funnel(0, eligible=0))
        self.assertEqual(h["state"], "FAILED")

    def test_stale_universe_age(self):
        h = cov.classify_health(self._funnel(990), universe_age_hours=500)
        self.assertEqual(h["state"], "STALE")

    def test_stale_price_age(self):
        h = cov.classify_health(self._funnel(990), price_ts_age_min=120)
        self.assertEqual(h["state"], "STALE")

    def test_stale_beats_degraded_precedence(self):
        # low coverage AND stale → STALE wins
        h = cov.classify_health(self._funnel(850), universe_age_hours=500)
        self.assertEqual(h["state"], "STALE")

    def test_100_percent_scan_healthy(self):
        f = self._funnel(1000)
        h = cov.classify_health(f)
        self.assertEqual(h["state"], "HEALTHY")
        self.assertEqual(f["coverage_pct"], 1.0)


class ReportTests(unittest.TestCase):
    def test_report_bundle_shape(self):
        f = cov.build_coverage_funnel(universe_version="v1", expected=100, eligible=98,
                                      attempted=98, price_success=95,
                                      skipped=[("X", "timeout")], indicator_complete=94,
                                      results=10, duration_sec=42.0)
        h = cov.classify_health(f)
        rep = cov.coverage_report(f, h, stale_symbols=3)
        self.assertEqual(rep["schema"], "hsf-coverage-1.0")
        self.assertIn("Coverage report", rep["text"])
        self.assertEqual(rep["summary"]["state"], "HEALTHY")
        self.assertEqual(rep["summary"]["stale_symbols"], 3)
        self.assertEqual(rep["summary"]["price_success"], 95)
        self.assertIn("price_coverage_pct", rep["summary"])


class WiringTests(unittest.TestCase):
    def test_coverage_artifact_written_from_sink(self):
        # Simulates the cron path: a coverage_sink populated by the scan (even a
        # partial/degraded one) -> funnel -> report -> artifact on disk.
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from scheduler import cron_runner

        sink = {"attempted": 500, "price_success": 300,
                "skipped": [(f"T{i}", "timeout") for i in range(200)]}
        funnel = cov.build_coverage_funnel(
            universe_version="SP500", expected=505, eligible=500,
            attempted=sink["attempted"], price_success=sink["price_success"],
            skipped=sink["skipped"], results=12)
        health = cov.classify_health(funnel)
        self.assertEqual(health["state"], "DEGRADED")  # 60% coverage
        report = cov.coverage_report(funnel, health)

        with tempfile.TemporaryDirectory() as d:
            with patch.object(cron_runner, "COVERAGE_DIR", Path(d) / "automation"):
                cron_runner._write_coverage_artifact("SP500", report)
                path = Path(d) / "automation" / "coverage_sp500.json"
                self.assertTrue(path.exists())
                loaded = json.loads(path.read_text())
                self.assertEqual(loaded["schema"], "hsf-coverage-1.0")
                self.assertEqual(loaded["health"]["state"], "DEGRADED")
                self.assertEqual(loaded["funnel"]["counts"]["price_failure"], 200)

    def test_engine_accepts_coverage_sink_param(self):
        import inspect

        from scan.engine import run_breakout_scan
        self.assertIn("coverage_sink",
                      inspect.signature(run_breakout_scan).parameters)


if __name__ == "__main__":
    unittest.main()
