"""Run 45 — scan reliability: health-gated snapshot safety, telemetry, taxonomy."""
import unittest

from analytics import scan_reliability as sr


class EventTaxonomyTests(unittest.TestCase):
    def test_rate_limit_timeout_policy_distinguished(self):
        skipped = ([("A", "rate limit 429")] * 3 + [("B", "timed out")] * 2
                   + [("C", "skipped_yf_fallback")] * 5 + [("D", "recent_missing:no bars")] * 4)
        e = sr.derive_event_counts(skipped)
        self.assertEqual(e["rate_limit_events"], 3)
        self.assertEqual(e["timeout_events"], 2)
        self.assertEqual(e["policy_filtered_events"], 5)
        self.assertEqual(e["no_price_data_events"], 4)
        # provider-trouble excludes the intentional policy skips
        self.assertEqual(e["provider_trouble_events"], 3 + 2 + 4)

    def test_policy_vs_provider_failure_distinction(self):
        # "195 intentionally skipped" must NOT count as provider trouble.
        e = sr.derive_event_counts([("X", "skipped_yf_fallback")] * 195)
        self.assertEqual(e["policy_filtered_events"], 195)
        self.assertEqual(e["provider_trouble_events"], 0)


class HealthMappingTests(unittest.TestCase):
    def test_operational_states(self):
        self.assertEqual(sr.to_operational_state("HEALTHY"), "HEALTHY")
        self.assertEqual(sr.to_operational_state("FAILED"), "FAILED")
        self.assertEqual(sr.to_operational_state("DEGRADED"), "DEGRADED")
        self.assertEqual(sr.to_operational_state("STALE"), "DEGRADED")  # folded
        self.assertEqual(sr.to_operational_state(None), "DEGRADED")


class SnapshotSafetyTests(unittest.TestCase):
    def test_healthy_promotes(self):
        d = sr.snapshot_decision("HEALTHY", 0.983)
        self.assertTrue(d["promote"])

    def test_degraded_not_promoted(self):
        d = sr.snapshot_decision("DEGRADED", 0.60)
        self.assertFalse(d["promote"])
        self.assertIn("degraded", d["reason"])

    def test_failed_not_promoted(self):
        self.assertFalse(sr.snapshot_decision("FAILED", 0.0)["promote"])

    def test_healthy_but_low_coverage_not_promoted(self):
        # defensive: even if health said HEALTHY, sub-floor coverage blocks it
        d = sr.snapshot_decision("HEALTHY", 0.80)
        self.assertFalse(d["promote"])
        self.assertIn("floor", d["reason"])

    def test_stale_folds_to_not_promoted(self):
        self.assertFalse(sr.snapshot_decision("STALE", 0.99)["promote"])


class PerformanceRecordTests(unittest.TestCase):
    def _rec(self, **kw):
        base = dict(run_id="r1", started_at="t0", completed_at="t1",
                    market_session="regular", universe="US_MARKET",
                    universe_source="live", provider_asset_count=14357,
                    eligible_symbol_count=11827, attempted_symbol_count=11826,
                    priced_symbol_count=11631, skipped_symbol_count=195,
                    candidate_count=100, coverage_percentage=0.983,
                    coverage_health="HEALTHY",
                    timings={"total_runtime_seconds": 125.0},
                    skipped=[("X", "skipped_yf_fallback")] * 195,
                    snapshot_promoted=True)
        base.update(kw)
        return sr.build_performance_record(**base)

    def test_throughput_calculation(self):
        rec = self._rec()
        self.assertAlmostEqual(rec["symbols_per_second"], round(11826 / 125.0, 1))
        self.assertEqual(rec["operational_state"], "HEALTHY")
        self.assertEqual(rec["policy_filtered_events"], 195)
        self.assertEqual(rec["provider_trouble_events"], 0)

    def test_null_timing_does_not_crash(self):
        rec = self._rec(timings={})  # no total runtime
        self.assertIsNone(rec["symbols_per_second"])
        self.assertIsNone(rec["timings"]["total_runtime_seconds"])
        # batch counters honestly null
        self.assertIsNone(rec["batches_attempted"])

    def test_missing_fields_safe(self):
        rec = sr.build_performance_record(
            run_id=None, started_at=None, completed_at=None, market_session=None,
            universe=None, universe_source=None, provider_asset_count=None,
            eligible_symbol_count=None, attempted_symbol_count=None,
            priced_symbol_count=None, skipped_symbol_count=None, candidate_count=None,
            coverage_percentage=None, coverage_health=None)
        self.assertIsNone(rec["symbols_per_second"])
        self.assertEqual(rec["operational_state"], "DEGRADED")

    def test_run_summary_renders(self):
        text = sr.render_run_summary(self._rec())
        self.assertIn("US_MARKET SCAN", text)
        self.assertIn("Coverage:   98.3%", text)
        self.assertIn("Snapshot promoted: YES", text)
        self.assertIn("Health: HEALTHY", text)

    def test_run_summary_suppression_reason(self):
        rec = self._rec(coverage_health="DEGRADED", coverage_percentage=0.6,
                        snapshot_promoted=False,
                        snapshot_suppression_reason="degraded scan — not promoted")
        text = sr.render_run_summary(rec)
        self.assertIn("Snapshot promoted: NO", text)
        self.assertIn("degraded", text)


class RepeatabilityTests(unittest.TestCase):
    def _hist(self, n=5, **over):
        base = dict(operational_state="HEALTHY", eligible_symbol_count=11800,
                    coverage_percentage=0.98, symbols_per_second=94.0,
                    provider_trouble_events=2, timings={"total_runtime_seconds": 125})
        base.update(over)
        return [dict(base) for _ in range(n)]

    def test_insufficient_history(self):
        self.assertFalse(sr.compare_to_recent({}, self._hist(1))["comparable"])

    def test_normal_run_no_flags(self):
        cur = dict(eligible_symbol_count=11850, coverage_percentage=0.982,
                   symbols_per_second=93.0, provider_trouble_events=3,
                   timings={"total_runtime_seconds": 130})
        res = sr.compare_to_recent(cur, self._hist())
        self.assertTrue(res["comparable"])
        self.assertFalse(res["is_anomalous"])

    def test_coverage_drop_flagged(self):
        cur = dict(eligible_symbol_count=11800, coverage_percentage=0.85,
                   symbols_per_second=90.0, provider_trouble_events=2,
                   timings={"total_runtime_seconds": 125})
        res = sr.compare_to_recent(cur, self._hist())
        self.assertTrue(res["is_anomalous"])
        self.assertTrue(any("coverage" in f for f in res["flags"]))

    def test_provider_trouble_spike_flagged(self):
        cur = dict(eligible_symbol_count=11800, coverage_percentage=0.98,
                   symbols_per_second=94.0, provider_trouble_events=500,
                   timings={"total_runtime_seconds": 125})
        res = sr.compare_to_recent(cur, self._hist())
        self.assertTrue(any("provider-trouble" in f for f in res["flags"]))

    def test_normal_universe_variation_not_flagged(self):
        # listings/delistings: ±5% universe change is normal, not an anomaly
        cur = dict(eligible_symbol_count=int(11800 * 1.05), coverage_percentage=0.98,
                   symbols_per_second=94.0, provider_trouble_events=2,
                   timings={"total_runtime_seconds": 125})
        self.assertFalse(sr.compare_to_recent(cur, self._hist())["is_anomalous"])


class OverlapLockTests(unittest.TestCase):
    def test_lock_acquire_and_release(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from scheduler import cron_runner
        lock = Path(tempfile.mkdtemp()) / "cron_scan.lock"
        with patch.object(cron_runner, "_SCAN_LOCK", lock):
            self.assertTrue(cron_runner._acquire_scan_lock())     # first acquires
            self.assertFalse(cron_runner._acquire_scan_lock())    # second blocked
            cron_runner._release_scan_lock()
            self.assertTrue(cron_runner._acquire_scan_lock())     # freed → acquires

    def test_stale_lock_reclaimed(self):
        import os
        import tempfile
        import time
        from pathlib import Path
        from unittest.mock import patch

        from scheduler import cron_runner
        lock = Path(tempfile.mkdtemp()) / "cron_scan.lock"
        lock.write_text("old")
        os.utime(lock, (time.time() - 4000, time.time() - 4000))  # very old
        with patch.object(cron_runner, "_SCAN_LOCK", lock):
            self.assertTrue(cron_runner._acquire_scan_lock(max_age_s=1800))  # reclaimed


if __name__ == "__main__":
    unittest.main()
