"""System Health capture checks count scanner research observations only.

Stair-stepper observations (a separate Day Trader study) have no research
cohort, research_metadata or scan_id by design; counting them raised
MISSING_COHORT_LABEL, METADATA_INCOMPLETE and REQUIRED_FIELD_NULLS.
"""
import datetime as dt
import unittest
from unittest import mock

from analytics import forward_readiness as fr
from analytics.stair_step_research import CONTEXT as STAIR_STEP_CONTEXT
from scripts import system_health as health

NOW = dt.datetime(2026, 10, 2, 21, 0, tzinfo=dt.timezone.utc)


def _scanner_obs(i):
    ts = (NOW - dt.timedelta(hours=1)).isoformat()
    return {"observation_id": f"s{i}", "symbol": "AAPL", "context": "scheduled:us_market",
            "timestamp": ts, "scan_timestamp": ts, "research_cohort": "CANDIDATE",
            "research_metadata": {"scoring": {}}, "market": {"price": 1.0},
            "market_context": {"scan_id": "scan-1", "research_cohort": "CANDIDATE"}}


def _stair_obs(i):
    ts = (NOW - dt.timedelta(hours=1)).isoformat()
    return {"observation_id": f"t{i}", "symbol": "MSFT", "context": STAIR_STEP_CONTEXT,
            "timestamp": ts, "scan_timestamp": ts, "market": {"price": 2.0}, "stair_step": {}}


class StairStepScopeTests(unittest.TestCase):
    def test_stair_stepper_rows_are_left_out(self):
        rows = [_scanner_obs(i) for i in range(6)] + [_stair_obs(i) for i in range(4)]
        with mock.patch("db.hsf_observations.load_recent_observations", return_value=rows):
            obs = health.recent_observations()
        self.assertEqual({o["context"] for o in obs}, {"scheduled:us_market"})
        self.assertEqual(len(obs), 6)

    def test_summary_of_scanner_rows_has_no_capture_warnings(self):
        rows = [_scanner_obs(i) for i in range(6)] + [_stair_obs(i) for i in range(4)]
        with mock.patch("db.hsf_observations.load_recent_observations", return_value=rows), \
                mock.patch.object(fr, "epoch_start", return_value=NOW - dt.timedelta(days=10)):
            summary = health.observations_summary(health.recent_observations(), NOW, None)
        self.assertEqual(summary["untagged"], 0)
        self.assertEqual(summary["metadata_block_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
