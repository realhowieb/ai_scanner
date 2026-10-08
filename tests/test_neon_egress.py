"""Neon egress fixes (2026-10-08): saved runs cached by stamp, slim maturation reads,
daily stair-stepper analysis."""
import copy
import datetime as dt
import json
import unittest
from pathlib import Path
from unittest import mock

from db import hsf_observations as store
from scripts import mature_observations as worker
from tests.test_maturation_parity import HIST_NOW
from tests.test_maturation_parity import dataset as parity_dataset
from tests.test_stair_step_research import ANCHOR, _bars, _observation

ROOT = Path(__file__).resolve().parents[1]


def _slim(observations):
    keep = set(worker.OBSERVATION_FIELDS) | {"outcomes"}
    return [{k: v for k, v in o.items() if k in keep and v is not None} for o in observations]


class SlimMaturationRecordsTests(unittest.TestCase):
    """Maturation must give the same report and outcomes from projected records."""

    def _run_both(self, observations, **kw):
        results = []
        for obs in (copy.deepcopy(observations), _slim(observations)):
            saved = []
            report = worker.mature_observations(obs, save_fn=lambda o: saved.append(o) or True, **kw)
            report.pop("generated_at", None)
            results.append((report, saved))
        return results

    def test_market_observations(self):
        obs, _ = parity_dataset(runs=1)
        t0 = dt.datetime.fromisoformat(obs[0]["scan_timestamp"])
        bars = [{"t": (t0 + dt.timedelta(minutes=i)).isoformat(), "c": 10 + i * 0.01} for i in range(200)]
        (ra, sa), (rb, sb) = self._run_both(
            obs, now=HIST_NOW, fetch_bars_batch=lambda s, x, y: {k: bars for k in s},
            retire_after=None, exclusion_reason=lambda s: None)
        self.assertTrue(sa)
        self.assertEqual(ra, rb)
        self.assertEqual(sa, sb)

    def test_stair_step_observations(self):
        obs = [_observation(), _observation(direction="down")]
        (ra, sa), (rb, sb) = self._run_both(
            obs, now=ANCHOR + dt.timedelta(minutes=31), slack_min=0,
            fetch_bars=lambda _s, _d: _bars(), max_symbols=0,
            exclusion_reason=lambda _s: None, retire_after=None)
        self.assertTrue(sa)
        self.assertEqual(ra, rb)
        self.assertEqual(sa, sb)


class RecordSelectTests(unittest.TestCase):
    def test_postgres_projects_only_named_keys(self):
        sql = store._record_select("o.", ("symbol", "market"), False)
        self.assertEqual(sql, "jsonb_build_object('symbol', o.record->'symbol', "
                              "'market', o.record->'market') AS record")

    def test_sqlite_and_no_fields_read_the_whole_record(self):
        self.assertEqual(store._record_select("o.", ("symbol",), True), "o.record")
        self.assertEqual(store._record_select("", None, False), "record")

    def test_unsafe_keys_are_dropped(self):
        sql = store._record_select("", ("symbol", "x'); DROP TABLE runs; --"), False)
        self.assertEqual(sql, "jsonb_build_object('symbol', record->'symbol') AS record")

    def test_sqlite_load_with_fields_keeps_behaviour(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            obs, _ = parity_dataset(runs=1)
            for o in obs:
                store.save_observation(o, conn=conn)
            full = store.load_recent_observations(limit=100, attach_outcomes=True, conn=conn)
            fielded = store.load_recent_observations(limit=100, attach_outcomes=True,
                                                     fields=worker.OBSERVATION_FIELDS, conn=conn)
            self.assertEqual(full, fielded)  # SQLite has no projection; nothing changes
        finally:
            conn.close()


class RunCacheTests(unittest.TestCase):
    def setUp(self):
        from api import today
        self.today = today
        today.clear_cache()
        self.addCleanup(today.clear_cache)
        self.payload = json.dumps([{"Ticker": "AAA", "Price": 1.0}])

    def _patch(self, stamps):
        loads = []

        def load(run_id):
            loads.append(run_id)
            return self.payload

        return loads, mock.patch.multiple("db.runs", load_run_results=load,
                                          load_run_stamp=lambda run_id: stamps[0])

    def _expire_short_entries(self):
        self.today._cache.clear()  # what CACHE_TTL_S expiry does after a minute

    def test_unchanged_run_is_downloaded_once(self):
        stamps = ["2026-10-08 13:35|100"]
        loads, patcher = self._patch(stamps)
        with patcher:
            for _ in range(3):
                self.assertEqual(list(self.today.run_df(7)["Ticker"]), ["AAA"])
                self._expire_short_entries()
        self.assertEqual(loads, [7])

    def test_rewritten_snapshot_is_reloaded(self):
        stamps = ["2026-10-08 13:35|100"]
        loads, patcher = self._patch(stamps)
        with patcher:
            self.today.run_df(7)
            self._expire_short_entries()
            stamps[0] = "2026-10-08 14:35|120"  # save_daily_snapshot rewrote the row
            self.today.run_df(7)
        self.assertEqual(loads, [7, 7])

    def test_without_a_stamp_falls_back_to_the_short_cache(self):
        loads, patcher = self._patch([None])
        with patcher:
            self.today.run_df(7)
            self.today.run_df(7)
            self._expire_short_entries()
            self.today.run_df(7)
        self.assertEqual(loads, [7, 7])

    def test_run_cache_is_bounded(self):
        stamps = ["s"]
        _loads, patcher = self._patch(stamps)
        with patcher:
            for rid in range(self.today.RUN_CACHE_MAX_ENTRIES + 5):
                self.today.run_df(rid)
        self.assertLessEqual(self.today._run_cache.size(), self.today.RUN_CACHE_MAX_ENTRIES)


class StairStepperScheduleTests(unittest.TestCase):
    def test_analysis_runs_only_in_the_post_close_slot(self):
        wf = (ROOT / ".github/workflows/mature-observations.yml").read_text()
        step = wf.split("- name: Analyze Stair-stepper outcomes", 1)[1].split("- name:", 1)[0]
        self.assertIn("2230", step)
        self.assertIn("STAIR_STEPPER", step)


if __name__ == "__main__":
    unittest.main()
