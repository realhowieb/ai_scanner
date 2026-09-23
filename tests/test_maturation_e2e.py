"""Run 38B — maturation eligibility, worker, and end-to-end lifecycle tests."""
import datetime as dt
import sqlite3
import unittest

from analytics import observation_capture as oc
from analytics import scanner_performance as sp
from db import hsf_observations as store
from scripts import mature_observations as worker


def _obs(scan_ts="2026-08-25T14:00:00+00:00"):
    return oc.build_scan_observations(
        [{"Ticker": "NVDA", "BreakoutScore": 8.0, "Last": 100.0, "Volume": 1e6,
          "PctChange": 1.0, "GapPct": 0.5, "Trend20D%": 2.0, "Trend10D%": 1.0,
          "VolRel20": 2.5, "DollarVol20": 5e8, "Volatility20D%": 3.0, "IsBreakout": True}],
        universe="SP500", scan_timestamp=scan_ts, session="morning")[0]


def _bars(day="2026-08-25", start_hhmm=(14, 0), n=70, base=100.0):
    """Rising 1-min bars starting at the anchor minute."""
    out = []
    h, m = start_hhmm
    t0 = dt.datetime(2026, 8, 25, h, m, tzinfo=dt.timezone.utc)
    for i in range(n):
        out.append({"t": (t0 + dt.timedelta(minutes=i)).isoformat(),
                    "o": base + i * 0.1, "h": base + i * 0.1 + 0.05,
                    "l": base + i * 0.1 - 0.05, "c": base + i * 0.1, "v": 1000})
    return out


class EligibilityTests(unittest.TestCase):
    def test_per_horizon_progression(self):
        anchor = "2026-08-25T14:00:00+00:00"
        # 10 min later: nothing ready (slack 15)
        e = oc.horizon_eligibility(anchor, "2026-08-25T14:10:00+00:00", [])
        self.assertEqual(set(v for v in e.values()), {"not_ready"})
        # 25 min later: +5m ready (5+15), others not
        e = oc.horizon_eligibility(anchor, "2026-08-25T14:25:00+00:00", [])
        self.assertEqual(e["+5m"], "ready")
        self.assertEqual(e["+15m"], "not_ready")
        # 80 min later: all ready
        e = oc.horizon_eligibility(anchor, "2026-08-25T15:20:00+00:00", [])
        self.assertTrue(all(v == "ready" for v in e.values()))

    def test_existing_marked_already(self):
        e = oc.horizon_eligibility("2026-08-25T14:00:00+00:00",
                                   "2026-08-25T16:00:00+00:00", ["+5m", "+15m"])
        self.assertEqual(e["+5m"], "already")
        self.assertEqual(e["+30m"], "ready")


class WorkerTests(unittest.TestCase):
    def test_partial_then_full_maturation_idempotent(self):
        obs = _obs()
        saved = {}

        def fake_save(o):
            key = (o["observation_id"], o["horizon"])
            if key in saved:
                return False  # first-write-wins
            saved[key] = o
            return True

        # Run at +25m → only +5m matures
        r1 = worker.mature_observations(
            [obs], now="2026-08-25T14:25:00+00:00", fetch_bars=lambda s, d: _bars(),
            save_fn=fake_save)
        self.assertEqual(r1["horizons"]["+5m"]["new"], 1)
        self.assertEqual(r1["horizons"]["+15m"]["not_ready"], 1)

        # attach outcome to obs, rerun at +80m → +15/30/60 mature, +5m skipped
        obs["outcomes"] = {"+5m": saved[(obs["observation_id"], "+5m")]}
        r2 = worker.mature_observations(
            [obs], now="2026-08-25T15:20:00+00:00", fetch_bars=lambda s, d: _bars(),
            save_fn=fake_save)
        self.assertEqual(r2["horizons"]["+5m"]["already"], 1)  # not re-matured
        self.assertEqual(r2["horizons"]["+15m"]["new"], 1)
        self.assertEqual(r2["horizons"]["+60m"]["new"], 1)

    def test_no_bars_is_price_data_unavailable(self):
        r = worker.mature_observations(
            [_obs()], now="2026-08-25T16:00:00+00:00",
            fetch_bars=lambda s, d: [], save_fn=lambda o: True)
        self.assertGreaterEqual(r["failures"].get("PRICE_DATA_UNAVAILABLE", 0), 1)

    def test_provider_error_isolated(self):
        def boom(s, d):
            raise RuntimeError("provider down")
        r = worker.mature_observations(
            [_obs()], now="2026-08-25T16:00:00+00:00", fetch_bars=boom,
            save_fn=lambda o: True)
        self.assertGreaterEqual(r["failures"].get("PROVIDER_ERROR", 0), 1)
        self.assertEqual(r["attached"], 0)  # did not crash

    def test_one_bad_symbol_does_not_block_others(self):
        good, bad = _obs(), _obs()
        bad["symbol"] = "BAD"
        bad["scanners"][0]["direction"] = "long"

        def fetch(sym, d):
            if sym == "BAD":
                raise RuntimeError("x")
            return _bars()
        r = worker.mature_observations(
            [good, bad], now="2026-08-25T16:00:00+00:00", fetch_bars=fetch,
            save_fn=lambda o: True)
        self.assertGreater(r["attached"], 0)                       # good matured
        self.assertGreaterEqual(r["failures"].get("PROVIDER_ERROR", 0), 1)  # bad isolated

    def test_dry_run_writes_nothing(self):
        writes = []
        r = worker.mature_observations(
            [_obs()], now="2026-08-25T16:00:00+00:00", dry_run=True,
            fetch_bars=lambda s, d: _bars(), save_fn=lambda o: writes.append(o) or True)
        self.assertEqual(writes, [])
        self.assertGreater(r["attached"], 0)  # counted as would-write

    def test_max_symbols_bounds_fetches(self):
        # 5 distinct symbols, cap at 2 → only 2 fetched, 3 deferred (backlog drains
        # on later runs). Prevents the CI timeout seen when cohort volume grew.
        def mk(sym, minute):
            o = dict(_obs())
            o["symbol"] = sym
            o["scan_timestamp"] = f"2026-08-25T14:0{minute}:00+00:00"
            return o
        obs = [mk(f"S{i}", i) for i in range(5)]
        fetched = []

        def fetch(sym, d):
            fetched.append(sym)
            return _bars()
        r = worker.mature_observations(
            obs, now="2026-08-25T16:00:00+00:00", fetch_bars=fetch,
            save_fn=lambda o: True, max_symbols=2)
        self.assertEqual(len(fetched), 2)                 # only 2 fetched
        self.assertEqual(r["symbols_with_ready_horizons"], 5)
        self.assertEqual(r["symbols_deferred"], 3)
        # oldest-anchor first: S0, S1 (earliest scan_timestamps) processed
        self.assertEqual(sorted(fetched), ["S0", "S1"])

    def test_no_cap_processes_all(self):
        def mk(sym, minute):
            o = dict(_obs()); o["symbol"] = sym
            o["scan_timestamp"] = f"2026-08-25T14:0{minute}:00+00:00"
            return o
        obs = [mk(f"S{i}", i) for i in range(4)]
        fetched = []
        worker.mature_observations(
            obs, now="2026-08-25T16:00:00+00:00", fetch_bars=lambda s, d: fetched.append(s) or _bars(),
            save_fn=lambda o: True, max_symbols=0)
        self.assertEqual(len(fetched), 4)  # <=0 → no cap


class EndToEndTests(unittest.TestCase):
    def test_full_lifecycle_scan_to_scoreboard(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            # 1-2. scan result -> canonical observation -> persist
            obs = _obs()
            self.assertTrue(store.save_observation(obs, conn=conn))
            # 3. immutable + NO future data inside the original observation
            self.assertNotIn("outcomes", obs)
            for scanner in obs["scanners"]:
                self.assertNotIn("return", str(scanner).lower())
            # 4-7. time advances -> mature -> attach (via store)
            def save_to_db(o):
                return store.save_outcome(o, conn=conn)
            report = worker.mature_observations(
                [store.load_observation(obs["observation_id"], conn=conn)],
                now="2026-08-25T15:20:00+00:00",
                fetch_bars=lambda s, d: _bars(), save_fn=save_to_db)
            self.assertGreater(report["attached"], 0)
            # 8. duplicate maturation does nothing harmful
            report2 = worker.mature_observations(
                [store.load_observation(obs["observation_id"], conn=conn)],
                now="2026-08-25T15:20:00+00:00",
                fetch_bars=lambda s, d: _bars(), save_fn=save_to_db)
            self.assertEqual(report2["attached"], 0)
            # 9. scoreboard consumes the matured canonical record
            matured = store.load_observation(obs["observation_id"], conn=conn)
            self.assertIn("outcomes", matured)
            records = sp.from_canonical_observations([matured])
            self.assertTrue(records)
            self.assertEqual(records[0]["source"], "scheduled")
            board = sp.build_scoreboard_report(records)
            self.assertIn("breakout", board["overall"]["scanners"])
            # original observation timestamp precedes every outcome eval time
            for oc_rec in matured["outcomes"].values():
                self.assertLess(matured["scan_timestamp"], oc_rec["evaluation_time"])
        finally:
            conn.close()


class ResearchFilterTests(unittest.TestCase):
    def test_filter_defaults_to_scheduled_clean(self):
        recs = [
            {"scanner": "A", "source": "scheduled", "fallback": False, "stale": False,
             "feature_completeness": 0.8, "returns": {"5d": 0.01}},
            {"scanner": "A", "source": "manual", "fallback": False, "stale": False,
             "feature_completeness": 0.8, "returns": {"5d": 0.01}},
            {"scanner": "A", "source": "scheduled", "fallback": True, "stale": False,
             "feature_completeness": 0.8, "returns": {"5d": 0.01}},
            {"scanner": "A", "source": "scheduled", "fallback": False, "stale": True,
             "feature_completeness": 0.8, "returns": {"5d": 0.01}},
        ]
        kept = sp.filter_research_records(recs)
        self.assertEqual(len(kept), 1)  # only scheduled, non-fallback, non-stale
        self.assertEqual(len(sp.filter_research_records(recs, source=None)), 2)  # +manual

    def test_sample_readiness(self):
        recs = [{"scanner": "unusual_vol", "returns": {"5m": 0.01}} for _ in range(42)]
        rr = sp.sample_readiness(recs, horizons=["5m"])
        self.assertTrue(rr["scanners"]["unusual_vol"]["horizons"]["5m"]["ready"])
        self.assertEqual(rr["scanners"]["unusual_vol"]["horizons"]["5m"]["n"], 42)


if __name__ == "__main__":
    unittest.main()
