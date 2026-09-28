"""Maturation must finish inside its CI timeout (2026-09-28 incident).

Both runs that day were cancelled at the 20-minute limit: a real 2000-symbol run
saved ~5k outcomes and each save opened its own database connection. The worker
now keeps ONE connection per run and stops starting new work at a time budget,
deferring the rest to the next run. Outcome values and write semantics
(INSERT ... ON CONFLICT DO NOTHING, commit per row) are unchanged.
"""
import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from analytics import observation_capture as oc
from db import hsf_observations as store
from scripts import mature_observations as worker

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-08-25T15:20:00+00:00"
TICKERS = ("NVDA", "AAPL", "MSFT", "AMZN")


def _observations(tickers=TICKERS):
    rows = [{"Ticker": t, "BreakoutScore": 8.0, "Last": 100.0, "Volume": 1e6, "PctChange": 1.0,
             "GapPct": 0.5, "Trend20D%": 2.0, "Trend10D%": 1.0, "VolRel20": 2.5,
             "DollarVol20": 5e8, "Volatility20D%": 3.0, "IsBreakout": True} for t in tickers]
    return oc.build_scan_observations(rows, universe="SP500",
                                      scan_timestamp="2026-08-25T14:00:00+00:00", session="morning")


def _bars(n=70):
    t0 = dt.datetime(2026, 8, 25, 14, 0, tzinfo=dt.timezone.utc)
    return [{"t": (t0 + dt.timedelta(minutes=i)).isoformat(), "o": 100 + i * 0.1,
             "h": 100 + i * 0.1, "l": 100 + i * 0.1, "c": 100 + i * 0.1, "v": 1000} for i in range(n)]


def _fetch(symbols, _start, _end):
    return {s: _bars() for s in symbols}


class FakeTime:
    """Time moves only when work happens: each fetch batch / each saved outcome."""

    def __init__(self, fetch_cost=0.0, save_cost=0.0):
        self.t, self.fetch_cost, self.save_cost = 0.0, fetch_cost, save_cost

    def clock(self):
        return self.t

    def fetch(self, symbols, start, end):
        self.t += self.fetch_cost
        return _fetch(symbols, start, end)


class RunSaverTests(unittest.TestCase):
    def test_one_connection_for_many_saves_and_duplicates_still_detected(self):
        conn = sqlite3.connect(":memory:")
        opens = []
        saver = worker._RunSaver(connect=lambda: opens.append(1) or conn)
        outcomes = [{"observation_id": f"o{i}", "horizon": "+5m", "schema_version": "1"} for i in range(5)]
        self.assertEqual([saver(o) for o in outcomes], [True] * 5)
        self.assertFalse(saver(outcomes[0]))                      # first write wins
        self.assertEqual(len(opens), 1)                            # was: one connection per save
        rows = conn.execute("SELECT COUNT(*) FROM hsf_observation_outcomes").fetchone()[0]
        self.assertEqual(rows, 5)
        saver.close()

    def _pg_like(self, statuses):
        """A psycopg-like conn whose transaction status is read from `statuses`."""
        seq = iter(statuses)

        class Info:
            @property
            def transaction_status(self):
                return SimpleNamespace(name=next(seq))

        conn = mock.MagicMock()
        conn.closed = conn.broken = False
        conn.info = Info()
        return conn

    def test_aborted_transaction_is_rolled_back_and_retried_not_counted_as_already(self):
        conn = self._pg_like(["INERROR"])
        saver = worker._RunSaver(connect=lambda: conn)
        with mock.patch("db.hsf_observations.save_outcome", side_effect=[False, True]):
            self.assertTrue(saver({"observation_id": "x", "horizon": "+5m"}))
        conn.rollback.assert_called_once()

    def test_persistent_failure_raises_so_it_is_reported_as_database_error(self):
        conn = self._pg_like(["INERROR", "INERROR"])
        saver = worker._RunSaver(connect=lambda: conn)
        with mock.patch("db.hsf_observations.save_outcome", return_value=False):
            with self.assertRaises(RuntimeError):
                saver({"observation_id": "x", "horizon": "+5m"})

    def test_broken_connection_is_reopened(self):
        broken = mock.MagicMock(closed=False, broken=True)
        fresh = sqlite3.connect(":memory:")
        conns = iter([broken, fresh])
        saver = worker._RunSaver(connect=lambda: next(conns))
        with mock.patch("db.hsf_observations.save_outcome",
                        side_effect=lambda o, conn=None: conn is fresh):
            self.assertTrue(saver({"observation_id": "x", "horizon": "+5m"}))
        self.assertEqual(saver.reopened, 1)

    def test_no_database_keeps_the_old_per_row_path(self):
        saver = worker._RunSaver(connect=lambda: None)
        with mock.patch("db.hsf_observations.save_outcome", return_value=True) as save:
            self.assertTrue(saver({"observation_id": "x", "horizon": "+5m"}))
        self.assertEqual(save.call_args.kwargs, {})


class DefaultSavePathTests(unittest.TestCase):
    """The run closes its connection at the end, so these use a file database."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = f"{self.tmp.name}/obs.db"

    def test_a_real_run_opens_one_connection_and_writes_every_outcome(self):
        opens = []
        with mock.patch("db.engine.get_neon_conn",
                        side_effect=lambda: opens.append(1) or sqlite3.connect(self.path)):
            report = worker.mature_observations(_observations(), now=NOW, exclusion_reason=lambda s: None,
                                                fetch_bars_batch=_fetch)
        self.assertEqual(len(opens), 1)
        self.assertEqual(report["attached"], len(TICKERS) * 4)      # 4 horizons each
        self.assertEqual(report["db_reconnects"], 0)
        n = sqlite3.connect(self.path).execute("SELECT COUNT(*) FROM hsf_observation_outcomes").fetchone()[0]
        self.assertEqual(n, report["attached"])

    def test_outcomes_identical_to_the_injected_writer_path(self):
        """Same outcome records whether saved by the run saver or a test writer."""
        with mock.patch("db.engine.get_neon_conn", side_effect=lambda: sqlite3.connect(self.path)):
            worker.mature_observations(_observations(), now=NOW, exclusion_reason=lambda s: None,
                                       fetch_bars_batch=_fetch)
        c2 = sqlite3.connect(":memory:")
        worker.mature_observations(_observations(), now=NOW, exclusion_reason=lambda s: None,
                                   fetch_bars_batch=_fetch,
                                   save_fn=lambda o: store.save_outcome(o, conn=c2))
        q = "SELECT observation_id, horizon, record FROM hsf_observation_outcomes ORDER BY 1, 2"
        self.assertEqual(sqlite3.connect(self.path).execute(q).fetchall(), c2.execute(q).fetchall())


class TimeBudgetTests(unittest.TestCase):
    def run_worker(self, ft=None, save=None, **kw):
        ft = ft or FakeTime()
        saved = []

        def default_save(o):
            ft.t += ft.save_cost
            saved.append(o)
            return True

        report = worker.mature_observations(
            _observations(), now=NOW, exclusion_reason=lambda s: None, fetch_bars_batch=ft.fetch,
            save_fn=save or default_save, batch_size=1, clock=ft.clock, **kw)
        return report, saved

    def test_no_budget_changes_nothing(self):
        report, saved = self.run_worker(FakeTime(fetch_cost=100, save_cost=100))
        self.assertIsNone(report["time_budget"]["hit_during"])
        self.assertEqual(report["backlog"]["processed_symbols"], len(TICKERS))
        self.assertEqual(len(saved), len(TICKERS) * 4)

    def test_budget_during_fetch_defers_the_rest_and_still_saves_what_was_fetched(self):
        trace = []
        # Batches start at t=0,1,2 (< 2.5); the 4th would start at t=3 -> deferred.
        report, saved = self.run_worker(FakeTime(fetch_cost=1.0), time_budget_s=2.5, trace=trace)
        tb, b = report["time_budget"], report["backlog"]
        self.assertEqual(tb["hit_during"], "fetch")
        self.assertEqual(tb["deferred_symbols"], 1)
        self.assertEqual(b["processed_symbols"], 3)
        self.assertEqual(b["deferred_symbols"], 1)
        self.assertEqual(report["unique_symbols_requested"], 3)
        self.assertEqual(report["failures"], {})                        # deferred, not failed
        self.assertEqual(len(saved), 3 * 4)                              # fetched bars are not wasted
        deferred = {t["symbol"] for t in trace if t["status"] == "DEFERRED"}
        self.assertEqual(len(deferred), 1)

    def test_budget_during_save_defers_remaining_symbols(self):
        # Budget 4 + 50% grace = 6: symbols start saving at t=0 and t=4, the third at t=8 is deferred.
        report, saved = self.run_worker(FakeTime(save_cost=1.0), time_budget_s=4.0)
        self.assertEqual(report["time_budget"]["hit_during"], "save")
        self.assertEqual(report["time_budget"]["deferred_symbols"], 2)
        self.assertEqual(report["backlog"]["processed_symbols"], 2)
        self.assertEqual(len(saved), 2 * 4)
        self.assertIsNotNone(report["backlog"]["oldest_pending_age_min"])

    def test_rerun_picks_up_the_deferred_symbols(self):
        store_rows = {}

        def save(o):
            key = (o["observation_id"], o["horizon"])
            if key in store_rows:
                return False
            store_rows[key] = o
            return True

        obs = _observations()
        ft = FakeTime(fetch_cost=1.0)
        worker.mature_observations(obs, now=NOW, exclusion_reason=lambda s: None, fetch_bars_batch=ft.fetch,
                                   save_fn=save, batch_size=1, time_budget_s=2.5, clock=ft.clock)
        first = len(store_rows)
        by_id = {}
        for (oid, h), o in store_rows.items():
            by_id.setdefault(oid, {})[h] = o
        again = [{**o, "outcomes": by_id.get(o["observation_id"], {})} for o in obs]
        worker.mature_observations(again, now=NOW, exclusion_reason=lambda s: None, fetch_bars_batch=_fetch,
                                   save_fn=save, batch_size=1)
        self.assertEqual(first, 3 * 4)
        self.assertEqual(len(store_rows), len(TICKERS) * 4)


class WiringTests(unittest.TestCase):
    def test_cli_default_budget_and_workflow_timeout(self):
        self.assertEqual(worker.TIME_BUDGET_MIN, 12.0)
        src = (ROOT / "scripts" / "mature_observations.py").read_text()
        self.assertIn('ap.add_argument("--time-budget-min", type=float, default=TIME_BUDGET_MIN', src)
        wf = (ROOT / ".github" / "workflows" / "mature-observations.yml").read_text()
        self.assertIn("timeout-minutes: 30", wf)
        self.assertIn("python -m scripts.mature_observations", wf)


if __name__ == "__main__":
    unittest.main()
