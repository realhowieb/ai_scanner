"""ML v4 data readiness: metrics, gates, maturation lifecycle, collection audit,
monitor alerts, the admin-only API, and the outcome cron's open-window guard."""
import datetime as dt
import random
import unittest
from unittest import mock

import pandas as pd

from analytics import ml_readiness as mr
from tests.research_fixtures import opp, scan, ts
from tests.test_api_v1 import DEPS, ApiTestCase

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 9, 1, 0, tzinfo=UTC)


def scored(oid, ticker, fired, *, r5=0.02, computed=None, bench=True, **kw):
    row = opp(oid, ticker, fired, scored=True, r1=r5 / 2, r3=r5, r5=r5, mfe=0.05, mae=-0.02,
              b1=0.001 if bench else None, b3=0.001 if bench else None, b5=0.001 if bench else None, **kw)
    if computed is not None:
        row["outcome_computed_at"] = ts(computed)
    return row


def synthetic(days=60, per_day=10, tickers=150, seed=7, end=dt.date(2026, 10, 8), mature_before=None):
    """Two snapshots a trading day of `per_day // 2` distinct tickers, labelled
    10 calendar days later when old enough."""
    rnd = random.Random(seed)
    names = [f"T{i:03d}" for i in range(tickers)]
    rows, scans, oid = [], [], 1
    d = end - dt.timedelta(days=days)
    mature_before = mature_before or (end - dt.timedelta(days=14))
    while d < end:
        if d.weekday() < 5:
            for slot in ("13:40:00", "19:40:00"):
                for t in rnd.sample(names, per_day // 2):
                    fired = f"{d.isoformat()}T{slot}"
                    if d < mature_before:
                        r = scored(oid, t, fired, r5=rnd.gauss(0.002, 0.05),
                                   computed=(ts(fired) + dt.timedelta(days=10)).isoformat()[:19])
                    else:
                        r = opp(oid, t, fired)
                    rows.append(r)
                    scans.append(scan(t, f"{d.isoformat()}T{slot[:2]}:35:00"))
                    oid += 1
        d += dt.timedelta(days=1)
    return rows, scans


class LifecycleTests(unittest.TestCase):
    def test_waiting_vs_due_vs_missed(self):
        row = opp(1, "AAA", "2026-10-01T16:40:00")
        t = mr.maturation_timing(row["fired_at"])
        self.assertEqual(t["entry_day"], dt.date(2026, 10, 1))
        self.assertEqual(t["window_end"], dt.date(2026, 10, 8))
        self.assertEqual(mr.maturation_stage(row, ts("2026-10-05T00:00:00"))["category"], mr.WAITING_FOR_WINDOW)
        self.assertEqual(mr.maturation_stage(row, ts("2026-10-09T20:00:00"))["category"], mr.MATURATION_DUE)
        self.assertEqual(mr.maturation_stage(row, ts("2026-10-20T00:00:00"))["category"],
                         mr.MATURATION_JOB_MISSED)

    def test_premature_empty_label_is_not_missing_price_data(self):
        # Saturday snapshot: entry Monday 9/14, window ends Monday 9/21 at the close.
        row = opp(1, "AAA", "2026-09-12T06:54:00")
        row["outcome_computed_at"] = ts("2026-09-21T12:35:00")
        self.assertEqual(mr.maturation_stage(row, NOW)["category"], mr.PREMATURE_LABEL_WRITE)
        row["outcome_computed_at"] = ts("2026-09-22T12:35:00")
        self.assertEqual(mr.maturation_stage(row, NOW)["category"], mr.MISSING_PRICE_DATA)
        self.assertEqual(mr.maturation_stage(row, NOW, price_probe={1: True})["category"],
                         mr.DATA_PROVIDER_FAILURE)

    def test_label_failures_and_eligibility(self):
        ok = scored(1, "AAA", "2026-09-01T16:40:00", computed="2026-09-12T13:00:00")
        self.assertEqual(mr.maturation_stage(ok, NOW, certified=True)["category"], mr.TRAINING_ELIGIBLE)
        self.assertEqual(mr.maturation_stage(ok, NOW, certified=False)["category"], mr.NOT_CERTIFIED)
        early = scored(2, "AAA", "2026-09-01T16:40:00", computed="2026-09-09T14:00:00")  # before 9/9 close
        self.assertEqual(mr.maturation_stage(early, NOW)["category"], mr.LABEL_FROM_INCOMPLETE_BAR)
        before = scored(3, "AAA", "2026-09-01T16:40:00", computed="2026-09-01T16:00:00")
        self.assertEqual(mr.maturation_stage(before, NOW)["category"], mr.LABEL_WRITE_FAILURE)
        partial = scored(4, "AAA", "2026-09-01T16:40:00", computed="2026-09-12T13:00:00")
        partial["return_5d"] = None
        self.assertEqual(mr.maturation_stage(partial, NOW)["category"], mr.LABEL_WRITE_FAILURE)
        bad = opp(5, "WAY-TOO-LONG", "2026-09-01T16:40:00")
        bad["outcome_computed_at"] = ts("2026-09-12T13:00:00")
        self.assertEqual(mr.maturation_stage(bad, NOW)["category"], mr.INVALID_SYMBOL)

    def test_malformed_observations(self):
        for row in (opp(1, "", "2026-09-01T16:40:00"), opp(2, "AAA", "2026-09-01T16:40:00", score=None),
                    opp(3, "AAA", "2026-09-01T16:40:00", score=140), {**opp(4, "AAA", "2026-09-01T16:40:00"),
                                                                      "fired_at": None}):
            self.assertEqual(mr.maturation_stage(row, NOW)["category"], mr.MALFORMED_OBSERVATION)


class ReportTests(unittest.TestCase):
    def test_empty_dataset(self):
        rep = mr.build_report([], [], now=NOW)
        self.assertEqual(rep["status"], mr.NOT_READY)
        self.assertEqual(rep["metrics"]["total_observations"], 0)
        self.assertIn("collection_active", " ".join(rep["blocking_reasons"]))
        view = mr.api_view(rep)
        self.assertEqual(view["observations"]["total"], 0)
        mr.render_markdown(rep)
        self.assertTrue(any(a["code"] == "COLLECTION_STOPPED" for a in mr.monitor(rep)))

    def test_partially_matured_dataset_counts(self):
        rows = [scored(1, "AAA", "2026-09-01T16:40:00", computed="2026-09-12T13:00:00"),
                scored(2, "AAA", "2026-09-01T19:40:00", computed="2026-09-12T13:00:00"),  # same signal-day
                scored(3, "BBB", "2026-09-02T16:40:00", r5=-0.03, computed="2026-09-12T13:00:00"),
                opp(4, "CCC", "2026-10-07T16:40:00"),                                     # waiting
                {**opp(5, "DDD", "2026-09-03T16:40:00"), "outcome_computed_at": ts("2026-09-08T13:00:00")}]
        rep = mr.build_report(rows, [], now=NOW)
        m = rep["metrics"]
        self.assertEqual(m["total_observations"], 5)
        self.assertEqual(m["signal_days"], 4)
        self.assertEqual(m["matured_observations_5d"], 3)
        self.assertEqual(m["matured_signal_days_5d"], 2)
        self.assertEqual(m["immature_observations"], 1)
        self.assertEqual(rep["maturation"]["by_category_observations"][mr.PREMATURE_LABEL_WRITE], 1)
        self.assertEqual(rep["horizons"]["5d"]["positive"], 1)
        self.assertEqual(rep["horizons"]["5d"]["negative"], 1)
        self.assertEqual(rep["horizons"]["5d"]["minority_class_share"], 0.5)
        for h in mr.UNCOLLECTED_HORIZONS:
            self.assertFalse(rep["horizons"][f"{h}_bar"]["collected"])
        self.assertEqual(rep["collection"]["duplicates"]["same_ticker_same_day_repeats"], 1)

    def test_chronological_coverage_and_folds_on_a_grown_dataset(self):
        rows, scans = synthetic()
        rep = mr.build_report(rows, scans, now=NOW)
        m = rep["metrics"]
        self.assertGreaterEqual(m["matured_entry_days_5d"], 30)
        self.assertEqual(m["matured_entry_weeks_5d"], len({ts(r["fired_at"].isoformat()[:19]).isocalendar()[:2]
                                                           for r in rows if r["return_5d"] is not None}))
        self.assertGreaterEqual(m["usable_walk_forward_folds_5d"], 3)
        self.assertLessEqual(m["earliest_observation"], m["latest_observation"])
        gates = {g["gate"]: g for g in rep["gates"]}
        self.assertEqual(gates["walk_forward_folds"]["status"], mr.PASS)
        # nothing freezes the served model version or AI Confidence yet -> structural fail
        self.assertEqual(gates["served_model_version"]["status"], mr.FAIL)
        self.assertEqual(rep["status"], mr.NOT_READY)
        self.assertEqual(rep["recommendation"], "ML_V4_NOT_READY")
        self.assertTrue(rep["projection"]["available"])

    def test_effective_samples_drop_overlapping_same_ticker_windows(self):
        rows = [scored(i, "AAA", f"2026-09-{d:02d}T16:40:00", computed="2026-09-30T13:00:00")
                for i, d in enumerate((1, 2, 3, 14), start=1)]
        rep = mr.build_report(rows, [], now=NOW)
        self.assertEqual(rep["metrics"]["matured_signal_days_5d"], 4)
        self.assertEqual(rep["metrics"]["effective_independent_samples_5d"], 2)

    def test_one_day_burst_cannot_pass_the_volume_gate_alone(self):
        rows = [scored(i, f"B{i:03d}", "2026-09-12T06:54:00", computed="2026-09-23T13:00:00")
                for i in range(1, 320)]
        rep = mr.build_report(rows, [], now=NOW)
        gates = {g["gate"]: g for g in rep["gates"]}
        self.assertEqual(gates["matured_signal_days"]["status"], mr.PASS)
        self.assertEqual(rep["metrics"]["matured_max_day_share_5d"], 1.0)
        self.assertEqual(gates["max_entry_day_share"]["status"], mr.FAIL)
        self.assertEqual(rep["collection"]["non_trading_day_observations"], 319)
        self.assertNotEqual(rep["status"], mr.READY)

    def test_exact_duplicates_and_late_freezes_are_counted(self):
        a = opp(1, "AAA", "2026-10-01T16:40:00")
        b = {**opp(2, "AAA", "2026-10-01T16:40:00")}
        late = {**opp(3, "BBB", "2026-10-01T16:40:00"), "created_at": ts("2026-10-03T10:00:00")}
        col = mr.build_report([a, b, late], [], now=NOW)["collection"]
        self.assertEqual(col["duplicates"]["exact_same_ticker_same_instant"], 1)
        self.assertEqual(col["late_frozen_observations"], 1)

    def test_lookahead_and_recorded_before_timestamp_are_pit_violations(self):
        row = {**opp(1, "AAA", "2026-10-01T16:40:00"), "created_at": ts("2026-10-01T10:00:00")}
        rep = mr.build_report([row], [], now=NOW)
        self.assertEqual(rep["point_in_time"]["recorded_before_timestamp"], 1)
        self.assertEqual({g["gate"]: g["status"] for g in rep["gates"]}["point_in_time_integrity"], mr.FAIL)

    def test_scan_join_rate_uses_backward_join_only(self):
        rows = [opp(1, "AAA", "2026-10-06T16:40:00"), opp(2, "BBB", "2026-10-06T16:40:00")]
        scans = [scan("AAA", "2026-10-06T16:35:00"), scan("BBB", "2026-10-06T16:45:00")]  # BBB scan is later
        fi = mr.build_report(rows, scans, now=NOW)["feature_integrity"]
        self.assertEqual(fi["recent_scan_join_rate"], 0.5)
        self.assertEqual(fi["scan_join_status"]["ONLY_LATER_SCANS"], 1)

    def test_api_view_has_no_tickers_or_row_ids(self):
        rows, scans = synthetic(days=20)
        view = mr.api_view(mr.build_report(rows, scans, now=NOW))
        blob = repr(view)
        self.assertNotIn("T00", blob)
        for k in ("status", "generated_at", "observations", "coverage", "validation", "gates", "blocking_reasons"):
            self.assertIn(k, view)
        self.assertEqual(set(view["observations"]) >= {"total", "matured", "maturity_pct"}, True)
        self.assertEqual(set(view["coverage"]) >= {"symbols", "trading_days", "date_start", "date_end"}, True)


class GateTests(unittest.TestCase):
    def _metrics(self, **over):
        base = {g.metric: (g.threshold if g.op != "<=" else g.threshold) for g in mr.GATES}
        base.update(over)
        return base

    def test_all_pass_is_ready(self):
        gates = mr.evaluate_gates(self._metrics())
        self.assertTrue(all(g["status"] == mr.PASS for g in gates))
        self.assertEqual(mr.decide_status(gates), mr.READY)

    def test_accumulating_only_failures_are_collecting_or_near_ready(self):
        near = mr.evaluate_gates(self._metrics(matured_signal_days_5d=250))
        self.assertEqual(mr.decide_status(near), mr.NEAR_READY)
        far = mr.evaluate_gates(self._metrics(matured_signal_days_5d=134))
        self.assertEqual(mr.decide_status(far), mr.COLLECTING)
        self.assertIn("matured_signal_days: 134 (needs >= 300)", mr.blocking_reasons(far))

    def test_structural_failure_is_not_ready(self):
        gates = mr.evaluate_gates(self._metrics(benchmark_coverage_5d=0.5))
        self.assertEqual(mr.decide_status(gates), mr.NOT_READY)
        self.assertIn("benchmark_coverage: 50.0% (needs >= 90.0%)", mr.blocking_reasons(gates))

    def test_missing_metric_fails_and_upper_bound_gates(self):
        gates = {g["gate"]: g for g in mr.evaluate_gates(self._metrics(recent_scan_join_rate=None,
                                                                         matured_top10_share_5d=0.6))}
        self.assertEqual(gates["scan_feature_join_rate"]["status"], mr.FAIL)
        self.assertEqual(gates["top10_symbol_share"]["status"], mr.FAIL)
        self.assertEqual(gates["top10_symbol_share"]["progress"], 0.5)

    def test_every_gate_documents_its_threshold(self):
        for g in mr.GATES:
            self.assertGreater(len(g.why), 40, g.name)
            self.assertIn(g.kind, (mr.ACCUMULATING, mr.STRUCTURAL))


class MonitorTests(unittest.TestCase):
    def test_alerts(self):
        rows = [opp(1, "AAA", "2026-09-01T16:40:00"), opp(2, "AAA", "2026-09-01T16:40:00")]
        codes = {a["code"] for a in mr.monitor(mr.build_report(rows, [], now=NOW))}
        self.assertTrue({"COLLECTION_STOPPED", "MATURATION_BEHIND", "DUPLICATE_OBSERVATIONS"} <= codes)

    def test_projection_refuses_short_history(self):
        rows = [opp(1, "AAA", "2026-10-07T16:40:00")]
        p = mr.build_report(rows, [], now=NOW)["projection"]
        self.assertFalse(p["available"])
        self.assertIn("collecting trading days", p["reason"])


class OutcomeCronGuardTests(unittest.TestCase):
    def _bars(self, start, n):
        idx = pd.bdate_range(start, periods=n)
        return pd.DataFrame({"Close": [100.0 + i for i in range(n)], "High": [101.0 + i for i in range(n)],
                             "Low": [99.0 + i for i in range(n)]}, index=idx)

    def test_open_window_stays_pending(self):
        from analytics import signal_outcomes as so

        bars = self._bars("2026-09-14", 5)  # entry 9/14 + 4 sessions: the 5th is missing
        fired = dt.datetime(2026, 9, 12, 6, 54, tzinfo=UTC)
        self.assertIsNone(so.score_signal(bars, fired))
        self.assertFalse(so.should_write_empty(bars, fired, now=ts("2026-09-21T12:35:00")))
        self.assertTrue(so.should_write_empty(None, fired, now=ts("2026-10-09T12:35:00")))
        self.assertTrue(so.should_write_empty(self._bars("2026-09-14", 6).assign(Close=0.0), fired,
                                              now=ts("2026-09-22T12:35:00")))

    def test_todays_unfinished_bar_is_dropped(self):
        from analytics import signal_outcomes as so

        bars = self._bars("2026-09-14", 6)  # last bar dated Mon 9/21
        self.assertEqual(len(so.complete_session_bars(bars, ts("2026-09-21T16:35:00"))), 5)
        self.assertEqual(len(so.complete_session_bars(bars, ts("2026-09-21T20:30:00"))), 6)

    def test_cron_skips_open_window_and_writes_complete_ones(self):
        from analytics import signal_outcomes as so

        pending = [{"id": 1, "ticker": "AAA", "fired_at": dt.datetime(2026, 9, 12, 6, 54, tzinfo=UTC)},
                   {"id": 2, "ticker": "BBB", "fired_at": dt.datetime(2026, 9, 8, 16, 40, tzinfo=UTC)}]
        bars = {"AAA": self._bars("2026-09-14", 5), "BBB": self._bars("2026-09-08", 10),
                "SPY": self._bars("2026-09-08", 10)}
        saved = []
        with mock.patch("db.signal_outcomes.list_pending_outcomes", return_value=pending), \
                mock.patch("db.signal_outcomes.save_outcome", side_effect=lambda **k: saved.append(k) or True), \
                mock.patch("db.signal_outcomes.save_benchmark", return_value=True), \
                mock.patch("data.price_alpaca.download_multi_alpaca", return_value=bars), \
                mock.patch.object(so, "complete_session_bars", side_effect=lambda b, now=None: b), \
                mock.patch.object(so, "should_write_empty",
                                  side_effect=lambda b, f, now=None: so.window_complete(b, f) if b is not None
                                  else False):
            n = so.score_pending_signal_outcomes()
        self.assertEqual(n, 1)
        self.assertEqual([s["signal_id"] for s in saved], [2])
        self.assertIsNotNone(saved[0]["return_5d"])


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class MlReadinessApiTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        from api import ml_readiness

        ml_readiness.clear_cache()
        self.addCleanup(ml_readiness.clear_cache)
        self.rows, self.scans = synthetic(days=30, end=dt.datetime.now(UTC).date())
        p = mock.patch
        p("db.research_datasets.fetch_readiness_rows", side_effect=lambda s, e, **k: list(self.rows)).start()
        p("db.research_datasets.fetch_scan_index", side_effect=lambda t, s, e, **k: list(self.scans)).start()

    def test_permissions(self):
        self.assertEqual(self.client.get("/v1/ml/readiness").status_code, 401)
        pro = self.auth(self.login("pro@example.com").json()["access_token"])
        r = self.client.get("/v1/ml/readiness", headers=pro)
        self.assertEqual(r.status_code, 403)

    def test_schema(self):
        admin = self.auth(self.login("boss@example.com").json()["access_token"])
        r = self.client.get("/v1/ml/readiness", headers=admin)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn(body["status"], (mr.NOT_READY, mr.COLLECTING, mr.NEAR_READY, mr.READY))
        self.assertIn(body["recommendation"], ("ML_V4_NOT_READY", "ML_V4_NEAR_READY", "ML_V4_READY"))
        self.assertEqual(body["observations"]["total"], len(self.rows))
        self.assertIsInstance(body["validation"]["usable_walk_forward_folds"], int)
        self.assertEqual(len(body["gates"]), len(mr.GATES))
        self.assertTrue(all({"gate", "status", "threshold", "why"} <= set(g) for g in body["gates"]))

    def test_database_outage_is_503(self):
        from db.research_datasets import ResearchDataUnavailable

        admin = self.auth(self.login("boss@example.com").json()["access_token"])
        with mock.patch("db.research_datasets.fetch_readiness_rows", side_effect=ResearchDataUnavailable("x")):
            r = self.client.get("/v1/ml/readiness", headers=admin)
        self.assertEqual(r.status_code, 503)

    def test_route_is_internal_and_read_only(self):
        from api import main

        spec = main.create_app(self.settings).openapi()
        item = spec["paths"]["/v1/ml/readiness"]
        self.assertEqual(set(item), {"get"})
        self.assertTrue(item["get"]["x-internal"])


if __name__ == "__main__":
    unittest.main()
