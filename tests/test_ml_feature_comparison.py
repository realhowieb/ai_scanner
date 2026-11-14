"""Predeclared December feature comparison: window guard, weekly folds, keep rule."""
from __future__ import annotations

import datetime as dt
import json
import random
import unittest
from pathlib import Path

from analytics import ml_feature_comparison as C
from analytics import ml_v3_audit as A
from analytics import research_dataset as rd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "docs/ml_feature_comparison_protocol_2026-12.json").read_text())
RULE = PROTOCOL["keep_rule"]


def row(i, day, feats, maturity=rd.MATURED):
    obs = dt.datetime(day.year, day.month, day.day, 15, tzinfo=dt.timezone.utc)
    return A.AuditRow(i, f"T{i}", obs, day.isoformat(), day, {h: rd.label_window_end(day, h) for h in A.HORIZONS},
                      feats, {}, {"5d": maturity}, True, None, "MATCHED", 60.0, None)


def weeks(auc_by_arm):
    """auc_by_arm: {arm: [week aucs]} -> run_arms-shaped result."""
    n = len(next(iter(auc_by_arm.values())))
    return {"weeks": [{"week": f"W{k}", "status": "ok", "auc": {a: v[k] for a, v in auc_by_arm.items()}}
                      for k in range(n)]}


class ProtocolTest(unittest.TestCase):
    def test_protocol_is_predeclared_and_before_reserved(self):
        self.assertEqual(PROTOCOL["status"], "PREDECLARED_NOT_RUN")
        self.assertLess(PROTOCOL["development_end_inclusive"], PROTOCOL["reserved_start"])
        self.assertEqual(tuple(PROTOCOL["arms"]), C.ARMS)

    def test_window_guard(self):
        C.check_window([row(1, dt.date(2026, 11, 10), {})], PROTOCOL)
        for bad in (dt.date(2026, 12, 23), dt.date(2026, 12, 15), dt.date(2026, 9, 1)):
            with self.subTest(bad), self.assertRaises(ValueError):
                C.check_window([row(1, bad, {})], PROTOCOL)


class FoldTest(unittest.TestCase):
    def test_weekly_folds_purge_and_identical_rows(self):
        rng = random.Random(1)
        data, i, d = [], 0, dt.date(2026, 10, 19)
        while d <= dt.date(2026, 12, 11):
            if d.weekday() < 5:
                for _ in range(20):
                    i += 1
                    data.append((row(i, d, {}), rng.randint(0, 1)))
            d += dt.timedelta(days=1)
        folds = C.weekly_folds(data, PROTOCOL)
        self.assertEqual(folds[0]["week"], "2026-W46")
        for f in folds:
            start = min(r.entry_day for r in f["validation"])
            self.assertTrue(all(r.window_end[5] < start for r in f["train"]))
        late = C.weekly_folds(data, PROTOCOL, train_from=dt.date(2026, 11, 2))
        self.assertEqual(late[0]["status"], "training_too_small")


class DecideTest(unittest.TestCase):
    base = [0.55, 0.52, 0.58, 0.54, 0.56]

    def test_nothing_passes_keeps_core(self):
        m = weeks({"core_v2": self.base, "core_rsi_v2": [x + 0.005 for x in self.base],
                   "core_regime_v2": self.base, "core_rsi_regime_v2": self.base})
        self.assertEqual(C.decide(m, m, RULE)["recommended_feature_set"], "core_v2")

    def test_consistent_rsi_gain_is_kept(self):
        m = weeks({"core_v2": self.base, "core_rsi_v2": [x + 0.02 for x in self.base],
                   "core_regime_v2": self.base, "core_rsi_regime_v2": [x + 0.021 for x in self.base]})
        d = C.decide(m, m, RULE)
        self.assertEqual(d["recommended_feature_set"], "core_rsi_v2")  # combo not better than RSI by 0.01

    def test_one_big_week_does_not_pass(self):
        rsi = [self.base[0] + 0.2] + [x - 0.005 for x in self.base[1:]]
        m = weeks({"core_v2": self.base, "core_rsi_v2": rsi, "core_regime_v2": self.base,
                   "core_rsi_regime_v2": self.base})
        t = C.decide(m, m, RULE)
        self.assertEqual(t["recommended_feature_set"], "core_v2")
        self.assertFalse(t["tests"]["core_rsi_v2"]["passed"])

    def test_gain_must_survive_check_run(self):
        m = weeks({"core_v2": self.base, "core_rsi_v2": [x + 0.03 for x in self.base],
                   "core_regime_v2": self.base, "core_rsi_regime_v2": self.base})
        c = weeks({"core_v2": self.base, "core_rsi_v2": [x - 0.01 for x in self.base],
                   "core_regime_v2": self.base, "core_rsi_regime_v2": self.base})
        d = C.decide(m, c, RULE)
        self.assertEqual(d["recommended_feature_set"], "core_v2")
        self.assertIn("first RSI/regime day", d["tests"]["core_rsi_v2"]["reason"])

    def test_combination_must_beat_best_single(self):
        m = weeks({"core_v2": self.base, "core_rsi_v2": [x + 0.02 for x in self.base],
                   "core_regime_v2": [x + 0.015 for x in self.base],
                   "core_rsi_regime_v2": [x + 0.04 for x in self.base]})
        self.assertEqual(C.decide(m, m, RULE)["recommended_feature_set"], "core_rsi_regime_v2")

    def test_too_few_weeks(self):
        m = weeks({a: self.base[:3] for a in C.ARMS})
        m["weeks"] = [{**w, "auc": {**w["auc"], "core_rsi_v2": w["auc"]["core_v2"] + 0.05}} for w in m["weeks"]]
        self.assertEqual(C.decide(m, m, RULE)["tests"]["core_rsi_v2"]["reason"], "too few usable weeks")


class ExactFractionTest(unittest.TestCase):
    def test_four_of_six_weeks_meets_two_thirds(self):
        base = [0.55] * 6
        rsi = [0.58, 0.58, 0.58, 0.58, 0.54, 0.54]  # better in exactly 4 of 6, mean gain 0.0167
        m = weeks({"core_v2": base, "core_rsi_v2": rsi, "core_regime_v2": base, "core_rsi_regime_v2": base})
        t = C.passes(m["weeks"], "core_rsi_v2", "core_v2", RULE)
        self.assertEqual(t["share_better"], "4/6")
        self.assertTrue(t["passed"])

    def test_three_of_five_fails(self):
        base = [0.55] * 5
        rsi = [0.60, 0.60, 0.60, 0.54, 0.54]
        m = weeks({"core_v2": base, "core_rsi_v2": rsi})
        self.assertFalse(C.passes(m["weeks"], "core_rsi_v2", "core_v2", RULE)["passed"])


class OutcomeGateTest(unittest.TestCase):
    def test_unavailable_and_invalid_excluded_with_counts(self):
        from analytics.outcome_intelligence import INVALID, UNAVAILABLE

        d = dt.date(2026, 11, 10)
        rows = [row(1, d, {}), row(2, d, {}, UNAVAILABLE), row(3, d, {}, INVALID), row(4, d, {})]
        c = C.outcome_gate(rows, 5)
        self.assertEqual((c["matured"], c["excluded_unavailable"], c["excluded_invalid"], c["pending"]), (2, 1, 1, 0))
        self.assertEqual(len(A.eligible(rows, 5, "F_clean_path_5d")), 0)  # no labels here; unavailable never enters

    def test_pending_blocks(self):
        from analytics.outcome_intelligence import PENDING

        d = dt.date(2026, 11, 10)
        for state in (PENDING, None):
            with self.subTest(state), self.assertRaises(C.OutcomesPending):
                C.outcome_gate([row(1, d, {}), row(2, d, {}, state)], 5)


class EndToEndTest(unittest.TestCase):
    def test_run_arms_with_lightgbm(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm not installed")
        from analytics import ml_v4_providers as P

        rng = random.Random(7)
        data, i, d = [], 0, dt.date(2026, 10, 19)
        while d <= dt.date(2026, 12, 11):
            if d.weekday() < 5:
                for _ in range(25):
                    i += 1
                    rsi = rng.uniform(30, 90)
                    f = {c: rng.gauss(0, 1) for c in A.AI_CONFIDENCE_FEATURE_MAP.values()}
                    f.update(core_inputs_source="scan_join", rsi_14=rsi if d >= dt.date(2026, 11, 2) else None,
                             spy_vs_sma20_pct=-0.5, spy_above_sma20=False)
                    y = int(rng.random() < (0.6 if rsi > 65 else 0.15))
                    data.append((row(i, d, f), y))
            d += dt.timedelta(days=1)
        res = C.run_arms(C.weekly_folds(data, PROTOCOL), lambda arm: P.ProviderModel("lightgbm", arm))
        ok = [w for w in res["weeks"] if w["status"] == "ok"]
        self.assertGreaterEqual(len(ok), 4)
        self.assertGreater(C.summary_auc(res)["core_rsi_v2"], C.summary_auc(res)["core_v2"])


if __name__ == "__main__":
    unittest.main()
