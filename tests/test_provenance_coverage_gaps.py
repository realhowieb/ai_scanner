import copy
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from analytics.prediction_provenance import COLUMN, models_by_ticker
from scripts.verify_prediction_provenance import reconcile


def prediction(time="2026-10-09T17:10:00Z"):
    return {"status": "captured", "source_scan_id": "scan-1", "inferred_at": time,
            "raw_probability": 0.2, "calibrated_probability": 0.3}


class CoverageGapTests(unittest.TestCase):
    def test_run_snapshot_are_one_prediction_event(self):
        payload = json.dumps([{"Ticker": "ABC", COLUMN: json.dumps({"prebreakout": prediction()})}])
        runs = [{"id": i, "created_at": "now", "is_snapshot": i == 2, "payload": payload} for i in (1, 2)]
        observations = [{"context": "scheduled:us_market", "cohort": cohort, "models": models}
                        for cohort, models in (("CANDIDATE", {"prebreakout": prediction()}), ("NEAR_MISS", {}), ("CONTROL", {}))]
        report = reconcile(runs, observations, [])
        self.assertEqual(report["unique_linkable_inferences"], 1)
        self.assertEqual(report["conflicting_duplicate_identities"], 0)
        self.assertEqual(report["cohort_coverage"]["scheduled:us_market:NEAR_MISS"]["prebreakout_expected_not_invoked"], 1)
        self.assertEqual(report["cohort_coverage"]["scheduled:us_market:CANDIDATE"]["prebreakout_present"], 1)

    def test_ambiguous_duplicates_are_not_fabricated(self):
        rows = [{"Ticker": "ABC", COLUMN: {"prebreakout": prediction(time)}} for time in ("old", "new")]
        result = models_by_ticker(rows)["ABC"]["prebreakout"]
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("raw_probability", result)

    def test_actual_freeze_path_preserves_non_pick_evidence(self):
        from analytics.opportunity_freeze import freeze_latest_opportunities
        from ui.opportunities import build_opportunities

        snapshot = datetime(2026, 10, 9, 17, tzinfo=timezone.utc)
        models = {"prebreakout": prediction()}
        data = {"snapshot_time": snapshot, "top_setups": [("ABC", 9)], "picks": [],
                "source_models": {"ABC": models}}
        baseline = build_opportunities({k: v for k, v in data.items() if k != "source_models"})
        after = build_opportunities(data)
        self.assertEqual([{k: v for k, v in o.items() if k != "models"} for o in baseline],
                         [{k: v for k, v in o.items() if k != "models"} for o in after])
        with patch("ui.market_brief._compute_brief", return_value=data), patch("db.signal_outcomes.freeze_signal", return_value=True) as save:
            self.assertEqual(freeze_latest_opportunities(), 1)
        self.assertEqual(save.call_args.kwargs["raw_signal"]["models"], models)
        self.assertIsNone(save.call_args.kwargs["prebreakout_prob"])
        self.assertEqual(save.call_args.kwargs["fired_at"], snapshot)

    def test_recalculation_replaces_only_actual_role(self):
        from ui.opportunities import build_opportunities

        old = {"prebreakout": prediction(), "ai_confidence": prediction()}
        new = prediction("2026-10-09T18:00:00Z")
        data = {"source_models": {"ABC": old}, "picks": [{"symbol": "ABC", "prob": 30, "models": {"prebreakout": new}}]}
        saved = copy.deepcopy(data)
        result = build_opportunities(data)[0]["models"]
        self.assertEqual(result["prebreakout"], new)
        self.assertEqual(result["ai_confidence"], old["ai_confidence"])
        self.assertEqual(data, saved)

    def test_scanner_consolidation_preserves_evidence_not_scores(self):
        from ui.results_intelligence import consolidate_scanner_results

        rows = [{"Ticker": "ABC", "BreakoutScore": 9, COLUMN: {"prebreakout": prediction()}}]
        before = consolidate_scanner_results([{k: v for k, v in rows[0].items() if k != COLUMN}])
        after = consolidate_scanner_results(rows)
        self.assertEqual(after[0]["models"], rows[0][COLUMN])
        self.assertEqual(before[0]["score"], after[0]["score"])
        self.assertEqual(before[0]["prob"], after[0]["prob"])

    def test_public_brief_strips_models_for_all_tiers(self):
        from api.market import brief

        core = {"data": {"picks": [{"symbol": "ABC", "prob": 30, "models": {"prebreakout": prediction()}}]},
                "compared": [{"ticker": "ABC", "models": {"prebreakout": prediction()}}],
                "phase": "market", "has_previous": False}
        for allowed in (False, True):
            with patch("api.market._brief_core", return_value=core):
                public = brief({"can_early_breakout": allowed})
            self.assertNotIn('"models"', json.dumps(public))
        self.assertIn("models", core["compared"][0])

    def test_legacy_and_malformed_rows_remain_safe(self):
        self.assertEqual(models_by_ticker([None, {}, {"Ticker": "ABC"}]), {})
        frame = pd.DataFrame({"Ticker": ["ABC"], COLUMN: [json.dumps({"prebreakout": prediction()})]})
        self.assertEqual(models_by_ticker(frame.to_dict("records"))["ABC"]["prebreakout"], prediction())

    def test_frozen_first_write_keeps_original_evidence(self):
        from unittest.mock import MagicMock

        from db import signal_outcomes as store

        conn, cursor = MagicMock(), MagicMock()
        conn.cursor.return_value = cursor
        persisted = {}

        def execute(sql, params):
            self.assertIn("ON CONFLICT (source, source_event_id, ticker, signal_type) DO NOTHING", sql)
            key = (params[0], params[1], params[5], params[2])
            cursor.rowcount = int(key not in persisted)
            persisted.setdefault(key, json.loads(params[-1]))

        cursor.execute.side_effect = execute
        snapshot = datetime(2026, 10, 9, 17, tzinfo=timezone.utc)
        with patch.object(store, "get_neon_conn", return_value=conn), patch.object(store, "_ensure_schema"):
            store.freeze_opportunity(snapshot, {"ticker": "ABC", "models": {"prebreakout": prediction("old")}})
            store.freeze_opportunity(snapshot, {"ticker": "ABC", "models": {"prebreakout": prediction("new")}})
        self.assertEqual(len(persisted), 1)
        self.assertEqual(next(iter(persisted.values()))["models"]["prebreakout"]["inferred_at"], "old")

    def test_brief_assembly_carries_all_existing_sources(self):
        from ui.market_brief import _compute_brief

        frame = pd.DataFrame({"Ticker": ["ABC"], "PctChange": [1.0], COLUMN: [json.dumps({"prebreakout": prediction()})]})
        with patch("ui.market_brief._brief_scan_df", return_value=frame), \
             patch("ui.market_brief._sector_leaders", return_value=[]), \
             patch("ui.market_brief._snapshot_time", return_value=None), \
             patch("scheduler.morning_digest._market_gappers", return_value=[]), \
             patch("scheduler.morning_digest._earnings_days_map", return_value={}), \
             patch("scheduler.morning_digest._earnings_today", return_value=[]), \
             patch("scheduler.morning_digest._todays_setups", return_value=([], [("ABC", 9)])), \
             patch("scheduler.morning_digest._prebreakout_picks", return_value=[]), \
             patch("scheduler.evening_wrap._market_close_context", return_value=[]), \
             patch("scheduler.evening_wrap._day_movers", return_value=([], [])):
            result = _compute_brief()
        self.assertEqual(result["source_models"]["ABC"]["prebreakout"], prediction())
        self.assertEqual(result["picks"], [])
