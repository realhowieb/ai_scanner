import json
import unittest

import pandas as pd

from analytics.prediction_provenance import COLUMN, attach, customer_frame, digest, models_from_row
from scripts.verify_prediction_provenance import inventory


class ProvenanceVerificationTests(unittest.TestCase):
    def test_exact_inputs_and_calibration_survive_pandas_json(self):
        value = 0.03425020499116203
        mapping = {"x": [value, 1.0], "y": [0.13126323218066338, 0.8]}
        frame = pd.DataFrame({"Ticker": ["ABC"], "Last": [100.0]})
        attach(frame, "prebreakout", pd.DataFrame({"f": [value]}), pd.DataFrame({"f": [False]}), [value], [mapping["y"][0]], {"calibration_map": mapping})
        models = models_from_row(json.loads(frame.to_json(orient="records"))[0])
        p = models["prebreakout"]
        self.assertEqual(p["feature_values"], [value])
        self.assertEqual(p["raw_probability"], value)
        self.assertEqual(p["calibration_snapshot"], mapping)
        self.assertEqual(p["calibration_hash"], digest(mapping))
        self.assertEqual(inventory([models])["prebreakout"]["invalid"]["calibration_hash"], 0)

    def test_audit_detects_pre_fix_rounding_defect(self):
        mapping = {"x": [0.03425020499116203], "y": [0.13126323218066338]}
        legacy = {"prebreakout": {"calibration_snapshot": mapping, "calibration_hash": digest(mapping)}}
        roundtrip = json.loads(pd.DataFrame({COLUMN: [legacy]}).to_json(orient="records"))[0][COLUMN]
        self.assertEqual(inventory([roundtrip])["prebreakout"]["invalid"]["calibration_hash"], 1)

    def test_customer_columns_do_not_mutate_storage_or_ranking(self):
        frame = pd.DataFrame({"Ticker": ["HIGH", "LOW"], "PreBreakoutProb%": [20, 10], COLUMN: ["private", "private"], "SourceScanId": ["scan", "scan"]})
        public = customer_frame(frame)
        self.assertNotIn(COLUMN, public.columns)
        self.assertNotIn("SourceScanId", public.to_csv(index=False))
        self.assertIn(COLUMN, frame.columns)
        self.assertEqual(public["Ticker"].tolist(), frame["Ticker"].tolist())
        self.assertEqual(public["PreBreakoutProb%"].tolist(), frame["PreBreakoutProb%"].tolist())

    def test_unavailable_and_absent_are_not_predictions(self):
        report = inventory([{}, {"ai_confidence": {"status": "unavailable", "reason": "missing_model"}}])
        self.assertEqual(report["ai_confidence"]["counts"], {"absent": 1, "unavailable": 1})

    def test_audit_sql_is_read_only_and_bounded(self):
        from pathlib import Path

        source = (Path(__file__).parents[1] / "scripts/verify_prediction_provenance.py").read_text()
        self.assertIn("READ ONLY", source)
        self.assertIn("LIMIT 51", source)
        self.assertIn("LIMIT 5001", source)
        self.assertNotIn("ensure_schema", source)
