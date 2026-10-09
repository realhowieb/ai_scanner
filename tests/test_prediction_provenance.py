import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analytics.prediction_provenance import COLUMN, artifact_identity, attach, digest, models_from_row


class PredictionProvenanceTests(unittest.TestCase):
    def test_saved_observation_and_freeze_preserve_models(self):
        from datetime import datetime, timezone

        from analytics.observation_capture import build_scan_observations
        from db.signal_outcomes import freeze_opportunity

        models = {"prebreakout": {"inferred_at": "2026-10-09T14:00:00Z", "raw_probability": 0.2}}
        frame = pd.DataFrame({"Ticker": ["ABC"], COLUMN: [models]})
        rows = json.loads(frame.to_json(orient="records"))
        observations = build_scan_observations(rows, universe="US_MARKET", scan_timestamp="2026-10-09T13:00:00Z", session="morning")
        self.assertEqual(observations[0]["models"], models)
        with patch("db.signal_outcomes.freeze_signal", return_value=True) as freeze:
            freeze_opportunity(datetime(2026, 10, 9, 13, tzinfo=timezone.utc), {"ticker": "ABC", "models": models})
        self.assertEqual(freeze.call_args.kwargs["raw_signal"]["models"], models)
        self.assertEqual(freeze.call_args.kwargs["fired_at"].hour, 13)

    def test_local_loaded_bytes_match_hash(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from analytics.prediction_provenance import load_local

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            path.write_bytes(b"actual-loaded-bytes")
            model, identity = load_local(SimpleNamespace(load=lambda stream: stream.read()), path)
        self.assertEqual(model, b"actual-loaded-bytes")
        self.assertEqual(identity["sha256"], hashlib.sha256(model).hexdigest())

    def test_strict_serialization_order_units_and_role_merge(self):
        frame = pd.DataFrame({"Ticker": ["ABC"], "Timestamp": ["2026-10-09T12:00:00Z"], "SourceScanId": ["scan-1"]})
        inputs = pd.DataFrame({"second": [0.0], "first": [2.0]})
        mask = pd.DataFrame({"second": [True], "first": [False]})
        calibration = {"x": [0.0, 1.0], "y": [0.1, 0.8]}
        metadata = {"loaded_artifact": artifact_identity(b"served", registry_id=13, version="actual"), "calibration_map": calibration}
        for role in ("prebreakout", "ai_confidence"):
            attach(frame, role, inputs, mask, [0.2], [0.24], metadata)
        payload = json.loads(frame.to_json(orient="records"))[0]
        models = models_from_row(payload)
        self.assertEqual(set(models), {"prebreakout", "ai_confidence"})
        p = models["prebreakout"]
        self.assertEqual(p["feature_names"], ["second", "first"])
        self.assertEqual(p["feature_values"], [0, 2])
        self.assertEqual(p["default_mask"], [True, False])
        self.assertEqual(p["calibration_hash"], digest(p["calibration_snapshot"]))
        self.assertEqual(p["raw_probability"], 0.2)
        self.assertEqual(p["source_scan_id"], "scan-1")
        self.assertEqual(p["loaded_artifact"]["version"], "actual")
        json.dumps(models, allow_nan=False)

    def test_legacy_and_missing_identity_are_explicit(self):
        self.assertEqual(models_from_row({}), {})
        self.assertEqual(models_from_row({COLUMN: "invalid"}), {})
        frame = pd.DataFrame({"Ticker": ["ABC"]})
        attach(frame, "prebreakout", pd.DataFrame({"f": [float("nan")]}), pd.DataFrame({"f": [True]}), [0.1], [0.1], {})
        p = frame.iloc[0][COLUMN]["prebreakout"]
        self.assertIsNone(p["loaded_artifact"])
        self.assertIsNone(p["input_timestamp"])
        self.assertEqual(p["feature_values"], [None])
        self.assertEqual(p["target_evidence"]["status"], "unavailable")

    def test_prebreakout_captures_existing_prediction_once(self):
        from unittest.mock import MagicMock

        import ml_prebreakout as ml
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2]])
        frame = pd.DataFrame({"Ticker": ["ABC"]})
        with patch.object(ml, "load_prebreakout_model", return_value={"model": model, "features": ["a", "b"]}), patch.object(ml, "_live_feature_frame", return_value=pd.DataFrame({"a": [np.nan]})):
            result = ml.score_prebreakout(frame)
        model.predict_proba.assert_called_once()
        p = result.iloc[0][COLUMN]["prebreakout"]
        self.assertEqual(p["feature_values"], [0.0, 0.0])
        self.assertEqual(p["default_mask"], [True, True])
        self.assertEqual(result.iloc[0]["PreBreakoutProb%"], 20)

    def test_ai_capture_follows_sorted_ticker(self):
        from unittest.mock import MagicMock

        from scan.ai_confidence import score_ai_confidence
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        with patch("scan.ai_confidence.load_ai_confidence_bundle", return_value=(model, {"feature_names": ["f"]}, None)):
            result = score_ai_confidence(pd.DataFrame({"Ticker": ["LOW", "HIGH"], "f": [1, 2]}))
        model.predict_proba.assert_called_once()
        self.assertEqual(result.iloc[0]["Ticker"], "HIGH")
        self.assertEqual(result.iloc[0][COLUMN]["ai_confidence"]["feature_values"], [2])
        self.assertEqual(result.iloc[0][COLUMN]["ai_confidence"]["raw_probability"], 0.9)
