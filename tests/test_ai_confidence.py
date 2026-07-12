import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scan.ai_confidence import (
    CONFIDENCE_COL,
    SOURCE_ATTR,
    TRAINED_AT_ATTR,
    WARNING_ATTR,
    save_ai_confidence_model_from_files,
    score_ai_confidence,
)


class FakeConfidenceModel:
    def __init__(self):
        self.seen_columns = None

    def predict_proba(self, frame):
        self.seen_columns = list(frame.columns)
        return np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        )


class AiConfidenceTests(unittest.TestCase):
    def setUp(self):
        # The loader caches bundles module-wide; clear so per-test mocks apply.
        from scan.ai_confidence import clear_bundle_cache

        clear_bundle_cache()

    def _metadata_path(self, tmp: str, feature_names=None) -> Path:
        path = Path(tmp) / "xgb_breakout_metadata.json"
        payload = {"trained_at": "2026-06-30T15:00:00Z"}
        if feature_names is not None:
            payload["feature_names"] = feature_names
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_model_returns_warning_without_scores(self):
        frame = pd.DataFrame({"Trend10D%": [1.0]})
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = self._metadata_path(tmp, ["Trend10D%"])
            fake_joblib = types.SimpleNamespace(load=lambda _path: FakeConfidenceModel())
            with patch("scan.ai_confidence.joblib", fake_joblib):
                result = score_ai_confidence(
                    frame,
                    model_path=Path(tmp) / "missing.joblib",
                    metadata_path=metadata_path,
                )

        self.assertNotIn(CONFIDENCE_COL, result.columns)
        self.assertIn("not available", result.attrs[WARNING_ATTR])
        self.assertEqual(result.attrs[TRAINED_AT_ATTR], "2026-06-30T15:00:00Z")

    def test_missing_feature_columns_returns_warning_without_scores(self):
        frame = pd.DataFrame({"Trend10D%": [1.0]})
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            model_path.write_text("placeholder", encoding="utf-8")
            metadata_path = self._metadata_path(tmp, ["Trend10D%", "VolRel20"])

            fake_joblib = types.SimpleNamespace(load=lambda _path: FakeConfidenceModel())
            with patch("scan.ai_confidence.joblib", fake_joblib):
                result = score_ai_confidence(frame, model_path=model_path, metadata_path=metadata_path)

        self.assertNotIn(CONFIDENCE_COL, result.columns)
        self.assertIn("missing feature columns", result.attrs[WARNING_ATTR])
        self.assertIn("VolRel20", result.attrs[WARNING_ATTR])

    def test_successful_scoring_uses_metadata_feature_order_and_sorts_desc(self):
        frame = pd.DataFrame(
            [
                {"Ticker": "LOW", "Trend10D%": 1.0, "VolRel20": 2.0},
                {"Ticker": "HIGH", "Trend10D%": 3.0, "VolRel20": 4.0},
            ]
        )
        model = FakeConfidenceModel()
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            model_path.write_text("placeholder", encoding="utf-8")
            metadata_path = self._metadata_path(tmp, ["VolRel20", "Trend10D%"])

            fake_joblib = types.SimpleNamespace(load=lambda _path: model)
            with patch("scan.ai_confidence.joblib", fake_joblib):
                result = score_ai_confidence(frame, model_path=model_path, metadata_path=metadata_path)

        self.assertEqual(model.seen_columns, ["VolRel20", "Trend10D%"])
        self.assertEqual(list(result["Ticker"]), ["HIGH", "LOW"])
        self.assertEqual(list(result[CONFIDENCE_COL]), [90.0, 20.0])
        self.assertEqual(result.attrs[TRAINED_AT_ATTR], "2026-06-30T15:00:00Z")

    def test_successful_scoring_prefers_database_model(self):
        frame = pd.DataFrame(
            [
                {"Ticker": "LOW", "Trend10D%": 1.0, "VolRel20": 2.0},
                {"Ticker": "HIGH", "Trend10D%": 3.0, "VolRel20": 4.0},
            ]
        )
        model = FakeConfidenceModel()
        bundle = {
            "model": model,
            "metadata": {
                "feature_names": ["VolRel20", "Trend10D%"],
                "trained_at": "2026-06-30T15:00:00Z",
                "source": "database",
            },
        }
        with (
            patch("scan.ai_confidence.load_latest_ai_confidence_model_bundle", return_value=bundle),
            patch("scan.ai_confidence.joblib", types.SimpleNamespace(load=lambda _path: object())),
        ):
            result = score_ai_confidence(frame, model_path=Path("/does/not/exist.joblib"))

        self.assertEqual(model.seen_columns, ["VolRel20", "Trend10D%"])
        self.assertEqual(list(result["Ticker"]), ["HIGH", "LOW"])
        self.assertEqual(result.attrs[SOURCE_ATTR], "database")

    def test_save_ai_confidence_model_from_files_persists_metadata(self):
        model = FakeConfidenceModel()
        save_mock = types.SimpleNamespace(calls=[])

        def fake_save(**kwargs):
            save_mock.calls.append(kwargs)
            return True

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            model_path.write_text("placeholder", encoding="utf-8")
            metadata_path = self._metadata_path(tmp, ["VolRel20", "Trend10D%"])

            fake_joblib = types.SimpleNamespace(
                load=lambda _path: model,
                dump=lambda _model, buffer: buffer.write(b"model-bytes"),
            )
            with (
                patch("scan.ai_confidence.joblib", fake_joblib),
                patch("scan.ai_confidence.save_ai_confidence_model", fake_save),
            ):
                saved = save_ai_confidence_model_from_files(
                    model_path=model_path,
                    metadata_path=metadata_path,
                )

        self.assertTrue(saved)
        self.assertEqual(len(save_mock.calls), 1)
        self.assertEqual(save_mock.calls[0]["model_bytes"], b"model-bytes")
        self.assertEqual(save_mock.calls[0]["feature_names"], ["VolRel20", "Trend10D%"])
        self.assertEqual(save_mock.calls[0]["trained_at"], "2026-06-30T15:00:00Z")


if __name__ == "__main__":
    unittest.main()
