import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from db import ai_confidence_models
from scan.ai_confidence import (
    CALIBRATION_ATTR,
    CONFIDENCE_COL,
    SOURCE_ATTR,
    TARGET_RULE_ATTR,
    TRAINED_AT_ATTR,
    WARNING_ATTR,
    _apply_calibration_map,
    recalibrate_active_ai_confidence,
    save_ai_confidence_model_from_files,
    score_ai_confidence,
)

_SKLEARN = importlib.util.find_spec("sklearn") is not None


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

    def _metadata_path(self, tmp: str, feature_names=None, extra=None) -> Path:
        path = Path(tmp) / "xgb_breakout_metadata.json"
        payload = {"trained_at": "2026-06-30T15:00:00Z"}
        if feature_names is not None:
            payload["feature_names"] = feature_names
        if extra:
            payload.update(extra)
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
            metadata_path = self._metadata_path(
                tmp,
                ["VolRel20", "Trend10D%"],
                {
                    "target_rule": "+4% before -2%",
                    "calibration": [{"bucket": "70-80%", "n": 3, "mean_confidence": 0.75, "hit_rate": 0.67}],
                },
            )

            fake_joblib = types.SimpleNamespace(load=lambda _path: model)
            with patch("scan.ai_confidence.joblib", fake_joblib):
                result = score_ai_confidence(frame, model_path=model_path, metadata_path=metadata_path)

        self.assertEqual(model.seen_columns, ["VolRel20", "Trend10D%"])
        self.assertEqual(list(result["Ticker"]), ["HIGH", "LOW"])
        self.assertEqual(list(result[CONFIDENCE_COL]), [90.0, 20.0])
        self.assertEqual(result.attrs[TRAINED_AT_ATTR], "2026-06-30T15:00:00Z")
        self.assertEqual(result.attrs[TARGET_RULE_ATTR], "+4% before -2%")
        self.assertEqual(result.attrs[CALIBRATION_ATTR][0]["bucket"], "70-80%")

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

    def test_score_applies_calibration_map_to_displayed_confidence(self):
        frame = pd.DataFrame(
            [
                {"Ticker": "A", "VolRel20": 1.0, "Trend10D%": 1.0},
                {"Ticker": "B", "VolRel20": 2.0, "Trend10D%": 2.0},
            ]
        )
        model = FakeConfidenceModel()  # raw proba[:,1] = [0.2, 0.9]
        # Map halves the score: 0.2 -> 0.1 (10%), 0.9 -> 0.45 (45%).
        bundle = {
            "model": model,
            "metadata": {
                "feature_names": ["VolRel20", "Trend10D%"],
                "source": "database",
                "calibration_map": {"x": [0.0, 1.0], "y": [0.0, 0.5]},
            },
        }
        with (
            patch("scan.ai_confidence.load_latest_ai_confidence_model_bundle", return_value=bundle),
            patch("scan.ai_confidence.joblib", types.SimpleNamespace(load=lambda _p: object())),
        ):
            result = score_ai_confidence(frame, model_path=Path("/does/not/exist.joblib"))
        # Sorted desc by calibrated confidence; B (45%) above A (10%).
        self.assertEqual(list(result["Ticker"]), ["B", "A"])
        self.assertEqual(list(result[CONFIDENCE_COL]), [45.0, 10.0])

    def test_apply_calibration_map_safe_noop(self):
        raw = np.array([0.1, 0.5, 0.9])
        for bad in (None, {}, {"x": [0.1], "y": [0.2]}):
            np.testing.assert_allclose(_apply_calibration_map(raw, bad), raw)

    @unittest.skipUnless(_SKLEARN, "scikit-learn not installed")
    def test_recalibrate_active_ai_confidence_patches_metadata(self):
        buckets = [
            {"n": 100, "mean_confidence": (i + 0.5) / 10, "hit_rate": (i + 0.5) / 10 * 0.5}
            for i in range(10)
        ]
        md = {"source": "database", "model_version": "ai-v1", "calibration": buckets}
        with (
            patch("scan.ai_confidence.load_ai_confidence_bundle", return_value=(object(), md, None)),
            patch("scan.ai_confidence.update_active_ai_confidence_model_metadata", return_value=True) as up,
        ):
            result = recalibrate_active_ai_confidence()
        self.assertTrue(result["ok"])
        self.assertIn("calibration_map", up.call_args[0][0])

    def test_recalibrate_active_ai_confidence_skips_and_guards(self):
        already = {"source": "database", "calibration_map": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
        with patch("scan.ai_confidence.load_ai_confidence_bundle", return_value=(object(), already, None)):
            self.assertTrue(recalibrate_active_ai_confidence()["skipped"])
        with patch("scan.ai_confidence.load_ai_confidence_bundle", return_value=(None, {}, "no model")):
            self.assertFalse(recalibrate_active_ai_confidence()["ok"])
        with patch("scan.ai_confidence.load_ai_confidence_bundle", return_value=(object(), {"source": "local"}, None)):
            self.assertFalse(recalibrate_active_ai_confidence()["ok"])

    def test_update_active_ai_confidence_metadata_merges(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (5, {"auc": 0.58})
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch.object(ai_confidence_models, "get_neon_conn", return_value=conn):
            ok = ai_confidence_models.update_active_ai_confidence_model_metadata(
                {"calibration_map": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}
            )
        self.assertTrue(ok)
        upd = [c for c in cursor.execute.call_args_list if "UPDATE ai_confidence_models SET metadata" in c[0][0]][0]
        written = json.loads(upd[0][1][0])
        self.assertIn("calibration_map", written)
        self.assertEqual(written["auc"], 0.58)
        self.assertEqual(upd[0][1][1], 5)

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
