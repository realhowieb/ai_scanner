import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import ml_prebreakout


class FakePrebreakoutClassifier:
    def fit(self, _x, _y):
        return self

    def predict_proba(self, frame):
        return np.array([[0.4, 0.6] for _ in range(len(frame))])


class PrebreakoutModelPersistenceTests(unittest.TestCase):
    def setUp(self):
        # The loader caches bundles module-wide; clear so per-test mocks apply.
        from ml_prebreakout import clear_model_cache

        clear_model_cache()

    def test_load_prebreakout_model_prefers_database(self):
        db_bundle = {
            "model": object(),
            "features": ["Trend10D%"],
            "auc": 0.91,
            "trained_at": "2026-06-30T15:00:00Z",
            "source": "database",
        }
        fake_joblib = types.SimpleNamespace(load=MagicMock(return_value={"source": "local"}))

        with (
            patch.object(ml_prebreakout, "joblib", fake_joblib),
            patch.object(ml_prebreakout, "load_latest_prebreakout_model_bundle", return_value=db_bundle),
        ):
            bundle = ml_prebreakout.load_prebreakout_model()

        self.assertEqual(bundle, db_bundle)
        fake_joblib.load.assert_not_called()

    def test_load_prebreakout_model_falls_back_when_database_load_fails(self):
        local_bundle = {"model": object(), "features": ["Trend10D%"]}
        fake_joblib = types.SimpleNamespace(load=MagicMock(return_value=local_bundle))

        with (
            patch.object(ml_prebreakout, "joblib", fake_joblib),
            patch.object(ml_prebreakout, "load_latest_prebreakout_model_bundle", side_effect=RuntimeError("db down")),
        ):
            bundle = ml_prebreakout.load_prebreakout_model()

        self.assertEqual(bundle["source"], "local")
        fake_joblib.load.assert_called_once()

    def test_load_prebreakout_model_does_not_use_local_when_database_has_no_model(self):
        fake_joblib = types.SimpleNamespace(load=MagicMock(return_value={"source": "local"}))

        with (
            patch.object(ml_prebreakout, "joblib", fake_joblib),
            patch.object(ml_prebreakout, "load_latest_prebreakout_model_bundle", return_value=None),
        ):
            bundle = ml_prebreakout.load_prebreakout_model()

        self.assertIsNone(bundle)
        fake_joblib.load.assert_not_called()

    def test_train_prebreakout_model_saves_active_database_model(self):
        x = pd.DataFrame({"Trend10D%": [1.0, 2.0], "VolRel20": [1.5, 3.0]})
        y = pd.Series([0, 1])
        fake_joblib = types.SimpleNamespace(dump=MagicMock())
        fake_classifier = FakePrebreakoutClassifier()
        save_mock = MagicMock(return_value=True)
        serialize_mock = MagicMock(return_value=b"model-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(ml_prebreakout, "joblib", fake_joblib),
                patch.object(ml_prebreakout, "train_test_split", return_value=(x, x, y, y)),
                patch.object(ml_prebreakout, "roc_auc_score", return_value=0.77),
                patch.object(ml_prebreakout, "XGBClassifier", return_value=fake_classifier),
                patch.object(ml_prebreakout, "load_run_history", return_value=pd.DataFrame({"IsBreakout": [0, 1]})),
                patch.object(
                    ml_prebreakout,
                    "add_future_breakout_label",
                    return_value=pd.DataFrame({"FutureBreakout": [0, 1]}),
                ),
                patch.object(ml_prebreakout, "build_ml_dataset", return_value=(x, y)),
                patch.object(ml_prebreakout, "serialize_model_to_bytes", serialize_mock),
                patch.object(ml_prebreakout, "save_prebreakout_model", save_mock),
            ):
                bundle = ml_prebreakout.train_prebreakout_model(model_path=str(Path(tmp) / "model.pkl"))

        self.assertEqual(bundle["source"], "database")
        self.assertEqual(bundle["features"], ["Trend10D%", "VolRel20"])
        self.assertEqual(bundle["model_version"], ml_prebreakout.MODEL_VERSION)
        serialize_mock.assert_called_once_with(fake_classifier, fake_joblib)
        save_mock.assert_called_once()
        self.assertEqual(save_mock.call_args.kwargs["model_bytes"], b"model-bytes")
        self.assertEqual(save_mock.call_args.kwargs["feature_names"], ["Trend10D%", "VolRel20"])
        self.assertEqual(save_mock.call_args.kwargs["auc"], 0.77)


if __name__ == "__main__":
    unittest.main()
