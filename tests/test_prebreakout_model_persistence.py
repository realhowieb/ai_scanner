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
        x = pd.DataFrame({"Trend10D%": [1.0, 2.0, 3.0, 4.0], "VolRel20": [1.5, 3.0, 2.0, 4.0]})
        y = pd.Series([0, 1, 0, 1])
        labeled = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
                "FutureQualitySetupHit": y,
            }
        )
        fake_joblib = types.SimpleNamespace(dump=MagicMock())
        fake_classifier = FakePrebreakoutClassifier()
        save_mock = MagicMock(return_value=True)
        serialize_mock = MagicMock(return_value=b"model-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(ml_prebreakout, "joblib", fake_joblib),
                patch.object(ml_prebreakout, "roc_auc_score", return_value=0.77),
                patch.object(ml_prebreakout, "XGBClassifier", return_value=fake_classifier),
                patch.object(ml_prebreakout, "load_run_history", return_value=pd.DataFrame({"Symbol": ["A", "B"]})),
                patch.object(
                    ml_prebreakout,
                    "add_prebreakout_target_label",
                    return_value=labeled,
                ),
                patch.object(ml_prebreakout, "build_ml_dataset", return_value=(x, y)),
                patch.object(ml_prebreakout, "walk_forward_split", return_value=(x.iloc[:2], x.iloc[2:], y.iloc[:2], y.iloc[2:])),
                patch.object(ml_prebreakout, "serialize_model_to_bytes", serialize_mock),
                patch.object(ml_prebreakout, "save_prebreakout_model", save_mock),
            ):
                bundle = ml_prebreakout.train_prebreakout_model(model_path=str(Path(tmp) / "model.pkl"))

        self.assertEqual(bundle["source"], "database")
        self.assertEqual(bundle["features"], ["Trend10D%", "VolRel20"])
        self.assertEqual(bundle["model_version"], ml_prebreakout.MODEL_VERSION)
        self.assertEqual(bundle["validation_method"], "walk_forward")
        self.assertEqual(bundle["target"], "FutureQualitySetupHit")
        self.assertEqual(bundle["lead_days"], 3)
        self.assertEqual(bundle["setup_score_threshold"], 8.0)
        self.assertIn("calibration", bundle)
        self.assertEqual(bundle["rows"], 4)
        self.assertEqual(bundle["positive_rows"], 2)
        self.assertEqual(bundle["validation_rows"], 2)
        serialize_mock.assert_called_once_with(fake_classifier, fake_joblib)
        save_mock.assert_called_once()
        self.assertEqual(save_mock.call_args.kwargs["model_bytes"], b"model-bytes")
        self.assertEqual(save_mock.call_args.kwargs["feature_names"], ["Trend10D%", "VolRel20"])
        self.assertEqual(save_mock.call_args.kwargs["auc"], 0.77)

    def test_add_forward_return_labels_uses_existing_return_column(self):
        df = pd.DataFrame(
            {
                "Symbol": ["AAA", "BBB"],
                "Timestamp": pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC"),
                "Return_5D": [0.05, -0.01],
            }
        )

        labeled = ml_prebreakout.add_forward_return_labels(df)

        self.assertEqual(list(labeled["ForwardReturnHit"]), [1, 0])
        self.assertEqual(list(labeled["Return_5D"]), [0.05, -0.01])

    def test_add_prebreakout_target_requires_future_quality_setup_and_outcome(self):
        df = pd.DataFrame(
            [
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-01T15:00:00Z"),
                    "IsBreakout": False,
                    "BreakoutScore": 5.0,
                    "Last": 90.0,
                    "High20": 100.0,
                    "Return_5D": 0.0,
                },
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-02T15:00:00Z"),
                    "IsBreakout": True,
                    "BreakoutScore": 9.0,
                    "Last": 101.0,
                    "High20": 101.0,
                    "Return_5D": 0.05,
                },
                {
                    "Symbol": "BBB",
                    "Timestamp": pd.Timestamp("2026-01-01T15:00:00Z"),
                    "IsBreakout": False,
                    "BreakoutScore": 5.0,
                    "Last": 90.0,
                    "High20": 100.0,
                    "Return_5D": 0.0,
                },
                {
                    "Symbol": "BBB",
                    "Timestamp": pd.Timestamp("2026-01-02T15:00:00Z"),
                    "IsBreakout": True,
                    "BreakoutScore": 9.0,
                    "Last": 101.0,
                    "High20": 101.0,
                    "Return_5D": -0.01,
                },
            ]
        )

        labeled = ml_prebreakout.add_prebreakout_target_label(df)

        by_symbol = {row["Symbol"]: row["FutureQualitySetupHit"] for _, row in labeled.iterrows()}
        self.assertEqual(by_symbol, {"AAA": 1, "BBB": 0})

    def test_walk_forward_split_validates_on_later_rows(self):
        x = pd.DataFrame({"feature": [10, 20, 30, 40, 50]})
        y = pd.Series([0, 1, 0, 1, 1])
        labeled = pd.DataFrame({"Timestamp": pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")})

        x_train, x_val, y_train, y_val = ml_prebreakout.walk_forward_split(x, y, labeled, validation_fraction=0.4)

        self.assertEqual(list(x_train["feature"]), [10, 20, 30])
        self.assertEqual(list(x_val["feature"]), [40, 50])
        self.assertEqual(list(y_train), [0, 1, 0])
        self.assertEqual(list(y_val), [1, 1])

    def test_confidence_bucket_diagnostics_reports_hit_rate(self):
        rows = ml_prebreakout.confidence_bucket_diagnostics(
            pd.Series([0, 1, 1]),
            pd.Series([0.25, 0.72, 0.78]),
        )

        by_bucket = {row["bucket"]: row for row in rows}
        self.assertEqual(by_bucket["20-30%"]["n"], 1)
        self.assertEqual(by_bucket["70-80%"]["n"], 2)
        self.assertEqual(by_bucket["70-80%"]["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
