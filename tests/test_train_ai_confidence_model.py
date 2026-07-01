import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from scripts import train_ai_confidence_model as trainer


class FakeXGBClassifier:
    def fit(self, _x, _y):
        return self

    def predict_proba(self, frame):
        return np.array([[0.7, 0.3] for _ in range(len(frame))])


class TrainAiConfidenceModelTests(unittest.TestCase):
    def test_build_dataset_uses_expected_feature_order_and_binary_labels(self):
        frame = pd.DataFrame(
            [
                {"Trend10D%": "1.5", "Trend20D%": 2, "VolRel20": 3, "IsBreakout": "true"},
                {"Trend10D%": None, "DollarVol20": 1000, "BreakoutScore": 4, "GapPct": 5, "IsBreakout": "False"},
            ]
        )

        x, y = trainer.build_ai_confidence_dataset(frame)

        self.assertEqual(list(x.columns), trainer.FEATURE_NAMES)
        self.assertEqual(list(y), [1, 0])
        self.assertEqual(float(x.loc[0, "Trend10D%"]), 1.5)
        self.assertEqual(float(x.loc[1, "Trend20D%"]), 0.0)

    def test_train_writes_artifacts_and_uploads_when_requested(self):
        history = pd.DataFrame(
            {
                "Trend10D%": [1.0, 2.0, 3.0, 4.0],
                "Trend20D%": [1.0, 2.0, 3.0, 4.0],
                "VolRel20": [1.0, 2.0, 3.0, 4.0],
                "DollarVol20": [100.0, 200.0, 300.0, 400.0],
                "BreakoutScore": [2.0, 3.0, 4.0, 5.0],
                "GapPct": [0.1, 0.2, 0.3, 0.4],
                "IsBreakout": [0, 1, 0, 1],
            }
        )
        fake_joblib = types.SimpleNamespace(dump=MagicMock(side_effect=lambda _model, path: Path(path).write_bytes(b"model")))
        upload_mock = MagicMock(return_value=True)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "xgb_breakout_model.joblib"
            metadata_path = Path(tmp) / "xgb_breakout_metadata.json"
            with (
                patch.object(trainer, "joblib", fake_joblib),
                patch.object(trainer, "load_run_history", return_value=history),
                patch.object(trainer, "train_test_split", return_value=(history[trainer.FEATURE_NAMES], history[trainer.FEATURE_NAMES], history["IsBreakout"], history["IsBreakout"])),
                patch.object(trainer, "roc_auc_score", return_value=0.82),
                patch.object(trainer, "XGBClassifier", return_value=FakeXGBClassifier()),
                patch.object(trainer, "save_ai_confidence_model_from_files", upload_mock),
            ):
                summary = trainer.train_ai_confidence_model(
                    model_path=model_path,
                    metadata_path=metadata_path,
                    upload_db=True,
                )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertTrue(summary["saved_to_db"])
        self.assertEqual(metadata["feature_names"], trainer.FEATURE_NAMES)
        self.assertEqual(metadata["auc"], 0.82)
        self.assertEqual(metadata["positive_rows"], 2)
        self.assertEqual(metadata["model_version"], trainer.MODEL_VERSION)
        upload_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
