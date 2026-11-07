"""No production writes and no overlap of sequential PreBreakout windows."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import ml_prebreakout as ml


class PrebreakoutEvaluationSafetyTests(unittest.TestCase):
    def test_every_target_purges_the_combined_label_window(self):
        frame = pd.DataFrame({
            "Symbol": ["AAA"] * 180,
            "Timestamp": pd.date_range("2026-01-01", periods=180, freq="B", tz="UTC"),
        })
        x = pd.DataFrame({"feature": range(len(frame))})
        y = pd.Series([0, 1] * 90)
        for spec in ml.RUN17_TARGET_SPECS:
            with self.subTest(target=spec["name"]):
                gap = ml._prebreakout_purge_days(spec)
                self.assertEqual(gap, spec["lead_days"] + spec["economic_horizon"])
                folds = ml.expanding_window_folds(x, y, frame, purge_days=gap)
                self.assertEqual(len(folds), 5)
                for fold in folds:
                    start = frame.loc[fold["val_idx"][0], "Timestamp"].date()
                    for idx in fold["train_idx"]:
                        distance = np.busday_count(frame.loc[idx, "Timestamp"].date(), start)
                        self.assertGreater(distance, gap)
                # A row 6 business days before validation leaked with the old
                # five-day purge but its three-day lead + five-day path is unfinished.
                if gap == 8:
                    first = folds[0]
                    unsafe = first["val_idx"][0] - 6
                    self.assertNotIn(unsafe, first["train_idx"])

    def test_report_dispatch_uses_combined_gap_without_champion_rollback(self):
        frame = pd.DataFrame({"Symbol": ["AAA"], "Timestamp": ["2026-01-01"]})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(ml, "_load_ml_libs"), \
             patch.object(ml, "joblib", object()), \
             patch.object(ml, "XGBClassifier", object()), \
             patch.object(ml, "roc_auc_score", object()), \
             patch.object(ml, "restore_previous_model_if_active_run16_incomplete") as rollback, \
             patch.object(ml, "save_prebreakout_model") as save, \
             patch.object(ml, "load_run_history", return_value=frame), \
             patch.object(ml, "load_prebreakout_model", return_value={}), \
             patch.object(ml, "load_benchmark_regime_context", return_value={}), \
             patch.object(ml, "add_historical_ohlcv_context", return_value=frame), \
             patch.object(ml, "_download_label_bars", return_value={}), \
             patch.object(ml, "add_prebreakout_target_label", return_value=frame), \
             patch.object(ml, "add_prebreakout_target_label_variant", return_value=frame), \
             patch.object(ml, "_run17_evaluate_binary", return_value={"experiment_status": "SKIPPED"}) as evaluate:
            ml.train_prebreakout_model_run17(model_path=str(Path(tmp) / "model.pkl"))
            self.assertEqual(evaluate.call_count, len(ml.RUN17_TARGET_SPECS))
            for call, spec in zip(evaluate.call_args_list, ml.RUN17_TARGET_SPECS):
                self.assertEqual(call.kwargs["purge_days"], spec["lead_days"] + spec["economic_horizon"])
            rollback.assert_not_called()
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
