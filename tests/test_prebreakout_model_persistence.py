import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import ml_prebreakout
from db import prebreakout_models

# Isotonic calibration needs scikit-learn; the lean CI env may not have it.
_SKLEARN = importlib.util.find_spec("sklearn") is not None


class FakePrebreakoutClassifier:
    def fit(self, _x, _y):
        return self

    def predict_proba(self, frame):
        return np.array([[0.4, 0.6] for _ in range(len(frame))])


class ImportancePrebreakoutClassifier(FakePrebreakoutClassifier):
    def fit(self, x, _y):
        self.feature_importances_ = np.linspace(0.0, 1.0, num=len(x.columns))
        return self


class FakePrebreakoutRanker:
    def __init__(self, *args, **kwargs):
        self.params = dict(kwargs)

    def fit(self, _x, _y, group=None):
        self.group = list(group or [])
        return self

    def predict(self, frame):
        return np.arange(len(frame), dtype=float)


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
        x = pd.DataFrame(
            {
                "Trend10D%": [1.0, 2.0, 3.0, 4.0],
                "VolRel20": [1.5, 3.0, 2.0, 4.0],
                "SPYTrend10D": [0.5, 0.6, 0.7, 0.8],
                "RSvsSPY10D": [0.5, 1.4, 2.3, 3.2],
                "RSvsSPY20D": [0.4, 1.3, 2.2, 3.1],
                "RSvsQQQ10D": [0.3, 1.2, 2.1, 3.0],
                "RSvsQQQ20D": [0.2, 1.1, 2.0, 2.9],
                "RVOLChange1D": [0.0, 1.5, -1.0, 2.0],
                "RVOLChange3D": [0.0, 0.0, 0.0, 2.5],
                "RVOLSlope3D": [0.0, 0.0, 0.0, 0.8],
                "RVOLSlope5D": [0.0, 0.0, 0.0, 0.5],
            }
        )
        y = pd.Series([0, 1, 0, 1])
        labeled = x.copy()
        labeled["Timestamp"] = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
        labeled["FutureQualitySetupHit"] = y
        fake_joblib = types.SimpleNamespace(dump=MagicMock())
        fake_classifier = FakePrebreakoutClassifier()
        save_mock = MagicMock(return_value=True)
        serialize_mock = MagicMock(return_value=b"model-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(ml_prebreakout, "joblib", fake_joblib),
                patch.object(ml_prebreakout, "roc_auc_score", return_value=0.77),
                patch.object(ml_prebreakout, "average_precision_score", return_value=0.61),
                patch.object(ml_prebreakout, "brier_score_loss", return_value=0.18),
                patch.object(ml_prebreakout, "log_loss", return_value=0.54),
                patch.object(ml_prebreakout, "XGBClassifier", return_value=fake_classifier),
                patch.object(ml_prebreakout, "load_run_history", return_value=pd.DataFrame({"Symbol": ["A", "B"]})),
                patch.object(
                    ml_prebreakout,
                    "add_prebreakout_target_label",
                    return_value=labeled,
                ),
                patch.object(ml_prebreakout, "build_ml_dataset", return_value=(x, y)),
                patch.object(ml_prebreakout, "validate_run6_reproduction_audit", return_value=[]),
                patch.object(ml_prebreakout, "validate_run11_dataset_audit", return_value=[]),
                patch.object(ml_prebreakout, "_run10_promotion_decision", return_value=("PROMOTE", [])),
                patch.object(
                    ml_prebreakout,
                    "expanding_window_folds",
                    return_value=[
                        {
                            "fold": 1,
                            "train_idx": [0, 1],
                            "val_idx": [2, 3],
                            "validation_start": "2026-01-03T00:00:00Z",
                            "validation_end": "2026-01-04T00:00:00Z",
                            "purge_days": 5,
                        }
                    ],
                ),
                patch.object(ml_prebreakout, "serialize_model_to_bytes", serialize_mock),
                patch.object(ml_prebreakout, "save_prebreakout_model", save_mock),
            ):
                bundle = ml_prebreakout.train_prebreakout_model(model_path=str(Path(tmp) / "model.pkl"))

        self.assertEqual(bundle["source"], "database")
        expected_features = [
            "Trend10D%",
            "VolRel20",
            "RSvsSPY10D",
            "RSvsSPY20D",
            "RSvsQQQ10D",
            "RSvsQQQ20D",
            "RVOLChange1D",
            "RVOLChange3D",
            "RVOLSlope3D",
            "RVOLSlope5D",
        ]
        expected_market_features = ["RSvsSPY10D", "RSvsSPY20D", "RSvsQQQ10D", "RSvsQQQ20D"]
        self.assertEqual(bundle["features"], expected_features)
        self.assertEqual(bundle["market_feature_names"], expected_market_features)
        self.assertEqual(bundle["model_version"], ml_prebreakout.MODEL_VERSION)
        self.assertEqual(bundle["validation_method"], "expanding_window_5fold_purged")
        self.assertEqual(bundle["market_regime_result"], "PROMOTE")
        self.assertEqual(bundle["target"], "FutureQualitySetupHit")
        self.assertEqual(bundle["lead_days"], 3)
        self.assertEqual(bundle["setup_score_threshold"], 8.0)
        self.assertIn("calibration", bundle)
        self.assertIn("validation_folds", bundle)
        self.assertIn("validation_summary", bundle)
        self.assertEqual(bundle["validation_summary"]["auc_mean"], 0.77)
        self.assertEqual(bundle["validation_summary"]["pr_auc_mean"], 0.61)
        self.assertEqual(bundle["rows"], 4)
        self.assertEqual(bundle["positive_rows"], 2)
        self.assertEqual(bundle["validation_rows"], 2)
        serialize_mock.assert_called_once_with(fake_classifier, fake_joblib)
        save_mock.assert_called_once()
        self.assertEqual(save_mock.call_args.kwargs["model_bytes"], b"model-bytes")
        self.assertEqual(save_mock.call_args.kwargs["feature_names"], expected_features)
        self.assertEqual(save_mock.call_args.kwargs["auc"], 0.77)
        self.assertEqual(save_mock.call_args.kwargs["metadata"]["target"], "FutureQualitySetupHit")
        self.assertEqual(save_mock.call_args.kwargs["metadata"]["market_feature_names"], expected_market_features)

    def test_run6_reproduction_audit_rejects_target_distribution_drift(self):
        audit = {
            "eligible_rows": 21445,
            "positive_rows": 547,
            "negative_rows": 20898,
            "positive_rate": 547 / 21445,
            "valid_fold_count": 4,
            "validation_rows": 14297,
        }

        failures = ml_prebreakout.validate_run6_reproduction_audit(audit)

        self.assertTrue(any("positive rows" in failure for failure in failures))
        self.assertTrue(any("valid fold count" in failure for failure in failures))
        self.assertTrue(any("validation rows" in failure for failure in failures))

    def test_run9_dataset_audit_requires_exact_controlled_counts(self):
        audit = {
            "eligible_rows": 21445,
            "positive_rows": 3381,
            "validation_rows": 7150,
        }

        failures = ml_prebreakout.validate_run9_dataset_audit(audit)

        self.assertEqual(failures, ["validation_rows 7150 != expected 7149"])

    def test_run11_dataset_audit_allows_small_live_history_drift(self):
        audit = {
            "eligible_rows": 21694,
            "positive_rows": 3392,
            "validation_rows": 7232,
        }

        failures = ml_prebreakout.validate_run11_dataset_audit(audit)

        self.assertEqual(failures, [])

    def test_run11_dataset_audit_blocks_material_live_history_drift(self):
        audit = {
            "eligible_rows": 23000,
            "positive_rows": 3700,
            "validation_rows": 7600,
        }

        failures = ml_prebreakout.validate_run11_dataset_audit(audit)

        self.assertGreaterEqual(len(failures), 3)

    def test_market_regime_merge_preserves_index_for_existing_masks(self):
        df = pd.DataFrame(
            [
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-03T15:00:00Z"),
                    "FutureQualitySetupHit": 1,
                    "BreakoutScore": 5.0,
                    "Last": 104.0,
                },
                {
                    "Symbol": "SPY",
                    "Timestamp": pd.Timestamp("2026-01-01T21:00:00Z"),
                    "Trend10D%": 1.0,
                    "Trend20D%": 2.0,
                    "EMA9": 100.0,
                    "EMA21": 99.0,
                    "ATR14": 2.0,
                    "Last": 100.0,
                },
                {
                    "Symbol": "BBB",
                    "Timestamp": pd.Timestamp("2026-01-02T15:00:00Z"),
                    "FutureQualitySetupHit": 0,
                    "BreakoutScore": 4.0,
                    "Last": 50.0,
                },
            ]
        )
        candidate_mask = pd.Series([True, False, True], index=df.index)

        featured = ml_prebreakout.add_market_regime_features(df)
        selected = featured.loc[candidate_mask]

        self.assertEqual(list(featured.index), list(df.index))
        self.assertEqual(list(featured["Symbol"]), ["AAA", "SPY", "BBB"])
        self.assertEqual(list(selected["Symbol"]), ["AAA", "BBB"])
        self.assertEqual(list(selected["FutureQualitySetupHit"]), [1, 0])

    def test_setup_evolution_features_are_chronological_and_restore_alignment(self):
        df = pd.DataFrame(
            [
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-03T15:00:00Z"),
                    "Last": 104.0,
                    "High20": 110.0,
                    "EMA9": 101.0,
                    "EMA21": 100.0,
                    "VolRel20": 2.0,
                    "ATR14": 4.0,
                },
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-01T15:00:00Z"),
                    "Last": 100.0,
                    "High20": 110.0,
                    "EMA9": 98.0,
                    "EMA21": 100.0,
                    "VolRel20": 1.0,
                    "ATR14": 3.0,
                },
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-02T15:00:00Z"),
                    "Last": 102.0,
                    "High20": 110.0,
                    "EMA9": 99.0,
                    "EMA21": 100.0,
                    "VolRel20": 1.5,
                    "ATR14": 3.5,
                },
                {
                    "Symbol": "BBB",
                    "Timestamp": pd.Timestamp("2026-01-03T15:00:00Z"),
                    "Last": 50.0,
                    "High20": 55.0,
                    "EMA9": 48.0,
                    "EMA21": 49.0,
                    "VolRel20": 0.8,
                    "ATR14": 1.0,
                },
            ],
            index=[10, 11, 12, 13],
        )

        featured = ml_prebreakout.add_prebreakout_features(df, include_market_features=False)

        self.assertEqual(list(featured.index), [10, 11, 12, 13])
        self.assertAlmostEqual(featured.loc[10, "EMA9_21Spread"], 1.0)
        self.assertAlmostEqual(featured.loc[10, "EMA9_21SpreadChange1D"], 2.0)
        self.assertAlmostEqual(featured.loc[10, "DistanceTo20DHighChange1D"], (104.0 - 110.0) / 110.0 * 100.0 - ((102.0 - 110.0) / 110.0 * 100.0))
        self.assertAlmostEqual(featured.loc[10, "RVOLChange1D"], 0.5)
        self.assertAlmostEqual(featured.loc[10, "ATRPercent"], 4.0 / 104.0 * 100.0)
        self.assertTrue(pd.isna(featured.loc[11, "EMA9_21SpreadChange1D"]))
        self.assertTrue(pd.isna(featured.loc[13, "EMA9_21SpreadChange1D"]))

    def test_compression_features_use_current_and_past_rows_only(self):
        rows = []
        for day in range(25):
            close = 100.0 + float(day)
            rows.append(
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-01T15:00:00Z") + pd.Timedelta(day, unit="D"),
                    "Close": close,
                    "High": close + 2.0,
                    "Low": close - 2.0,
                    "High20": close + 4.0,
                    "BreakoutPos20D": day / 25.0,
                }
            )
        df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

        featured = ml_prebreakout.add_prebreakout_features(df, include_market_features=False)
        last_idx = int(featured.index[0])
        chronological = df.sort_values(["Symbol", "Timestamp"], kind="mergesort")
        close = chronological["Close"]
        sma20 = close.rolling(20, min_periods=20).mean().iloc[-1]
        std20 = close.rolling(20, min_periods=20).std().iloc[-1]
        expected_bb_width = (4.0 * std20) / sma20 * 100.0

        self.assertAlmostEqual(featured.loc[last_idx, "BBWidth20Pct"], expected_bb_width)
        self.assertGreater(float(featured.loc[last_idx, "ATR14Pct"]), 0.0)
        self.assertGreater(float(featured.loc[last_idx, "ATRRatio5D"]), 0.0)
        self.assertGreater(float(featured.loc[last_idx, "RangeCompression5D"]), 0.0)
        self.assertIn("ATRCompression5D", featured.columns)

    def test_historical_ohlcv_context_uses_prior_completed_bar(self):
        scans = pd.DataFrame(
            {
                "Symbol": ["AAA", "AAA"],
                "Timestamp": [
                    pd.Timestamp("2026-01-02T15:00:00Z"),
                    pd.Timestamp("2026-01-03T15:00:00Z"),
                ],
                "Last": [100.0, 101.0],
            }
        )
        bars = pd.DataFrame(
            {
                "Open": [90.0, 100.0, 200.0],
                "High": [91.0, 101.0, 201.0],
                "Low": [89.0, 99.0, 199.0],
                "Close": [90.5, 100.5, 200.5],
                "Volume": [1000, 1100, 9999],
            },
            index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"], utc=True),
        )

        with patch("data.price_alpaca.download_multi_alpaca", return_value={"AAA": bars}) as download_mock:
            enriched = ml_prebreakout.add_historical_ohlcv_context(scans, days_back=90)

        download_mock.assert_called_once()
        self.assertEqual(float(enriched.loc[0, "High"]), 91.0)
        self.assertEqual(float(enriched.loc[1, "High"]), 101.0)
        self.assertNotEqual(float(enriched.loc[1, "High"]), 201.0)
        self.assertEqual(list(enriched.index), [0, 1])

    def test_structure_features_higher_lows_and_resistance_touches(self):
        rows = []
        for day in range(10):
            rows.append(
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-02-01T15:00:00Z") + pd.Timedelta(day, unit="D"),
                    "Close": 95.0 + day,
                    "High": 99.4 if day >= 5 else 96.0 + day,
                    "Low": 90.0 + day,
                    "High20": 100.0,
                    "BreakoutPos20D": day / 10.0,
                }
            )
        df = pd.DataFrame(rows)

        featured = ml_prebreakout.add_prebreakout_features(df, include_market_features=False)

        self.assertAlmostEqual(featured.loc[9, "DistanceToHigh20Pct"], 100.0 - 104.0)
        self.assertAlmostEqual(featured.loc[9, "BreakoutPosSlope5D"], 0.1)
        self.assertEqual(float(featured.loc[9, "HigherLowCount5D"]), 5.0)
        self.assertAlmostEqual(featured.loc[9, "HigherLowRatio5D"], 1.0)
        self.assertEqual(float(featured.loc[9, "ResistanceTouchCount10D"]), 7.0)

    def test_run11_price_structure_features_are_chronological_and_aligned(self):
        rows = []
        for day in range(55):
            close = 100.0 + day * 0.2
            rows.append(
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-03-01T15:00:00Z") + pd.Timedelta(day, unit="D"),
                    "Close": close,
                    "High": close + 1.0,
                    "Low": 90.0 + day * 0.15,
                    "Volume": 2000.0 - day * 5.0,
                }
            )
        df = pd.DataFrame(rows).sample(frac=1.0, random_state=42)

        featured = ml_prebreakout.add_prebreakout_features(df, include_market_features=False)
        ordered = df.sort_values(["Symbol", "Timestamp"], kind="mergesort")
        last_original_index = ordered.index[-1]
        high20 = ordered["High"].rolling(20, min_periods=3).max().iloc[-1]
        high50 = ordered["High"].rolling(50, min_periods=10).max().iloc[-1]

        self.assertEqual(list(featured.index), list(df.index))
        self.assertAlmostEqual(featured.loc[last_original_index, "High20D"], high20)
        self.assertAlmostEqual(featured.loc[last_original_index, "High50D"], high50)
        self.assertAlmostEqual(featured.loc[last_original_index, "DistanceTo20DHighPct"], ordered["Close"].iloc[-1] / high20 - 1.0)
        self.assertGreaterEqual(float(featured.loc[last_original_index, "HigherLowCount20D"]), 19.0)
        self.assertIn("ResistanceTouches20D", featured.columns)
        self.assertIn("VolumeDryUp20D", featured.columns)

    def test_run12_compression_quality_features_are_chronological_and_aligned(self):
        rows = []
        for day in range(30):
            width = 12.0 - day * 0.25
            rows.append(
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-04-01T15:00:00Z") + pd.Timedelta(day, unit="D"),
                    "Close": 100.0 + day * 0.05,
                    "High": 100.0 + width / 2.0,
                    "Low": 100.0 - width / 2.0 + day * 0.08,
                    "Volume": 5000.0 - day * 70.0,
                    "Trend10D%": 4.0,
                    "Trend20D%": 7.0,
                    "RSvsSPY10D": 1.5,
                    "RSvsSPY20D": 2.0,
                }
            )
        df = pd.DataFrame(rows).sample(frac=1.0, random_state=7)

        featured = ml_prebreakout.add_prebreakout_features(df, include_market_features=False)
        ordered = df.sort_values(["Symbol", "Timestamp"], kind="mergesort")
        last_original_index = ordered.index[-1]

        self.assertEqual(list(featured.index), list(df.index))
        for col in [
            "RangeContractionRatio3v10",
            "ConsecutiveContractingRangeDays",
            "RangePctStd10D",
            "HighLowChannelWidth20D",
            "ChannelCompression5v20",
            "VolumeDryUp10D",
            "CompressionWithVolumeDryUp",
            "Compression_x_RS10",
        ]:
            self.assertIn(col, featured.columns)
            self.assertFalse(pd.isna(featured.loc[last_original_index, col]))
        self.assertGreater(float(featured.loc[last_original_index, "ConsecutiveContractingRangeDays"]), 1.0)
        self.assertLess(float(featured.loc[last_original_index, "RangeContractionRatio3v10"]), 1.0)
        self.assertLess(float(featured.loc[last_original_index, "VolumeDryUp10D"]), 1.0)

    def test_run12_inside_day_and_higher_low_quality_features(self):
        rows = []
        for day in range(12):
            rows.append(
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-05-01T15:00:00Z") + pd.Timedelta(day, unit="D"),
                    "Close": 100.0,
                    "High": 110.0 - day * 0.4,
                    "Low": 90.0 + day * 0.3,
                    "Volume": 1000.0,
                }
            )
        df = pd.DataFrame(rows)

        featured = ml_prebreakout.add_prebreakout_features(df, include_market_features=False)

        self.assertEqual(float(featured.loc[11, "InsideDay"]), 1.0)
        self.assertEqual(float(featured.loc[11, "InsideDayCount5D"]), 5.0)
        self.assertAlmostEqual(float(featured.loc[11, "HigherLowRatio10D"]), 1.0)
        self.assertGreater(float(featured.loc[11, "LowSlopeAcceleration"]), -1.0)
        self.assertAlmostEqual(float(featured.loc[11, "LowConsistency10D"]), 1.0)

    def test_run16_structural_interaction_features_are_time_safe(self):
        rows = []
        for day in range(25):
            rows.append(
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-06-01T15:00:00Z") + pd.Timedelta(day, unit="D"),
                    "Close": 100.0 + day * 0.1,
                    "High": 104.0 - day * 0.03,
                    "Low": 95.0 + day * 0.2,
                    "RSvsSPY10D": 2.0,
                    "RSvsQQQ10D": 1.5,
                }
            )
        df = pd.DataFrame(rows).sample(frac=1.0, random_state=11)

        featured = ml_prebreakout.add_prebreakout_features(df, include_market_features=False)
        last_idx = df.sort_values(["Symbol", "Timestamp"], kind="mergesort").index[-1]

        for col in [
            "HigherLowRatio10D_x_RangeCompression10D",
            "LowConsistency10D_x_RangeCompression10D",
            "LowSlope10DPct_x_RangeCompression20D",
            "LowSlopeAcceleration_x_RangeCompression10D",
            "HigherLowRatio10D_x_RangeCompression20D",
        ]:
            self.assertIn(col, featured.columns)
            self.assertFalse(pd.isna(featured.loc[last_idx, col]))

    def test_run10_rejects_empty_feature_experiment(self):
        x = pd.DataFrame({"Trend10D%": [1.0, 2.0]})
        y = pd.Series([0, 1])
        labeled = pd.DataFrame({"Timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC")})

        evaluation = ml_prebreakout._run10_eval(
            "Missing family",
            x,
            y,
            labeled,
            [],
            ["Trend10D%"],
            ["MissingFeature"],
            "missing_family",
        )

        self.assertEqual(evaluation["experiment_status"], "INVALID_EMPTY_FEATURES")
        self.assertEqual(evaluation["features_added"], [])

    def test_run12_rejects_features_already_in_control(self):
        x = pd.DataFrame({"Trend10D%": [1.0, 2.0], "RangePct": [0.1, 0.2]})
        y = pd.Series([0, 1])
        labeled = pd.DataFrame({"Timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC")})

        evaluation = ml_prebreakout._run10_eval(
            "Already controlled",
            x,
            y,
            labeled,
            [],
            ["Trend10D%", "RangePct"],
            ["RangePct"],
            "duplicate_family",
        )

        self.assertEqual(evaluation["experiment_status"], "INVALID_EMPTY_FEATURES")
        self.assertEqual(evaluation["features_added"], [])
        self.assertIn("already present", evaluation["skip_reason"])

    def test_run14_train_fold_clipping_thresholds_do_not_use_validation_outlier(self):
        x = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 1000.0]})
        y = pd.Series([0, 1, 0, 1])
        labeled = pd.DataFrame({"Timestamp": pd.date_range("2026-01-01", periods=4, tz="UTC")})
        folds = [
            {
                "fold": 1,
                "train_idx": [0, 1],
                "val_idx": [2, 3],
                "validation_start": "2026-01-03T00:00:00Z",
                "validation_end": "2026-01-04T00:00:00Z",
                "purge_days": 5,
            }
        ]

        with (
            patch.object(ml_prebreakout, "roc_auc_score", return_value=0.7),
            patch.object(ml_prebreakout, "average_precision_score", return_value=0.4),
            patch.object(ml_prebreakout, "brier_score_loss", return_value=0.2),
            patch.object(ml_prebreakout, "log_loss", return_value=0.5),
            patch.object(ml_prebreakout, "XGBClassifier", return_value=FakePrebreakoutClassifier()),
        ):
            result = ml_prebreakout.evaluate_prebreakout_feature_set(
                x,
                y,
                labeled,
                folds,
                ["feature"],
                transform={"clip": True, "lower_q": 0.01, "upper_q": 0.99},
            )

        threshold = result["fold_metrics"][0]["transform_notes"]["clip_thresholds"]["feature"]["upper"]
        self.assertLess(threshold, 1000.0)

    def test_run14_preprocessing_plan_adds_missing_flags_and_removes_features(self):
        x = pd.DataFrame({"keep": [1.0, np.nan, 3.0], "drop": [0.0, 0.0, 0.0]})
        plan = {
            "clip_thresholds": {"keep": {"lower": 1.0, "upper": 2.0}},
            "missing_indicators": ["keep"],
            "features_removed": ["drop"],
        }

        transformed, features = ml_prebreakout._apply_preprocessing_plan(x, ["keep", "drop"], plan)

        self.assertEqual(features, ["keep", "keepMissing"])
        self.assertEqual(float(transformed.loc[2, "keep"]), 2.0)
        self.assertEqual(float(transformed.loc[1, "keepMissing"]), 1.0)

    def test_run16_prune_to_count_uses_training_fold_importance(self):
        x = pd.DataFrame({f"f{i}": [i, i + 1, i + 2, 999.0] for i in range(6)})
        y = pd.Series([0, 1, 0, 1])
        labeled = pd.DataFrame({"Timestamp": pd.date_range("2026-01-01", periods=4, tz="UTC")})
        folds = [
            {
                "fold": 1,
                "train_idx": [0, 1],
                "val_idx": [2, 3],
                "validation_start": "2026-01-03T00:00:00Z",
                "validation_end": "2026-01-04T00:00:00Z",
                "purge_days": 5,
            }
        ]

        with (
            patch.object(ml_prebreakout, "roc_auc_score", return_value=0.7),
            patch.object(ml_prebreakout, "average_precision_score", return_value=0.4),
            patch.object(ml_prebreakout, "brier_score_loss", return_value=0.2),
            patch.object(ml_prebreakout, "log_loss", return_value=0.5),
            patch.object(ml_prebreakout, "XGBClassifier", return_value=ImportancePrebreakoutClassifier()),
        ):
            result = ml_prebreakout.evaluate_prebreakout_feature_set(
                x,
                y,
                labeled,
                folds,
                list(x.columns),
                transform={"prune_to_count": 3},
            )

        notes = result["fold_metrics"][0]["transform_notes"]
        self.assertEqual(result["fold_metrics"][0]["feature_count"], 3)
        self.assertEqual(notes["features_removed"], ["f0", "f1", "f2"])

    def test_run16_feature_stability_report_ranks_consistent_importance(self):
        x = pd.DataFrame({f"f{i}": np.arange(10) + i for i in range(4)})
        y = pd.Series([0, 1] * 5)
        folds = [
            {"fold": 1, "train_idx": [0, 1, 2, 3], "val_idx": [4, 5], "validation_start": "", "validation_end": "", "purge_days": 5},
            {"fold": 2, "train_idx": [0, 1, 2, 3, 4, 5], "val_idx": [6, 7], "validation_start": "", "validation_end": "", "purge_days": 5},
        ]

        with patch.object(ml_prebreakout, "XGBClassifier", return_value=ImportancePrebreakoutClassifier()):
            report = ml_prebreakout.feature_importance_stability_report(x, y, folds, list(x.columns))

        self.assertEqual(report[0]["feature"], "f3")
        self.assertEqual(report[0]["fold_presence"], 2)

    def test_run16_promotion_requires_all_five_valid_folds(self):
        champion = {
            "validation_summary": {
                "auc_mean": 0.65,
                "min_fold_auc": 0.63,
                "lift_over_baseline_mean": 1.7,
            }
        }
        candidate = {
            "experiment_status": "VALID",
            "validation_summary": {
                "auc_mean": 0.66,
                "min_fold_auc": 0.64,
                "lift_over_baseline_mean": 1.7,
            },
            "fold_metrics": [{"fold": 4, "auc": 0.64}, {"fold": 5, "auc": 0.68}],
        }

        result, reasons = ml_prebreakout._run10_promotion_decision(candidate, champion, [])

        self.assertEqual(result, "NO_PROMOTION")
        self.assertTrue(any("valid fold count 2 != expected 5" in reason for reason in reasons))

    def test_run16_restore_previous_model_if_active_run16_incomplete(self):
        active = (
            10,
            "prebreakout-xgb-v16",
            {"run_number": 16, "run16_result": "PROMOTED", "run16_dataset_audit": {"valid_fold_count": 2}},
            0.657,
            "2026-09-10T17:59:29Z",
        )
        previous = (9, "prebreakout-xgb-v14", 0.652, "2026-09-10T16:00:00Z")
        cursor = MagicMock()
        cursor.fetchone.side_effect = [active, previous]
        conn = MagicMock()
        conn.cursor.return_value = cursor

        with patch.object(prebreakout_models, "get_neon_conn", return_value=conn):
            result = prebreakout_models.restore_previous_model_if_active_run16_incomplete(min_valid_folds=5)

        self.assertTrue(result["restored"])
        self.assertEqual(result["deactivated_id"], 10)
        self.assertEqual(result["restored_id"], 9)
        cursor.execute.assert_any_call("UPDATE prebreakout_models SET is_active = TRUE WHERE id = %s", (9,))

    def test_run17_forward_path_stop_wins_same_bar(self):
        bars = pd.DataFrame(
            {
                "Close": [100.0, 100.0],
                "High": [100.0, 106.0],
                "Low": [100.0, 97.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True),
        )

        stats = ml_prebreakout._forward_path_stats(bars, pd.Timestamp("2026-01-02").date(), 1, 0.04, -0.02)

        self.assertFalse(stats["hit"])
        self.assertTrue(stats["stopped"])
        self.assertEqual(stats["stopped_day"], 1)

    def test_run17_forward_label_variant_respects_trading_horizon(self):
        df = pd.DataFrame(
            {
                "Symbol": ["AAA"],
                "Timestamp": [pd.Timestamp("2026-01-02T15:00:00Z")],
                "Last": [100.0],
                "High20": [110.0],
                "BreakoutScore": [5.0],
                "IsBreakout": [False],
            }
        )
        bars = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 104.5],
                "High": [100.0, 101.0, 104.5],
                "Low": [100.0, 99.0, 101.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"], utc=True),
        )

        one_day = ml_prebreakout.add_forward_return_labels_variant(
            df,
            horizon_days=1,
            hit_threshold=0.04,
            stop_threshold=-0.02,
            bars_by_symbol={"AAA": bars},
            force_path=True,
        )
        two_day = ml_prebreakout.add_forward_return_labels_variant(
            df,
            horizon_days=2,
            hit_threshold=0.04,
            stop_threshold=-0.02,
            bars_by_symbol={"AAA": bars},
            force_path=True,
        )

        self.assertEqual(int(one_day.loc[0, "ForwardReturnHit"]), 0)
        self.assertEqual(int(two_day.loc[0, "ForwardReturnHit"]), 1)

    def test_run17_prebreakout_target_variant_preserves_candidate_rule(self):
        df = pd.DataFrame(
            {
                "Symbol": ["AAA", "AAA"],
                "Timestamp": pd.to_datetime(["2026-01-02T15:00:00Z", "2026-01-05T15:00:00Z"], utc=True),
                "Last": [100.0, 105.0],
                "High20": [110.0, 106.0],
                "BreakoutScore": [5.0, 8.0],
                "IsBreakout": [False, False],
            }
        )
        bars = pd.DataFrame(
            {
                "Close": [100.0, 105.0, 110.0],
                "High": [100.0, 105.0, 110.0],
                "Low": [100.0, 104.0, 108.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"], utc=True),
        )

        labeled = ml_prebreakout.add_prebreakout_target_label_variant(
            df,
            lead_days=1,
            horizon_days=1,
            hit_threshold=0.04,
            stop_threshold=-0.02,
            bars_by_symbol={"AAA": bars},
            force_path=True,
        )

        self.assertEqual(len(labeled), 1)
        self.assertEqual(int(labeled.loc[0, "FutureQualitySetupHit"]), 1)
        self.assertLess(float(labeled.loc[0, "BreakoutScore"]), ml_prebreakout.PREBREAKOUT_SETUP_SCORE_THRESHOLD)

    def test_run17_session_groups_are_timestamp_local(self):
        df = pd.DataFrame(
            {
                "Symbol": ["B", "A", "C"],
                "Timestamp": pd.to_datetime(
                    ["2026-01-03T15:00:00Z", "2026-01-02T15:00:00Z", "2026-01-03T14:00:00Z"],
                    utc=True,
                ),
            },
            index=[20, 10, 30],
        )

        ordered, groups = ml_prebreakout._session_order_and_groups(df, [20, 10, 30])

        self.assertEqual(ordered, [10, 30, 20])
        self.assertEqual(groups, [1, 2])

    def test_run17_ranking_objective_uses_session_groups(self):
        x = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}, index=[0, 1, 2, 3])
        y = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 3])
        labeled = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(
                    ["2026-01-01T15:00:00Z", "2026-01-01T15:01:00Z", "2026-01-02T15:00:00Z", "2026-01-02T15:01:00Z"],
                    utc=True,
                )
            },
            index=[0, 1, 2, 3],
        )
        folds = [{"fold": 1, "train_idx": [0, 1], "val_idx": [2, 3], "validation_start": "", "validation_end": "", "purge_days": 5}]

        with (
            patch.object(ml_prebreakout, "XGBRanker", FakePrebreakoutRanker),
            patch.object(ml_prebreakout, "roc_auc_score", return_value=0.7),
            patch.object(ml_prebreakout, "average_precision_score", return_value=0.4),
            patch.object(ml_prebreakout, "brier_score_loss", return_value=0.2),
            patch.object(ml_prebreakout, "log_loss", return_value=0.5),
        ):
            result = ml_prebreakout.evaluate_prebreakout_ranking_feature_set(x, y, labeled, folds, ["f"])

        self.assertEqual(result["fold_metrics"][0]["ranking_group_count"], 1)
        self.assertEqual(result["validation_index"], [2, 3])

    def test_run17_topk_economic_diagnostics_reports_forward_returns(self):
        frame = pd.DataFrame({"Return_5D": [0.01, 0.08, -0.02, 0.03], "MFE_5D": [0.02, 0.1, 0.01, 0.04], "MAE_5D": [-0.01, -0.02, -0.04, -0.01]})

        result = ml_prebreakout.topk_economic_diagnostics(
            frame,
            [0, 1, 0, 1],
            [0.1, 0.9, 0.2, 0.8],
            return_column="Return_5D",
            mfe_column="MFE_5D",
            mae_column="MAE_5D",
        )

        self.assertEqual(result["top_10pct"]["n"], 1)
        self.assertEqual(result["top_10pct"]["hit_rate"], 1.0)
        self.assertAlmostEqual(result["top_10pct"]["avg_return"], 0.08)

    def test_run17_calibration_comparison_keeps_ranking_separate(self):
        with (
            patch.object(ml_prebreakout, "brier_score_loss", return_value=0.12),
        ):
            result = ml_prebreakout.calibration_comparison([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.9])

        self.assertIn("raw", result)
        self.assertIn("buckets", result["raw"])
        self.assertIn("platt", result)

    def test_run17_binary_eval_sets_explicit_status(self):
        df_labeled = pd.DataFrame(
            {
                "Symbol": ["AAA", "BBB", "CCC", "DDD"],
                "Timestamp": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"],
                    utc=True,
                ),
                "Return_5D": [0.01, 0.05, -0.02, 0.04],
            },
            index=[10, 11, 12, 13],
        )
        x = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]}, index=df_labeled.index)
        y = pd.Series([0, 1, 0, 1], index=df_labeled.index)
        fake_eval = {
            "fold_metrics": [
                {"fold": 1, "validation_rows": 2, "auc": 0.7},
                {"fold": 2, "validation_rows": 2, "auc": None},
            ],
            "validation_summary": {"auc_mean": 0.7},
            "validation_actual": [0, 1],
            "validation_proba": [0.2, 0.8],
            "validation_index": [12, 13],
        }

        with (
            patch.object(ml_prebreakout, "build_ml_dataset", return_value=(x, y)),
            patch.object(ml_prebreakout, "expanding_window_folds", return_value=[{"fold": 1}]),
            patch.object(ml_prebreakout, "_run9_eval", return_value=fake_eval),
        ):
            result = ml_prebreakout._run17_evaluate_binary(
                "A - Current Production Target",
                df_labeled,
                ["f"],
                None,
                purge_days=5,
                return_column="Return_5D",
                mfe_column=None,
                mae_column=None,
                family="target_matrix",
            )

        self.assertEqual(result["experiment_status"], "VALID")
        self.assertEqual(result["valid_fold_count"], 1)
        self.assertIn("validation_warning", result)

    @unittest.skipUnless(_SKLEARN, "scikit-learn not installed")
    def test_isotonic_calibration_map_corrects_and_preserves_ranking(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.0, 1.0, 2000)
        # Over-confident model: true hit prob is half the raw score.
        y = (rng.uniform(0.0, 1.0, 2000) < raw * 0.5).astype(int)

        cmap = ml_prebreakout.fit_isotonic_calibration_map(y, raw)
        self.assertIsInstance(cmap, dict)
        self.assertEqual(cmap["method"], "isotonic")
        self.assertEqual(len(cmap["x"]), len(cmap["y"]))

        calibrated = ml_prebreakout.apply_calibration_map(raw, cmap)
        # Magnitude corrected toward the true base rate...
        self.assertAlmostEqual(float(calibrated.mean()), float(y.mean()), places=2)
        self.assertGreater(raw.mean(), calibrated.mean())
        # ...and monotonic, so ranking is never reordered.
        order_raw = np.argsort(raw, kind="stable")
        self.assertTrue(np.all(np.diff(calibrated[order_raw]) >= -1e-9))

    @unittest.skipUnless(_SKLEARN, "scikit-learn not installed")
    def test_fit_calibration_map_from_buckets_corrects_overconfidence(self):
        buckets = [
            {"bucket": f"{i}", "n": 100, "mean_confidence": (i + 0.5) / 10, "hit_rate": (i + 0.5) / 10 * 0.5}
            for i in range(10)
        ]
        cmap = ml_prebreakout.fit_isotonic_calibration_map_from_buckets(buckets)
        self.assertEqual(cmap["method"], "isotonic_from_buckets")
        self.assertEqual(cmap["n"], 1000)
        calibrated = ml_prebreakout.apply_calibration_map(np.array([0.9]), cmap)
        self.assertLess(float(calibrated[0]), 0.9)

    def test_fit_calibration_map_from_buckets_none_when_sparse(self):
        self.assertIsNone(ml_prebreakout.fit_isotonic_calibration_map_from_buckets([]))
        self.assertIsNone(
            ml_prebreakout.fit_isotonic_calibration_map_from_buckets(
                [{"n": 1, "mean_confidence": 0.5, "hit_rate": 0.3}]
            )
        )

    @unittest.skipUnless(_SKLEARN, "scikit-learn not installed")
    def test_recalibrate_active_champion_patches_metadata(self):
        buckets = [
            {"bucket": f"{i}", "n": 100, "mean_confidence": (i + 0.5) / 10, "hit_rate": (i + 0.5) / 10 * 0.5}
            for i in range(10)
        ]
        bundle = {"source": "database", "model_version": "v9", "auc": 0.668, "calibration": buckets}
        with (
            patch.object(ml_prebreakout, "load_prebreakout_model", return_value=bundle),
            patch.object(ml_prebreakout, "update_active_prebreakout_model_metadata", return_value=True) as up,
            patch.object(ml_prebreakout, "clear_model_cache"),
        ):
            result = ml_prebreakout.recalibrate_active_champion()
        self.assertTrue(result["ok"])
        self.assertEqual(result["model_version"], "v9")
        self.assertIn("calibration_map", up.call_args[0][0])

    def test_recalibrate_active_champion_skips_when_already_calibrated(self):
        bundle = {"source": "database", "calibration_map": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
        with patch.object(ml_prebreakout, "load_prebreakout_model", return_value=bundle):
            result = ml_prebreakout.recalibrate_active_champion()
        self.assertTrue(result["ok"])
        self.assertTrue(result["skipped"])

    def test_recalibrate_active_champion_requires_active_db_model(self):
        with patch.object(ml_prebreakout, "load_prebreakout_model", return_value=None):
            self.assertFalse(ml_prebreakout.recalibrate_active_champion()["ok"])
        with patch.object(ml_prebreakout, "load_prebreakout_model", return_value={"source": "local"}):
            self.assertFalse(ml_prebreakout.recalibrate_active_champion()["ok"])

    def test_update_active_model_metadata_merges_patch(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = (7, {"auc": 0.668, "calibration": []})
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch.object(prebreakout_models, "get_neon_conn", return_value=conn):
            ok = prebreakout_models.update_active_prebreakout_model_metadata(
                {"calibration_map": {"x": [0.0, 1.0], "y": [0.0, 0.5]}}
            )
        self.assertTrue(ok)
        update_call = [c for c in cursor.execute.call_args_list if "UPDATE prebreakout_models SET metadata" in c[0][0]][0]
        written = json.loads(update_call[0][1][0])
        self.assertIn("calibration_map", written)
        self.assertEqual(written["auc"], 0.668)  # existing keys preserved
        self.assertEqual(update_call[0][1][1], 7)  # active id

    def test_update_active_model_metadata_false_without_active_row(self):
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        conn = MagicMock()
        conn.cursor.return_value = cursor
        with patch.object(prebreakout_models, "get_neon_conn", return_value=conn):
            self.assertFalse(
                prebreakout_models.update_active_prebreakout_model_metadata({"calibration_map": {"x": [0], "y": [0]}})
            )

    def test_apply_calibration_map_is_safe_noop_without_map(self):
        raw = np.array([0.1, 0.5, 0.9])
        for bad in (None, {}, {"x": [0.1], "y": [0.2]}, {"x": [], "y": []}):
            np.testing.assert_allclose(ml_prebreakout.apply_calibration_map(raw, bad), raw)

    def test_fit_isotonic_calibration_map_returns_none_when_insufficient(self):
        self.assertIsNone(ml_prebreakout.fit_isotonic_calibration_map([0, 1], [0.2, 0.8]))
        self.assertIsNone(
            ml_prebreakout.fit_isotonic_calibration_map([0] * 100, [0.3] * 100)
        )

    def test_run17_target_change_candidate_decision_is_separate(self):
        baseline = {
            "name": "A",
            "valid_fold_count": 5,
            "validation_summary": {"auc_mean": 0.66, "lift_over_baseline_mean": 1.7},
        }
        challenger = {
            "name": "B",
            "valid_fold_count": 5,
            "validation_summary": {"auc_mean": 0.675, "lift_over_baseline_mean": 1.72},
        }

        result, best, reasons = ml_prebreakout._run17_target_change_decision([baseline, challenger], baseline)

        self.assertEqual(result, "TARGET_CHANGE_CANDIDATE")
        self.assertEqual(best, challenger)
        self.assertTrue(reasons)

    def test_run17_target_change_requires_full_baseline_folds(self):
        baseline = {
            "name": "A",
            "valid_fold_count": 3,
            "validation_summary": {"auc_mean": 0.66, "lift_over_baseline_mean": 1.7},
        }
        challenger = {
            "name": "B",
            "valid_fold_count": 5,
            "validation_summary": {"auc_mean": 0.675, "lift_over_baseline_mean": 1.9},
        }

        result, best, reasons = ml_prebreakout._run17_target_change_decision([baseline, challenger], baseline)

        self.assertEqual(result, "NO_TARGET_CHANGE")
        self.assertIsNone(best)
        self.assertTrue(any("requires 5" in reason for reason in reasons))

    def test_run17_target_change_ignores_incomplete_challenger_folds(self):
        baseline = {
            "name": "A",
            "valid_fold_count": 5,
            "validation_summary": {"auc_mean": 0.66, "lift_over_baseline_mean": 1.7},
        }
        challenger = {
            "name": "B",
            "valid_fold_count": 3,
            "validation_summary": {"auc_mean": 0.72, "lift_over_baseline_mean": 2.1},
        }

        result, best, reasons = ml_prebreakout._run17_target_change_decision([baseline, challenger], baseline)

        self.assertEqual(result, "NO_TARGET_CHANGE")
        self.assertIsNone(best)
        self.assertTrue(reasons)

    def test_run14_insufficient_qualifiers_status(self):
        evaluation = ml_prebreakout._insufficient_qualifiers_experiment(
            "J - Best overall combination",
            "requires two independently useful families",
            "best_overall_combination",
        )

        self.assertEqual(evaluation["experiment_status"], "SKIPPED_INSUFFICIENT_QUALIFIERS")
        self.assertEqual(evaluation["features_added"], [])

    def test_run10_promotion_requires_meaningful_auc_delta(self):
        champion = {
            "validation_summary": {
                "auc_mean": 0.6602646969335696,
                "min_fold_auc": 0.61,
                "lift_over_baseline_mean": 1.6489,
            }
        }
        candidate = {
            "experiment_status": "VALID",
            "requested_features": ["BBWidth20Pct"],
            "features_added": ["BBWidth20Pct"],
            "validation_summary": {
                "auc_mean": 0.6610,
                "min_fold_auc": 0.61,
                "lift_over_baseline_mean": 1.65,
            },
        }

        result, reasons = ml_prebreakout._run10_promotion_decision(candidate, champion, [])

        self.assertEqual(result, "TIE")
        self.assertTrue(any("below +0.003" in reason for reason in reasons))

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

    def test_add_prebreakout_target_rejects_late_future_setup(self):
        df = pd.DataFrame(
            [
                {
                    "Symbol": "LATE",
                    "Timestamp": pd.Timestamp("2026-01-05T15:00:00Z"),
                    "IsBreakout": False,
                    "BreakoutScore": 5.0,
                    "Last": 90.0,
                    "High20": 100.0,
                    "Return_5D": 0.0,
                },
                {
                    "Symbol": "LATE",
                    "Timestamp": pd.Timestamp("2026-01-12T15:00:00Z"),
                    "IsBreakout": True,
                    "BreakoutScore": 9.0,
                    "Last": 101.0,
                    "High20": 101.0,
                    "Return_5D": 0.05,
                },
            ]
        )

        labeled = ml_prebreakout.add_prebreakout_target_label(df)

        self.assertEqual(list(labeled["FutureQualitySetupHit"]), [0])

    def test_add_prebreakout_features_adds_price_slopes_from_sparkline(self):
        df = pd.DataFrame(
            {
                "Symbol": ["AAA"],
                "Timestamp": [pd.Timestamp("2026-01-08T15:00:00Z")],
                "Spark10D": [[95.0, 96.0, 97.0, 98.0, 99.0, 100.0]],
                "BreakoutScore": [5.0],
            }
        )

        featured = ml_prebreakout.add_prebreakout_features(df)

        self.assertAlmostEqual(float(featured.loc[0, "PriceReturn1D"]), 0.010101, places=6)
        self.assertAlmostEqual(float(featured.loc[0, "PriceReturn3D"]), 0.030928, places=6)
        self.assertAlmostEqual(float(featured.loc[0, "PriceSlope5D"]), 0.010526, places=6)

    def test_add_prebreakout_features_adds_symbol_history_deltas(self):
        df = pd.DataFrame(
            {
                "Symbol": ["AAA", "AAA", "AAA", "AAA"],
                "Timestamp": pd.date_range("2026-01-01", periods=4, freq="B", tz="UTC"),
                "BreakoutScore": [2.0, 4.0, 5.0, 8.0],
                "Trend10D%": [1.0, 2.0, 4.0, 7.0],
            }
        )

        featured = ml_prebreakout.add_prebreakout_features(df)

        self.assertEqual(float(featured.loc[3, "BreakoutScoreDelta1D"]), 3.0)
        self.assertEqual(float(featured.loc[3, "BreakoutScoreDelta3D"]), 6.0)
        self.assertEqual(float(featured.loc[3, "BreakoutScoreSlope3D"]), 2.0)
        self.assertEqual(float(featured.loc[3, "Trend10DDelta1D"]), 3.0)

    def test_add_market_regime_features_from_snapshot_context(self):
        ts = pd.Timestamp("2026-01-08T15:00:00Z")
        df = pd.DataFrame(
            [
                {
                    "Symbol": "SPY",
                    "Timestamp": ts,
                    "Trend10D%": 2.0,
                    "Trend20D%": 4.0,
                    "Volatility20D%": 1.1,
                    "PctChange": 0.5,
                    "Last": 100.0,
                    "EMA9": 101.0,
                    "EMA21": 99.0,
                },
                {
                    "Symbol": "QQQ",
                    "Timestamp": ts,
                    "Trend10D%": 3.0,
                    "Trend20D%": 6.0,
                    "Volatility20D%": 1.4,
                    "PctChange": 0.8,
                    "Last": 200.0,
                    "EMA9": 198.0,
                    "EMA21": 202.0,
                },
                {
                    "Symbol": "AAA",
                    "Timestamp": ts,
                    "PctChange": 2.0,
                    "Volatility20D%": 8.0,
                    "Trend10D%": 5.0,
                    "Trend20D%": 9.0,
                    "Last": 50.0,
                    "EMA21": 45.0,
                },
                {
                    "Symbol": "BBB",
                    "Timestamp": ts,
                    "PctChange": -1.0,
                    "Volatility20D%": 4.0,
                    "Trend10D%": -1.0,
                    "Trend20D%": 1.0,
                    "Last": 40.0,
                    "EMA21": 44.0,
                },
                {
                    "Symbol": "CCC",
                    "Timestamp": ts,
                    "PctChange": 1.0,
                    "Volatility20D%": 6.0,
                    "Trend10D%": 3.0,
                    "Trend20D%": 5.0,
                    "Last": 30.0,
                    "EMA21": 25.0,
                },
            ]
        )

        featured = ml_prebreakout.add_market_regime_features(df)
        aaa = featured.loc[featured["Symbol"] == "AAA"].iloc[0]

        self.assertEqual(float(aaa["SPYTrend10D"]), 2.0)
        self.assertEqual(float(aaa["QQQTrend20D"]), 6.0)
        self.assertEqual(float(aaa["SPYAboveEMA21"]), 1.0)
        self.assertEqual(float(aaa["QQQAboveEMA21"]), 0.0)
        self.assertAlmostEqual(float(aaa["SPYEMA9EMA21Spread"]), 2.0)
        self.assertAlmostEqual(float(aaa["QQQEMA9EMA21Spread"]), -2.0)
        self.assertEqual(float(aaa["SPYVolatility20D"]), 1.1)
        self.assertEqual(float(aaa["QQQVolatility20D"]), 1.4)
        self.assertEqual(float(aaa["RSvsSPY10D"]), 3.0)
        self.assertEqual(float(aaa["RSvsSPY20D"]), 5.0)
        self.assertEqual(float(aaa["RSvsQQQ10D"]), 2.0)
        self.assertEqual(float(aaa["RSvsQQQ20D"]), 3.0)
        self.assertAlmostEqual(float(aaa["MarketBreadthAboveEMA21"]), 2 / 3)

    def test_market_regime_features_never_use_future_benchmark_rows(self):
        df = pd.DataFrame(
            [
                {
                    "Symbol": "SPY",
                    "Timestamp": pd.Timestamp("2026-01-01T15:00:00Z"),
                    "Trend10D%": 1.0,
                    "Trend20D%": 2.0,
                },
                {
                    "Symbol": "AAA",
                    "Timestamp": pd.Timestamp("2026-01-02T15:00:00Z"),
                    "Trend10D%": 5.0,
                    "Trend20D%": 7.0,
                },
                {
                    "Symbol": "SPY",
                    "Timestamp": pd.Timestamp("2026-01-05T15:00:00Z"),
                    "Trend10D%": 9.0,
                    "Trend20D%": 12.0,
                },
            ]
        )

        featured = ml_prebreakout.add_market_regime_features(df)
        aaa = featured.loc[featured["Symbol"] == "AAA"].iloc[0]

        self.assertEqual(float(aaa["SPYTrend10D"]), 1.0)
        self.assertEqual(float(aaa["SPYTrend20D"]), 2.0)
        self.assertEqual(float(aaa["RSvsSPY10D"]), 4.0)

    def test_benchmark_context_from_bars_is_available_next_day(self):
        dates = pd.date_range("2026-01-01", periods=25, freq="B", tz="UTC")
        closes = [100.0] * (len(dates) - 1) + [125.0]
        bars = pd.DataFrame({"Close": closes}, index=dates)
        context = ml_prebreakout._benchmark_context_from_bars(bars, "SPY")
        df = pd.DataFrame(
            {
                "Symbol": ["AAA"],
                "Timestamp": [dates[-1]],
                "Trend10D%": [5.0],
                "Trend20D%": [8.0],
            }
        )

        featured = ml_prebreakout.add_market_regime_features(df, benchmark_context={"SPY": context})

        self.assertEqual(float(featured.loc[0, "SPYTrend10D"]), 0.0)
        next_day_df = df.copy()
        next_day_df["Timestamp"] = [dates[-1] + pd.Timedelta(1, unit="D")]
        next_day_featured = ml_prebreakout.add_market_regime_features(next_day_df, benchmark_context={"SPY": context})
        self.assertGreater(float(next_day_featured.loc[0, "SPYTrend10D"]), 0.0)

    def test_prebreakout_features_restore_original_index_after_sorting(self):
        df = pd.DataFrame(
            {
                "Symbol": ["BBB", "AAA", "AAA"],
                "Timestamp": [
                    pd.Timestamp("2026-01-03T15:00:00Z"),
                    pd.Timestamp("2026-01-02T15:00:00Z"),
                    pd.Timestamp("2026-01-01T15:00:00Z"),
                ],
                "BreakoutScore": [9.0, 5.0, 2.0],
                "Trend10D%": [1.0, 4.0, 1.0],
            },
            index=[20, 10, 5],
        )

        featured = ml_prebreakout.add_prebreakout_features(df)

        self.assertEqual(list(featured.index), [20, 10, 5])
        self.assertEqual(float(featured.loc[10, "BreakoutScoreDelta1D"]), 3.0)

    def test_build_dataset_and_scoring_share_market_feature_columns(self):
        df = pd.DataFrame(
            {
                "Symbol": ["SPY", "AAA"],
                "Timestamp": [pd.Timestamp("2026-01-08T15:00:00Z")] * 2,
                "Trend10D%": [1.0, 4.0],
                "Trend20D%": [2.0, 6.0],
                "VolRel20": [1.0, 2.0],
                "DollarVol20": [1_000_000.0, 2_000_000.0],
                "BreakoutScore": [1.0, 5.0],
                "GapPct": [0.1, 0.2],
                "FutureQualitySetupHit": [0, 1],
            }
        )

        x, _ = ml_prebreakout.build_ml_dataset(df)
        fake_model = FakePrebreakoutClassifier()
        bundle = {"model": fake_model, "features": list(x.columns)}

        with patch.object(ml_prebreakout, "load_prebreakout_model", return_value=bundle):
            scored = ml_prebreakout.score_prebreakout(df.copy())

        self.assertIn("SPYTrend10D", x.columns)
        self.assertIn("RSvsSPY10D", x.columns)
        self.assertEqual(float(scored.loc[0, "PreBreakoutProb"]), 0.6)

    def test_build_ml_dataset_deduplicates_feature_columns(self):
        df = pd.DataFrame(
            {
                "Symbol": ["AAA", "AAA"],
                "Timestamp": pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC"),
                "Trend10D%": [1.0, 2.0],
                "DistanceTo20DHighPct": [-0.02, -0.01],
                "FutureQualitySetupHit": [0, 1],
            }
        )

        x, _ = ml_prebreakout.build_ml_dataset(
            df,
            include_market_features=False,
            feature_cols=["Trend10D%", "DistanceTo20DHighPct", "DistanceTo20DHighPct"],
        )

        self.assertEqual(list(x.columns), ["Trend10D%", "DistanceTo20DHighPct"])

    def test_score_prebreakout_neutral_fills_missing_market_data(self):
        fake_model = FakePrebreakoutClassifier()
        bundle = {
            "model": fake_model,
            "features": ["Trend10D%", "SPYTrend10D", "RSvsSPY10D"],
        }
        df = pd.DataFrame({"Symbol": ["AAA"], "Timestamp": [pd.Timestamp("2026-01-08T15:00:00Z")], "Trend10D%": [3.0]})

        with patch.object(ml_prebreakout, "load_prebreakout_model", return_value=bundle):
            scored = ml_prebreakout.score_prebreakout(df)

        self.assertEqual(float(scored.loc[0, "PreBreakoutProb"]), 0.6)

    def test_walk_forward_split_validates_on_later_rows(self):
        x = pd.DataFrame({"feature": [10, 20, 30, 40, 50]})
        y = pd.Series([0, 1, 0, 1, 1])
        labeled = pd.DataFrame({"Timestamp": pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")})

        x_train, x_val, y_train, y_val = ml_prebreakout.walk_forward_split(x, y, labeled, validation_fraction=0.4)

        self.assertEqual(list(x_train["feature"]), [10, 20, 30])
        self.assertEqual(list(x_val["feature"]), [40, 50])
        self.assertEqual(list(y_train), [0, 1, 0])
        self.assertEqual(list(y_val), [1, 1])

    def test_expanding_window_folds_apply_trading_day_purge(self):
        x = pd.DataFrame({"feature": range(42)})
        y = pd.Series([0, 1] * 21)
        labeled = pd.DataFrame(
            {
                "Symbol": ["AAA"] * 42,
                "Timestamp": pd.date_range("2026-01-01", periods=42, freq="B", tz="UTC"),
            }
        )

        folds = ml_prebreakout.expanding_window_folds(x, y, labeled, n_splits=5, purge_days=5)

        self.assertEqual(len(folds), 5)
        first = folds[0]
        self.assertEqual(first["val_idx"], list(range(7, 14)))
        self.assertEqual(first["train_idx"], [0, 1])
        validation_start = labeled.loc[first["val_idx"][0], "Timestamp"].date()
        self.assertTrue(
            all(
                np.busday_count(labeled.loc[idx, "Timestamp"].date(), validation_start) > 5
                for idx in first["train_idx"]
            )
        )

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
