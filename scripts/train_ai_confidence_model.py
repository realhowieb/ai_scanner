"""Train and optionally upload the AI Confidence XGBoost model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_prebreakout import load_run_history  # noqa: E402
from scan.ai_confidence import (  # noqa: E402
    METADATA_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    save_ai_confidence_model_from_files,
)

try:
    import joblib
except Exception:  # pragma: no cover - optional ML dependency
    joblib = None  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency in import smoke jobs
    pd = None  # type: ignore[assignment]

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - optional ML dependency
    roc_auc_score = None  # type: ignore[assignment]
    train_test_split = None  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional ML dependency
    XGBClassifier = None  # type: ignore[assignment]


FEATURE_NAMES = [
    "Trend10D%",
    "Trend20D%",
    "VolRel20",
    "DollarVol20",
    "BreakoutScore",
    "GapPct",
]


def _as_binary_label(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y"} else 0
    return 1 if bool(value) else 0


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_ai_confidence_dataset(frame: Any) -> tuple[Any, Any]:
    """Build the model matrix from historical scan rows."""
    if pd is None or frame is None or frame.empty:
        return pd.DataFrame(columns=FEATURE_NAMES) if pd is not None else None, None
    if "IsBreakout" not in frame.columns:
        return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int)

    working = frame.copy()
    for col in FEATURE_NAMES:
        if col not in working.columns:
            working[col] = 0.0

    x = working.loc[:, FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = working["IsBreakout"].apply(_as_binary_label).astype(int)
    valid = y.notna()
    return x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def _can_stratify(y: Any) -> bool:
    counts = y.value_counts()
    return len(counts) == 2 and int(counts.min()) >= 2


def train_ai_confidence_model(
    *,
    days_back: int = 90,
    max_runs: int = 2000,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    upload_db: bool = False,
) -> dict[str, Any]:
    """Train the XGBoost model and write the expected model/metadata files."""
    if joblib is None or pd is None or roc_auc_score is None or train_test_split is None or XGBClassifier is None:
        raise RuntimeError("ML dependencies are not installed. Install requirements-ml.txt first.")

    history = load_run_history(days_back=days_back, max_runs=max_runs)
    x, y = build_ai_confidence_dataset(history)
    if x is None or y is None or x.empty:
        raise RuntimeError("No training rows are available from scan history.")
    if y.nunique() < 2:
        raise RuntimeError("Training data needs both breakout and non-breakout rows.")

    stratify = y if _can_stratify(y) else None
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)

    auc = None
    if y_val.nunique() >= 2:
        auc = float(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))

    metadata = {
        "feature_names": FEATURE_NAMES,
        "trained_at": _utc_timestamp(),
        "auc": auc,
        "rows": int(len(x)),
        "positive_rows": int(y.sum()),
        "model_version": MODEL_VERSION,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    saved_to_db = False
    if upload_db:
        saved_to_db = save_ai_confidence_model_from_files(
            model_path=model_path,
            metadata_path=metadata_path,
            model_version=MODEL_VERSION,
        )
        if not saved_to_db:
            raise RuntimeError("Model files were written, but DB upload failed.")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "saved_to_db": saved_to_db,
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-back", type=int, default=90)
    parser.add_argument("--max-runs", type=int, default=2000)
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--upload-db", action="store_true", help="Upload the trained model to the configured DB")
    args = parser.parse_args()

    summary = train_ai_confidence_model(
        days_back=args.days_back,
        max_runs=args.max_runs,
        model_path=Path(args.model),
        metadata_path=Path(args.metadata),
        upload_db=args.upload_db,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
