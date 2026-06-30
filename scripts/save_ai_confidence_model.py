"""Save the local AI Confidence model artifact into the configured database."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scan.ai_confidence import METADATA_PATH, MODEL_PATH, save_ai_confidence_model_from_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_PATH), help="Path to xgb_breakout_model.joblib")
    parser.add_argument("--metadata", default=str(METADATA_PATH), help="Path to xgb_breakout_metadata.json")
    args = parser.parse_args()

    saved = save_ai_confidence_model_from_files(
        model_path=Path(args.model),
        metadata_path=Path(args.metadata),
    )
    if not saved:
        raise SystemExit("AI Confidence model was not saved. Check model file, metadata feature_names, and DB config.")
    print("AI Confidence model saved to database.")


if __name__ == "__main__":
    main()
