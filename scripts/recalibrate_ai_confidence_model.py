"""Attach an isotonic calibration map to the live AI Confidence model, in place.

The live model may carry no calibration map (e.g. trained before calibration
existed), so its displayed confidence is the raw, over-confident score. This
fits the map from the model's own stored calibration buckets and patches the
active Neon row's metadata — no retrain, no model swap.

Run from an environment with the Neon secret set (NEON_DATABASE_URL/DATABASE_URL).
Exit code 0 on success or a benign skip (already calibrated); 1 on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from scan.ai_confidence import recalibrate_active_ai_confidence

    result = recalibrate_active_ai_confidence()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result.get("ok"):
        print(f"[recalibrate-ai] FAILED: {result.get('reason')}")
        return 1
    if result.get("skipped"):
        print(f"[recalibrate-ai] Nothing to do: {result.get('skipped')}")
        return 0
    print(
        f"[recalibrate-ai] Model {result.get('model_version')} calibrated in place "
        f"({result.get('calibration_points')} points from n={result.get('n')})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
