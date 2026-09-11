"""Attach an isotonic calibration map to the live PreBreakout champion, in place.

Promotions carry a calibration map automatically, but when no challenger beats
the champion the live model never gains one and its displayed probability stays
raw/over-confident. This fits the map from the champion's own stored calibration
buckets and patches the active Neon row's metadata — no retrain, no model swap.

Run from an environment with the Neon secret set (NEON_DATABASE_URL/DATABASE_URL).
Exit code 0 on success or a benign skip (already calibrated); 1 on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Running `python scripts/foo.py` puts scripts/ on sys.path, not the repo root
# where ml_prebreakout.py lives; add it so the import resolves (matches
# scripts/train_ai_confidence_model.py).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import ml_prebreakout

    result = ml_prebreakout.recalibrate_active_champion()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result.get("ok"):
        print(f"[recalibrate] FAILED: {result.get('reason')}")
        return 1
    if result.get("skipped"):
        print(f"[recalibrate] Nothing to do: {result.get('skipped')}")
        return 0
    print(
        f"[recalibrate] Champion {result.get('model_version')} calibrated in place "
        f"({result.get('calibration_points')} points from n={result.get('n')})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
