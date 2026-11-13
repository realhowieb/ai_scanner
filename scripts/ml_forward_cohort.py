"""Register or collect the predeclared forward cohort; never read outcome data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="Explicitly write the prospective protocol before its start")
    parser.add_argument("--protocol", default=str(ROOT / "docs/ml_forward_cohort_2026-12-23.json"))
    args = parser.parse_args()
    from db import ml_forward_cohorts as store

    protocol = json.loads(Path(args.protocol).read_text())
    if args.register:
        store.register(protocol)
    print(json.dumps(store.collect(protocol), indent=2))


if __name__ == "__main__":
    main()
