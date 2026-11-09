"""One-off: run backfill_scanner_opportunities.py --apply (owner approved 2026-11-09).

Lets the manual Diagnostics workflow, which runs scripts without arguments, do the
write once on a branch ref. Not merged into dev/main.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backfill_scanner_opportunities import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--apply"]))
