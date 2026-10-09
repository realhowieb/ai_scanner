"""Run an existing module unchanged with aggregate DB telemetry, even on failure."""
import json
import logging
import runpy
import sys
import time
from pathlib import Path

from db.traffic import scope


def main():
    module = sys.argv[1]
    if not module.startswith(("scripts.", "scheduler.", "analytics.")):
        raise SystemExit("Expected a scripts.*, scheduler.* or analytics.* module")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.argv = [module, *sys.argv[2:]]
    started = time.perf_counter()
    try:
        with scope(module) as metrics:
            runpy.run_module(module, run_name="__main__")
    finally:
        try:
            path = Path("artifacts/db_traffic") / (module.replace(".", "_") + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            report = dict(scope=module, measurement="application_payload_estimate",
                          elapsed_ms=round((time.perf_counter()-started)*1000, 2), **metrics.snapshot())
            path.write_text(json.dumps(report, indent=2) + "\n")
        except OSError:
            logging.getLogger("hsf.db_traffic").warning("Could not write aggregate traffic artifact")


if __name__ == "__main__":
    main()
