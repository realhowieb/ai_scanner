"""Deployment readiness checks for the scanner runtime."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _exists(path: str) -> CheckResult:
    file_path = ROOT / path
    return CheckResult(
        name=f"file:{path}",
        ok=file_path.exists(),
        detail="present" if file_path.exists() else "missing",
    )


def _env_any(name: str, candidates: tuple[str, ...], *, required: bool) -> CheckResult:
    found = [key for key in candidates if os.getenv(key)]
    return CheckResult(
        name=name,
        ok=bool(found) or not required,
        detail=f"found {', '.join(found)}" if found else f"missing any of {', '.join(candidates)}",
        required=required,
    )


def _import_module(module: str, *, required: bool = True) -> CheckResult:
    try:
        importlib.import_module(module)
    except Exception as exc:
        return CheckResult(module, ok=not required, detail=f"{type(exc).__name__}: {exc}", required=required)
    return CheckResult(module, ok=True, detail="import ok", required=required)


def run_checks(*, strict_env: bool = False) -> list[CheckResult]:
    checks = [
        _exists("requirements-core.txt"),
        _exists("requirements.txt"),
        _exists(".github/workflows/smoke.yml"),
        _exists(".github/workflows/scheduled-scans.yml"),
        _env_any("database-url", ("NEON_DATABASE_URL", "DATABASE_URL"), required=strict_env),
        _env_any("alpaca-key", ("ALPACA_API_KEY_ID",), required=strict_env),
        _env_any("alpaca-secret", ("ALPACA_API_SECRET_KEY",), required=strict_env),
        _import_module("config"),
        _import_module("data.prices"),
        _import_module("scan.engine"),
        _import_module("scheduler.cron_runner"),
        _import_module("db.runs"),
        _import_module("ui.scans"),
    ]
    return checks


def print_report(checks: list[CheckResult]) -> None:
    for check in checks:
        status = "OK" if check.ok else ("WARN" if not check.required else "FAIL")
        print(f"[{status}] {check.name}: {check.detail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-env", action="store_true", help="fail when production secrets are missing")
    args = parser.parse_args()

    checks = run_checks(strict_env=args.strict_env)
    print_report(checks)

    if any((not check.ok) and check.required for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
