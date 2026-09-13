"""Run 30 — centralized configuration validation + build identification.

Read-only, side-effect-free, secret-safe. Reports WHICH configuration is present
by NAME and purpose only — it never reads, returns, logs, or echoes any secret
VALUE. A missing OPTIONAL provider degrades a feature; a missing REQUIRED value
means the app cannot operate. Used by admin diagnostics and the deploy smoke test.

(Standalone module, not a package — the repo already has a `config.py` module.)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# name, required, purpose, failure behavior if absent. VALUES ARE NEVER READ HERE
# beyond presence. Alternatives (A|B) mean any one satisfies the requirement.
_CONFIG_SPEC = [
    ("NEON_DATABASE_URL|DATABASE_URL", True, "Primary Neon/Postgres database",
     "App cannot persist or read intelligence; pages degrade to empty state."),
    ("COOKIE_PASSWORD", False, "Auth cookie signing (login persistence)",
     "Login sessions may not persist across reloads."),
    ("ALPACA_API_KEY_ID", False, "Alpaca market-data key id",
     "Primary market data degrades; fallback provider used where available."),
    ("ALPACA_API_SECRET_KEY", False, "Alpaca market-data secret",
     "Primary market data degrades; fallback provider used where available."),
    ("RESEND_API_KEY", False, "Transactional email delivery (alerts)",
     "Email alerts are skipped; in-app alert feed still works."),
    ("ADMIN_USERS", False, "Comma-separated admin usernames",
     "Admin diagnostics surfaces are hidden; core product unaffected."),
]


def _present(name_spec: str) -> bool:
    """True if any of the alternative env names (A|B) is set and non-empty."""
    for name in name_spec.split("|"):
        if str(os.environ.get(name) or "").strip():
            return True
    return False


def validate_config() -> Dict[str, Any]:
    """Return configuration health WITHOUT any secret values.

    {ok, missing_required: [names], items: [{name, required, configured,
    purpose, failure_behavior}]}. `ok` is False only when a REQUIRED item is
    absent — an optional gap never fails validation.
    """
    items: List[Dict[str, Any]] = []
    missing_required: List[str] = []
    for name, required, purpose, failure in _CONFIG_SPEC:
        configured = _present(name)
        if required and not configured:
            missing_required.append(name)
        items.append({
            "name": name, "required": bool(required), "configured": bool(configured),
            "purpose": purpose, "failure_behavior": failure,
        })
    return {"ok": not missing_required, "missing_required": missing_required, "items": items}


def _first_env(*names: str) -> Optional[str]:
    for n in names:
        v = str(os.environ.get(n) or "").strip()
        if v:
            return v
    return None


def build_info() -> Dict[str, Any]:
    """Identify the deployed build for admin diagnostics — commit SHA, environment
    label, and version. Commit SHA is read from common deploy env vars first (no
    shelling out on the hot path); falls back to a short `git` read only when none
    are set. Never exposes secrets."""
    sha = _first_env("HSF_COMMIT_SHA", "GIT_COMMIT", "RENDER_GIT_COMMIT",
                     "SOURCE_VERSION", "STREAMLIT_COMMIT_SHA", "COMMIT_SHA")
    if not sha:
        try:
            import subprocess

            sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip() or None
        except Exception:
            sha = None
    return {
        "commit_sha": sha or "unknown",
        "environment": _first_env("HSF_ENV", "APP_ENV", "ENVIRONMENT") or "production",
        "version": "HSF Market Intelligence V1.0 RC",
    }


def config_health_summary() -> Dict[str, Any]:
    """Compact admin view: config + build in one read-only object (no secrets)."""
    cfg = validate_config()
    return {
        "build": build_info(),
        "ok": cfg["ok"],
        "missing_required": cfg["missing_required"],
        "configured": [i["name"] for i in cfg["items"] if i["configured"]],
        "degraded": [i["name"] for i in cfg["items"] if not i["configured"] and not i["required"]],
        "items": cfg["items"],
    }
