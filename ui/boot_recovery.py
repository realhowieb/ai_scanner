"""Bounded self-heal for a stale first-party module after a Streamlit Cloud redeploy.

Run 83B follow-up. The Run 81 hotfix covers app.py's first import block; this
covers the second one (the big feature-import block), which turns any
ImportError into a permanent "Startup problem" page. After a redeploy the
process can still hold the previous version of one of our modules, so a name
added in the same commit is "missing" although it is on disk. Dropping that
module and rerunning imports the file on disk. Retries are bounded, so a real
missing dependency still surfaces as the original error.

Kept in its own module on purpose: a module that did not exist before a deploy
cannot itself be cached stale.
"""
from __future__ import annotations

import sys
from typing import Any, MutableMapping, Optional

FIRST_PARTY = frozenset({"ui", "auth", "db", "scan", "data", "utils", "analytics", "scheduler", "integrations"})
RETRY_KEY = "_boot_feature_import_retries"
MAX_RETRIES = 3


def retry_stale_import(module_name: Optional[str], session_state: MutableMapping[str, Any]) -> bool:
    """Drop a stale first-party module and return True if the caller should rerun."""
    name = str(module_name or "")
    if name.split(".")[0] not in FIRST_PARTY:
        return False
    tries = int(session_state.get(RETRY_KEY, 0) or 0)
    if tries >= MAX_RETRIES:
        return False
    session_state[RETRY_KEY] = tries + 1
    sys.modules.pop(name, None)
    return True


def clear_retries(session_state: MutableMapping[str, Any]) -> None:
    session_state.pop(RETRY_KEY, None)
