"""Lazy DB package exports.

Keep ``import db`` lightweight: optional UI/DB dependencies such as Streamlit,
Pandas, and psycopg should only be needed when callers use the features that
actually require them.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "get_db_status": ("db.engine", "get_db_status"),
    "get_neon_conn": ("db.engine", "get_neon_conn"),
    "get_sqlite_conn": ("db.engine", "get_sqlite_conn"),
    "save_run": ("db.runs", "save_run"),
    "save_daily_snapshot": ("db.runs", "save_daily_snapshot"),
    "list_runs": ("db.runs", "list_runs"),
    "load_run_results": ("db.runs", "load_run_results"),
    "load_users": ("db.users", "load_users"),
    "fetch_all_users": ("db.users", "fetch_all_users"),
    "seed_neon_users_from_local": ("db.users", "seed_neon_users_from_local"),
    "ensure_neon_runs_schema": ("db.schema", "ensure_neon_runs_schema"),
    "ensure_sqlite_runs_schema": ("db.schema", "ensure_sqlite_runs_schema"),
    "ensure_neon_users_schema": ("db.schema", "ensure_neon_users_schema"),
    "ensure_sqlite_users_schema": ("db.schema", "ensure_sqlite_users_schema"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
