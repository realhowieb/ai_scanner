"""ai_scanner package bootstrap.

This package initializer keeps imports lightweight and forwards a few
high-level helpers for convenience while avoiding circular imports.

Structure (after migration):
- ai_scanner/db/engine.py    -> get_engine(), ensure_schema()
- ai_scanner/db/schema.py    -> Base metadata, migration helpers
- ai_scanner/db/runs.py      -> save_run(), list_runs(), load_run_results()
- ai_scanner/scan/...        -> scanning logic (breakout, gap/unusual, ta)
- ai_scanner/data/...        -> universe, fetch, filters, market time utils
- ai_scanner/scheduler/...   -> apscheduler jobs + optional UI hooks
- ai_scanner/logging_utils   -> UI/console logger adapters
- ai_scanner/config.py       -> get_settings() for central config
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

# Cache for lazily imported modules so we don't re-import repeatedly
_LAZY_CACHE: dict[str, Any] = {}

__all__ = [
    "__version__",
    "PKG_DIR",
    "get_engine",
    "ensure_schema",
    "init_db",
    "save_run",
    "list_runs",
    "load_run_results",
    "lazy_submodule",
    "get_settings",
    "run_scan",
    "run_and_save",
    "run_premarket_headless",
    "run_postmarket_headless",
    "run_sp500_headless",
    "render_app",
]

__version__ = "0.2.0"


# Absolute path of this package directory
PKG_DIR: Path = Path(__file__).resolve().parent

# Ensure project repo root is on sys.path for local runs (supports implicit namespace packages, PEP 420)
def _ensure_repo_root_on_path() -> None:
    root = PKG_DIR.parent
    # Insert at the front so it takes precedence over site-packages
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

_ensure_repo_root_on_path()

# ---------- Namespace shims for local development ----------
# If running locally without `pip install -e .`, map top-level packages like
# `ui/`, `db/`, etc. into the `ai_scanner.*` namespace so imports like
# `import ai_scanner.ui.pages_main` still work.
for _name in ("ui", "db", "data", "scan", "scheduler", "logging_utils"):
    try:
        # If submodule isn't already registered under ai_scanner, alias it
        if f"{__name__}.{_name}" not in sys.modules:
            _mod = import_module(_name)
            # Mirror its __path__ so subimports (e.g., pages_main) resolve
            sys.modules[f"{__name__}.{_name}"] = _mod
    except Exception:
        # It's fine if a local package doesn't exist; we only alias what we find
        pass


# ---------- Lazy import helpers ----------

def _lazy(name: str, attr: str | None = None) -> Any:
    """Import a module (and optional attribute) lazily.

    Parameters
    ----------
    name: str
        Dotted module path relative to the ai_scanner package or absolute.
    attr: str | None
        Optional attribute to fetch from the imported module.
    """
    if not name.startswith("ai_scanner"):
        name = f"ai_scanner.{name}"
    mod = _LAZY_CACHE.get(name)
    if mod is None:
        mod = import_module(name)
        _LAZY_CACHE[name] = mod
    return getattr(mod, attr) if attr else mod


def lazy_submodule(name: str) -> Any:
    """Return a submodule on first access, importing it as needed.

    Example
    -------
    >>> runs = lazy_submodule("db.runs")
    >>> runs.save_run(...)
    """
    return _lazy(name)


# ---------- Forwarded convenience functions (lazy) ----------
# UI entrypoint forwarder for diagnostic probe/launch
def render_app(*args: Any, **kwargs: Any):
    """Forward to ai_scanner.ui.pages.render_app (lazy)."""
    fn: Callable[..., Any] = _lazy("ui.pages", "render_app")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def get_engine(*args: Any, **kwargs: Any):
    """Return a SQLAlchemy engine from ai_scanner.db.engine.get_engine.

    Kept here to give callers a stable, short import path:
        from ai_scanner import get_engine
    """
    fn: Callable[..., Any] = _lazy("db.engine", "get_engine")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def ensure_schema(engine=None) -> None:
    """Create/upgrade schema using ai_scanner.db.schema.ensure_schema."""
    fn: Callable[..., Any] = _lazy("db.schema", "ensure_schema")  # type: ignore[assignment]
    return fn(engine)


def init_db(*args: Any, **kwargs: Any):
    """Initialize database engine + schema (idempotent).

    This calls ai_scanner.db.engine.init_db lazily so importing ai_scanner
    stays lightweight while still exposing a stable entry point.
    """
    fn: Callable[..., Any] = _lazy("db.engine", "init_db")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def get_settings(*args: Any, **kwargs: Any):
    """Return application settings from ai_scanner.config.get_settings."""
    fn: Callable[..., Any] = _lazy("config", "get_settings")  # type: ignore[assignment]
    return fn(*args, **kwargs)


# The run helpers are the most frequently used; expose them at package level.
# We resolve them lazily so importing ai_scanner at app startup is cheap and
# also resilient during partial migrations.

def save_run(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("db.runs", "save_run")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def list_runs(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("db.runs", "list_runs")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def load_run_results(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("db.runs", "load_run_results")  # type: ignore[assignment]
    return fn(*args, **kwargs)


# ---------- Headless scan helpers (optional, used by scheduler/CLI) ----------

def run_scan(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("scan.pre_post", "run_scan")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def run_and_save(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("scan.pre_post", "run_and_save")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def run_premarket_headless(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("scan.pre_post", "run_premarket_headless")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def run_postmarket_headless(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("scan.pre_post", "run_postmarket_headless")  # type: ignore[assignment]
    return fn(*args, **kwargs)


def run_sp500_headless(*args: Any, **kwargs: Any):
    fn: Callable[..., Any] = _lazy("scan.pre_post", "run_sp500_headless")  # type: ignore[assignment]
    return fn(*args, **kwargs)


# ---------- Optional: PEP 562 module-level __getattr__ for dotted access ----------
# Allows `from ai_scanner import db` to work as a lazy module. This keeps old
# imports alive during the refactor without pulling heavy modules until used.

def __getattr__(name: str) -> Any:  # pragma: no cover - trivial passthrough
    # Known first-level subpackages
    if name in {"db", "scan", "data", "scheduler", "logging_utils", "ui"}:
        return _lazy(name)
    raise AttributeError(name)