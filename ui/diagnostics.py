

"""Lightweight diagnostics utilities for the Streamlit AI Scanner UI.

These helpers are intentionally dependency-light and safe to import anywhere
inside the app. They NO-OP when Streamlit is not available.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:  # Optional Streamlit dependency (safe in non-UI contexts)
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------

def _callable_name(fn: Any) -> str:
    mod = getattr(fn, "__module__", "") or ""
    name = getattr(fn, "__name__", "") or type(fn).__name__
    full = f"{mod}.{name}".strip(".")
    return full or "<unknown>"


def _df_rows(df: Any) -> Optional[int]:
    if pd is None:
        return None
    if isinstance(df, pd.DataFrame):
        try:
            return int(len(df))
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------------------
# Timing utilities
# --------------------------------------------------------------------------------------

@dataclass
class RunResult:
    duration_s: float
    rows: Optional[int]


@contextlib.contextmanager
def timer() -> Any:
    """Context manager measuring wall-clock duration in seconds."""
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0


# --------------------------------------------------------------------------------------
# Public API – UI diagnostics (safe without Streamlit)
# --------------------------------------------------------------------------------------

def show_runner_info(label: str, target: Any) -> None:
    """Show which concrete function will run for a given UI button.

    Parameters
    ----------
    label: Button label / human-readable name.
    target: The callable that will run (or wrapper with `_target_fn` attrs).
    """
    if st is None:  # pragma: no cover
        return
    fn = getattr(target, "_target_fn", target)
    name = getattr(target, "_target_name", None) or _callable_name(fn)
    st.caption(f"Using: {name}")


def summarize_result(result: Any, *, df_hint_keys: tuple[str, ...] = ("df", "data", "results", "table")) -> RunResult:
    """Inspect a runner result and compute basic summary (duration handled by caller).

    Returns a RunResult with `rows` best-effort inferred.
    """
    rows: Optional[int] = None
    if pd is not None and isinstance(result, pd.DataFrame):
        rows = _df_rows(result)
    elif isinstance(result, dict):
        # try common dataframe keys first
        if pd is not None:
            for k in df_hint_keys:
                v = result.get(k)
                if isinstance(v, pd.DataFrame):
                    rows = _df_rows(v)
                    break
        # fall back to run-id style dicts (rows unknown at this stage)
    elif isinstance(result, (list, tuple)) and pd is not None:
        for item in result:
            if isinstance(item, pd.DataFrame):
                rows = _df_rows(item)
                break
    return RunResult(duration_s=0.0, rows=rows)


def show_result_summary(label: str, rr: RunResult) -> None:
    """Render a small caption with runtime and row count."""
    if st is None:  # pragma: no cover
        return
    parts = []
    parts.append(f"⏱ {rr.duration_s:.2f}s")
    if rr.rows is not None:
        parts.append(f"🧮 {rr.rows:,} rows")
    st.caption("  ·  ".join(parts))


def rate_limit_hint(source: str = "Yahoo Finance") -> None:
    """Display a one-line hint for suspected rate limiting."""
    if st is None:  # pragma: no cover
        return
    st.caption(f"If this table is empty, {source} may be rate limiting. Try again in ~60s.")


# --------------------------------------------------------------------------------------
# Convenience one-shot runner (optional use in buttons)
# --------------------------------------------------------------------------------------

def run_with_diagnostics(label: str, fn: Callable[[], Any]) -> Any:
    """Execute a callable, render runtime + row summary, and return the result.

    Use in UI handlers:
        res = run_with_diagnostics("Run S&P 500 Breakout", bound_runner)
    """
    with timer() as elapsed:
        result = fn()
    rr = summarize_result(result)
    rr.duration_s = elapsed()
    show_result_summary(label, rr)
    return result