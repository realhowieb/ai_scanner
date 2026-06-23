"""Manual scan runner discovery and Streamlit session binding."""

from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st
from ui.market_heat import fetch_hot_stocks, fetch_most_active_stocks


def _optional_attr(module_name: str, attr_name: str):
    """Return an optional attribute from a module without failing page import."""
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        return None


def _discover_runner(candidates: tuple[tuple[str, str], ...]):
    for module_name, attr_name in candidates:
        runner = _optional_attr(module_name, attr_name)
        if runner is not None:
            return runner
    return None


def _fallback_sp500(
    universe: str = "S&P 500",
    min_price: float | None = None,
    max_price: float | None = None,
    min_dollar_vol: float | None = None,
    **kwargs,
):
    del universe, kwargs
    df = fetch_hot_stocks(count=100) if callable(fetch_hot_stocks) else pd.DataFrame()
    return _apply_basic_filters(df, min_price, max_price, min_dollar_vol)


def _fallback_nasdaq(
    universe: str = "Nasdaq 100",
    min_price: float | None = None,
    max_price: float | None = None,
    min_dollar_vol: float | None = None,
    **kwargs,
):
    del universe, kwargs
    df = fetch_most_active_stocks(count=100) if callable(fetch_most_active_stocks) else pd.DataFrame()
    return _apply_basic_filters(df, min_price, max_price, min_dollar_vol)


def _fallback_premarket(
    min_price: float | None = None,
    max_price: float | None = None,
    min_dollar_vol: float | None = None,
    **kwargs,
):
    del kwargs
    df = fetch_most_active_stocks(count=50) if callable(fetch_most_active_stocks) else pd.DataFrame()
    return _apply_basic_filters(df, min_price, max_price, min_dollar_vol)


def _fallback_postmarket(
    min_price: float | None = None,
    max_price: float | None = None,
    min_dollar_vol: float | None = None,
    **kwargs,
):
    del kwargs
    df = fetch_hot_stocks(count=50) if callable(fetch_hot_stocks) else pd.DataFrame()
    return _apply_basic_filters(df, min_price, max_price, min_dollar_vol)


def _apply_basic_filters(
    df: pd.DataFrame,
    min_price: float | None,
    max_price: float | None,
    min_dollar_vol: float | None,
) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame) and not df.empty:
        if min_price is not None and "price" in df:
            df = df[df["price"] >= float(min_price)]
        if max_price is not None and max_price > 0 and "price" in df:
            df = df[df["price"] <= float(max_price)]
        if min_dollar_vol is not None and "price" in df and "volume" in df:
            df = df[(df["price"] * df["volume"]) >= float(min_dollar_vol)]
    return df.reset_index(drop=True)


def _collect_scan_context() -> dict:
    """Gather common scan parameters from session_state."""
    return {
        "universe": st.session_state.get("universe"),
        "min_price": st.session_state.get("min_price"),
        "max_price": st.session_state.get("max_price"),
        "min_dollar_vol": st.session_state.get("min_dollar_vol"),
        "include_ta": st.session_state.get("include_ta"),
        "apply_gap_filter": st.session_state.get("apply_gap_filter"),
        "show_diagnostics_ui": st.session_state.get("show_diagnostics_ui"),
    }


def _map_universe(val: str) -> str:
    mapping = {
        "S&P 500": "SP500",
        "Nasdaq 100": "NASDAQ100",
        "S&P 600": "SP600",
        "All (US)": "US",
        "Custom": "CUSTOM",
    }
    return mapping.get(val, val)


def _bind_session_args(fn):
    """Invoke a runner with compatible values from Streamlit session_state."""
    if fn is None:
        return None

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    fn_name = getattr(fn, "__name__", "") or ""
    fn_mod = getattr(fn, "__module__", "") or ""

    def _runner():
        ctx = _collect_scan_context()
        kwargs = {}

        for key in (
            "universe",
            "min_price",
            "max_price",
            "include_ta",
            "apply_gap_filter",
            "show_diagnostics_ui",
        ):
            if key in accepted and key in ctx:
                kwargs[key] = ctx[key]

        if "min_dollar_vol" in ctx:
            if "min_dollar_vol" in accepted:
                kwargs["min_dollar_vol"] = ctx["min_dollar_vol"]
            elif "min_dollar_volume" in accepted:
                kwargs["min_dollar_volume"] = ctx["min_dollar_vol"]
            elif "dollar_volume_min" in accepted:
                kwargs["dollar_volume_min"] = ctx["min_dollar_vol"]

        universe_val = ctx.get("universe")
        if universe_val and "index" in accepted and "universe" not in accepted:
            kwargs["index"] = _map_universe(universe_val)

        if "session" in accepted and ("premarket" in fn_name or "premarket" in fn_mod):
            kwargs["session"] = "pre"
        if "session" in accepted and ("postmarket" in fn_name or "postmarket" in fn_mod):
            kwargs["session"] = "post"

        if ("sp500" in fn_name or "sp500" in fn_mod) and "universe" not in accepted and "index" in accepted:
            kwargs.setdefault("index", "SP500")
        if ("nasdaq" in fn_name or "nasdaq" in fn_mod) and "universe" not in accepted and "index" in accepted:
            kwargs.setdefault("index", "NASDAQ100")

        try:
            return fn(**kwargs) if kwargs else fn()
        except TypeError:
            return fn()

    setattr(_runner, "_target_fn", fn)
    setattr(_runner, "_target_name", f"{getattr(fn, '__module__', '')}.{getattr(fn, '__name__', '')}")
    return _runner


def _call_with_overrides(bound_fn, universe_token: str, universe_name: str):
    """Call the underlying target runner with a forced universe/index."""
    if bound_fn is None:
        raise RuntimeError("Runner function is not available")
    target = getattr(bound_fn, "_target_fn", bound_fn)
    sig = inspect.signature(target)
    accepted = set(sig.parameters.keys())

    ctx = _collect_scan_context()
    kwargs = {}

    for key in ("min_price", "max_price"):
        if key in accepted and ctx.get(key) is not None:
            kwargs[key] = ctx[key]

    min_dollar_vol = ctx.get("min_dollar_vol")
    if "min_dollar_vol" in accepted and min_dollar_vol is not None:
        kwargs["min_dollar_vol"] = min_dollar_vol
    elif "min_dollar_volume" in accepted and min_dollar_vol is not None:
        kwargs["min_dollar_volume"] = min_dollar_vol
    elif "dollar_volume_min" in accepted and min_dollar_vol is not None:
        kwargs["dollar_volume_min"] = min_dollar_vol

    if "index" in accepted:
        kwargs["index"] = universe_token
    if "universe" in accepted:
        kwargs["universe"] = universe_token
    if "universe_token" in accepted:
        kwargs["universe_token"] = universe_token
    if "universe_name" in accepted:
        kwargs["universe_name"] = universe_name

    for key in ("include_ta", "apply_gap_filter", "show_diagnostics_ui"):
        if key in accepted and key in ctx:
            kwargs[key] = ctx[key]

    return target(**kwargs) if kwargs else target()


def _wrap_override(bound_fn, universe_token: str, universe_name: str):
    """Return a callable that invokes bound_fn's target with forced universe/index."""

    def _runner():
        return _call_with_overrides(bound_fn, universe_token, universe_name)

    target = getattr(bound_fn, "_target_fn", bound_fn)
    setattr(_runner, "_target_fn", target)
    setattr(
        _runner,
        "_target_name",
        f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')} "
        f"[override:{universe_token}]",
    )
    return _runner


_run_sp500 = _discover_runner(
    (
        ("scheduler.jobs", "run_sp500_now"),
        ("ai_scanner.scheduler.jobs", "run_sp500_now"),
    )
) or _fallback_sp500
_run_nasdaq = _discover_runner(
    (
        ("scheduler.jobs", "run_nasdaq_now"),
        ("ai_scanner.scheduler.jobs", "run_nasdaq_now"),
    )
) or _fallback_nasdaq
_run_premarket = _discover_runner(
    (
        ("scheduler.jobs", "run_premarket_now"),
        ("scan.pre_post", "run_premarket_headless"),
        ("ai_scanner.scheduler.jobs", "run_premarket_now"),
        ("ai_scanner.scan.pre_post", "run_premarket_headless"),
    )
) or _fallback_premarket
_run_postmarket = _discover_runner(
    (
        ("scheduler.jobs", "run_postmarket_now"),
        ("scan.pre_post", "run_postmarket_headless"),
        ("ai_scanner.scheduler.jobs", "run_postmarket_now"),
        ("ai_scanner.scan.pre_post", "run_postmarket_headless"),
    )
) or _fallback_postmarket

run_sp500 = _bind_session_args(_run_sp500)
run_nasdaq = _bind_session_args(_run_nasdaq)
run_premarket = _bind_session_args(_run_premarket)
run_postmarket = _bind_session_args(_run_postmarket)
run_sp500_button = _wrap_override(run_sp500, universe_token="SP500", universe_name="S&P 500")
run_nasdaq_button = _wrap_override(run_nasdaq, universe_token="NASDAQ100", universe_name="Nasdaq 100")


__all__ = [
    "run_sp500",
    "run_nasdaq",
    "run_premarket",
    "run_postmarket",
    "run_sp500_button",
    "run_nasdaq_button",
]
