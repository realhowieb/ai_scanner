"""Universe resolution helpers for scanner flows.

This module keeps Streamlit state mechanics out of the selection rules so SP500,
NASDAQ, and Combo behavior can be tested without running the UI.
"""
from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from typing import Any

from .options import normalize_market


SymbolLoader = Callable[[], Sequence[str]]
SymbolTransform = Callable[[Sequence[str]], list[str]]
SafeCall = Callable[..., Any]


def _state_bool(state: MutableMapping[str, Any], key: str) -> bool | None:
    value = state.get(key)
    if value is None:
        return None
    return bool(value)


def _state_int(state: MutableMapping[str, Any], key: str) -> int | None:
    value = state.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _state_list(state: MutableMapping[str, Any], key: str) -> list[str]:
    value = state.get(key)
    if not value:
        return []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def _load_filtered_universe(
    *,
    label: str,
    loader: SymbolLoader,
    safe_call: SafeCall,
    filter_universe: SymbolTransform,
    sanitize_symbols: SymbolTransform,
) -> list[str]:
    base = safe_call(loader, label=label)
    filtered = filter_universe(base or [])
    return sanitize_symbols(filtered)


def resolve_scan_universe(
    market: object,
    state: MutableMapping[str, Any],
    *,
    is_admin: bool,
    safe_call: SafeCall,
    load_sp500_universe: SymbolLoader,
    load_nasdaq_universe: SymbolLoader,
    filter_universe: SymbolTransform,
    sanitize_symbols: SymbolTransform,
) -> list[str]:
    """Resolve the selected three-step scanner universe and update state caches."""
    market_name = normalize_market(market)

    sp500 = _state_list(state, "sp500_universe")
    nasdaq = _state_list(state, "nasdaq_universe")
    nasdaq_capped = _state_list(state, "nasdaq_capped")
    combo_capped = _state_list(state, "combo_capped")

    def ensure_sp500() -> list[str]:
        nonlocal sp500
        if not sp500:
            sp500 = _load_filtered_universe(
                label="SP500 universe (3-step)",
                loader=load_sp500_universe,
                safe_call=safe_call,
                filter_universe=filter_universe,
                sanitize_symbols=sanitize_symbols,
            )
            state["sp500_universe"] = sp500
        return sp500

    def ensure_nasdaq() -> list[str]:
        nonlocal nasdaq, nasdaq_capped
        if not nasdaq:
            nasdaq = _load_filtered_universe(
                label="NASDAQ universe (3-step)",
                loader=load_nasdaq_universe,
                safe_call=safe_call,
                filter_universe=filter_universe,
                sanitize_symbols=sanitize_symbols,
            )
            state["nasdaq_universe"] = nasdaq
        configured_limit = int(state.get("max_nasdaq_scan", 2000))
        effective_limit = max(configured_limit, len(nasdaq)) if is_admin else configured_limit
        cached_limit = _state_int(state, "nasdaq_capped_limit")
        cached_admin = _state_bool(state, "nasdaq_capped_admin")
        if not nasdaq_capped or cached_limit != effective_limit or cached_admin != is_admin:
            nasdaq_capped = nasdaq[:effective_limit]
            state["nasdaq_capped"] = nasdaq_capped
            state["nasdaq_capped_limit"] = effective_limit
            state["nasdaq_capped_admin"] = is_admin
        return nasdaq_capped

    def ensure_combo() -> list[str]:
        nonlocal combo_capped
        base_sp = ensure_sp500()
        base_nq = ensure_nasdaq()
        universe = list(base_sp or []) + list(base_nq or [])
        configured_limit = int(state.get("max_combo_scan", 4000))
        effective_limit = max(configured_limit, len(universe)) if is_admin else configured_limit
        cached_limit = _state_int(state, "combo_capped_limit")
        cached_admin = _state_bool(state, "combo_capped_admin")
        cached_source_count = _state_int(state, "combo_capped_source_count")
        if (
            not combo_capped
            or cached_limit != effective_limit
            or cached_admin != is_admin
            or cached_source_count != len(universe)
        ):
            combo_capped = universe[:effective_limit]
            state["combo_capped"] = combo_capped
            state["combo_capped_limit"] = effective_limit
            state["combo_capped_admin"] = is_admin
            state["combo_capped_source_count"] = len(universe)
        return combo_capped

    if market_name == "SP500":
        return ensure_sp500()
    if market_name == "NASDAQ":
        return ensure_nasdaq()
    return ensure_combo()
