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
        if not nasdaq_capped:
            max_nasdaq_scan = int(state.get("max_nasdaq_scan", 2000))
            if is_admin:
                max_nasdaq_scan = max(max_nasdaq_scan, len(nasdaq))
            nasdaq_capped = nasdaq[:max_nasdaq_scan]
            state["nasdaq_capped"] = nasdaq_capped
        return nasdaq_capped

    def ensure_combo() -> list[str]:
        nonlocal combo_capped
        if combo_capped:
            return combo_capped
        base_sp = ensure_sp500()
        base_nq = ensure_nasdaq()
        universe = list(base_sp or []) + list(base_nq or [])
        max_combo_scan = int(state.get("max_combo_scan", 4000))
        if is_admin:
            max_combo_scan = max(max_combo_scan, len(universe))
        combo_capped = universe[:max_combo_scan]
        state["combo_capped"] = combo_capped
        return combo_capped

    if market_name == "SP500":
        return ensure_sp500()
    if market_name == "NASDAQ":
        return ensure_nasdaq()
    return ensure_combo()
