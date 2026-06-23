"""Watchlist actions shown inside result detail panels."""
from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

import streamlit as st


WatchlistGetter = Callable[[object, str], list[str]]
WatchlistSetter = Callable[[object, str, list[str]], Any]


def _normalize_symbols(symbols: object) -> set[str]:
    return {str(symbol).strip().upper() for symbol in (symbols or []) if str(symbol).strip()}


def _resolve_watchlist_fns() -> tuple[WatchlistGetter | None, WatchlistSetter | None]:
    candidates = (
        ("db.watchlists", "get_watchlist_tickers", "set_watchlist_tickers"),
        ("db.watchlist", "get_watchlist_tickers", "set_watchlist_tickers"),
        ("watchlists", "get_watchlist_tickers", "set_watchlist_tickers"),
        ("watchlist", "get_watchlist_tickers", "set_watchlist_tickers"),
    )
    for mod_name, get_name, set_name in candidates:
        try:
            mod = import_module(mod_name)
        except ImportError:
            continue

        get_fn = getattr(mod, get_name, None)
        set_fn = getattr(mod, set_name, None)
        if callable(get_fn) and callable(set_fn):
            return get_fn, set_fn

    return None, None


def render_watchlist_action(ticker: str, *, key_prefix: str = "results") -> None:
    """Render and handle adding a result ticker to the active watchlist."""
    normalized_ticker = (ticker or "").strip().upper()
    if not normalized_ticker:
        return

    active_id = st.session_state.get("active_watchlist_id")
    username = st.session_state.get("username") or st.session_state.get("user") or "anonymous"
    current_norm = _normalize_symbols(st.session_state.get("active_watchlist_tickers", []))

    if active_id is None:
        st.caption("📋 Watchlist: select an active watchlist to add tickers.")
        return

    already = normalized_ticker in current_norm

    action_col, caption_col = st.columns([1, 2])
    with action_col:
        clicked = st.button(
            "⭐ Add to Watchlist" if not already else "✅ In Watchlist",
            key=f"{key_prefix}_btn_details_add_watchlist_{normalized_ticker}",
            disabled=already,
            width="stretch",
        )
    with caption_col:
        st.caption("Adds this ticker to your active watchlist.")

    if not clicked:
        return

    get_fn, set_fn = _resolve_watchlist_fns()
    updated_norm = set(current_norm)
    updated_norm.add(normalized_ticker)
    st.session_state["active_watchlist_tickers"] = sorted(updated_norm)

    if not callable(get_fn) or not callable(set_fn):
        st.info(
            "Added locally (no DB watchlist functions found). "
            "If it disappears after refresh, wire get_watchlist_tickers/set_watchlist_tickers into this module."
        )
        return

    try:
        existing_db_norm = _normalize_symbols(get_fn(active_id, username) or [])
        new_db = sorted(existing_db_norm | {normalized_ticker})
        set_fn(active_id, username, list(new_db))

        verify_norm = _normalize_symbols(get_fn(active_id, username) or [])
        st.session_state["active_watchlist_tickers"] = sorted(verify_norm)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        st.warning(
            f"Added **{normalized_ticker}** locally, but DB save failed: {exc}. "
            "It may disappear after refresh if DB cannot be reached."
        )
        return

    if normalized_ticker in verify_norm:
        st.success(f"Added **{normalized_ticker}** to your active watchlist.")
    else:
        st.warning(
            f"Tried to add **{normalized_ticker}**, but it did not appear in the DB on verification. "
            "Your DB write may be failing or the watchlist is keyed differently."
        )
