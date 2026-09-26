"""Run 68 — load secondary panels only when the user asks for them.

Streamlit executes the contents of every tab on every rerun, so a closed
"Historical research" or "Scan History" tab still ran its database reads and
model work each time anything on the Scanner changed. A panel wrapped in
`lazy_open` shows a switch instead and does its work only once switched on;
the choice is remembered for the session.
"""
from __future__ import annotations

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def lazy_open(key: str, label: str, *, help: str | None = None) -> bool:
    """Render a 'Load …' switch; True when the panel should render. Never raises."""
    if st is None:
        return True
    try:
        return bool(st.toggle(label, key=f"hsf_lazy_{key}", value=False, help=help))
    except Exception:
        return True
