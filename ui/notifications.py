"""In-app alert notification line: '🔔 N alerts fired today'.

Surfaces fired alerts to users who don't read email — a glanceable count at
the top of the page linking attention to the Alerts panel below. Collapses
duplicate messages the same way the triggered feed does. Never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st


@st.cache_data(ttl=60, show_spinner=False)
def _recent_events(user_id: str):
    from db.alerts import list_recent_events

    return list_recent_events(user_id, limit=25) or []


def render_alert_bell(user_id: str) -> None:
    if not user_id:
        return
    try:
        events = _recent_events(user_id)
    except Exception:
        return
    today = datetime.now(timezone.utc).date()
    seen = set()
    for ev in events:
        fired = ev.get("fired_at")
        if fired is None or not hasattr(fired, "date") or fired.date() != today:
            continue
        seen.add(str(ev.get("message") or "")[:120])
    if not seen:
        return
    n = len(seen)
    st.caption(f"🔔 **{n} alert{'s' if n != 1 else ''} fired today** — details in the Alerts section below.")
