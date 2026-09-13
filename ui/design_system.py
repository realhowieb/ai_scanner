"""Small shared presentation primitives for HSF product cohesion."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


STATUS_LABELS = {"STRONG", "WATCH", "CAUTION", "FADING"}
EVENT_LABELS = {
    "STATUS_UPGRADE": "Status upgraded",
    "STATUS_DOWNGRADE": "Status downgraded",
    "NEW_OPPORTUNITY": "New opportunity",
    "DROPPED": "Left ranking",
    "FADING": "Fading",
    "RISING": "Rising",
    "FALLING": "Falling",
    "SIGNAL_ADDED": "Signal added",
    "SIGNAL_REMOVED": "Signal removed",
    "VERSION_CHANGED": "Version changed",
}


def ticker_label(value: object) -> str:
    return str(value or "").strip().upper()


def status_label(value: object) -> str:
    label = str(value or "").strip().upper()
    return label if label in STATUS_LABELS else (label or "UNRANKED")


def event_label(value: object) -> str:
    raw = str(value or "").strip().upper()
    return EVENT_LABELS.get(raw, raw.replace("_", " ").title() if raw else "No recent change")


def hsf_score_value(value: object) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "--"


def hsf_score_line(score: object, status: object = None, *, compact: bool = True) -> str:
    score_text = hsf_score_value(score)
    status_text = status_label(status) if status else ""
    if compact:
        return f"HSF Score {score_text}" + (f" · {status_text}" if status_text else "")
    return f"HSF Score\n{score_text}" + (f"\n{status_text}" if status_text else "")


def freshness_label(value: Any, *, stale_after_minutes: int = 360) -> str:
    if value is None:
        return "Updated time unavailable"
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        mins = max(0, int((datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds() // 60))
    except Exception:
        return f"Updated {value}"
    if mins < 90:
        text = f"Updated {mins}m ago"
    elif mins < 36 * 60:
        text = f"Updated {mins // 60}h ago"
    else:
        text = "Updated yesterday" if mins < 60 * 60 else f"Updated {mins // 1440}d ago"
    return text + (" · Stale" if mins >= stale_after_minutes else "")


def render_page_header(title: str, subtitle: str) -> None:
    if st is None:
        return
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def render_empty_state(title: str, body: str, *, action_page: Optional[str] = None, action_label: str = "") -> None:
    if st is None:
        return
    st.info(title)
    st.caption(body)
    if action_page and action_label:
        st.page_link(action_page, label=action_label)
