"""Acquisition attribution and funnel events.

Best-effort and privacy-light: no raw email addresses, names, passwords, or IPs.
Events are useful for the first external traffic batch, but must never block the
app if the database is missing or unavailable.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from db.engine import schema_once

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

EVENTS_TABLE = "acquisition_events"
SESSION_ATTR_KEY = "hsf_acquisition_attribution"
LANDING_VISIT_KEY = "hsf_acq_landing_visit_tracked"
AUTH_SESSION_KEY = "hsf_acq_auth_session_tracked_for"
SCANNER_VIEW_KEY = "hsf_acq_scanner_view_tracked_for"

ALLOWED_EVENTS = {
    "landing_visit",
    "primary_cta_click",
    "signup_started",
    "signup_completed",
    "first_authenticated_session",
    "return_session",
    "first_scanner_view",
    "first_scanner_use",
    "upgrade_started",
    "successful_paid_conversion",
}
ALLOWED_UTM_KEYS = ("utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term")


def _clean(value: object, *, max_len: int = 120) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.:/?&=%+-]", "", text)
    return text[:max_len]


def _first(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0] if value else "")
    return str(value or "")


def _query_params() -> dict[str, str]:
    if st is None:
        return {}
    try:
        qp = getattr(st, "query_params", {}) or {}
        return {str(k): _first(v) for k, v in dict(qp).items()}
    except Exception:
        return {}


def classify_source(params: Mapping[str, object] | None, referrer: str | None = None) -> str:
    """Classify acquisition source into reddit/direct/other_unknown."""
    params = params or {}
    src = _first(params.get("utm_source")).strip().lower()
    medium = _first(params.get("utm_medium")).strip().lower()
    ref = (referrer or _first(params.get("ref")) or _first(params.get("referrer"))).strip().lower()
    if src == "reddit" or "reddit.com" in ref:
        return "reddit"
    if src or medium or ref:
        return "other_unknown"
    return "direct"


def attribution_from_params(params: Mapping[str, object] | None, referrer: str | None = None) -> dict[str, str]:
    params = params or {}
    out = {"source": classify_source(params, referrer)}
    for key in ALLOWED_UTM_KEYS:
        value = _clean(_first(params.get(key)))
        if value:
            out[key] = value
    ref = _clean(referrer or _first(params.get("referrer")) or _first(params.get("ref")), max_len=180)
    if ref:
        try:
            out["referrer_domain"] = _clean(urlparse(ref).netloc or ref, max_len=120)
        except Exception:
            out["referrer_domain"] = ref[:120]
    return out


def capture_attribution(params: Mapping[str, object] | None = None) -> dict[str, str]:
    """Persist attribution in session_state and return it."""
    attr = attribution_from_params(params if params is not None else _query_params())
    if st is not None:
        try:
            st.session_state.setdefault(SESSION_ATTR_KEY, attr)
            # Fill in UTM keys discovered later without replacing the first source.
            current = dict(st.session_state.get(SESSION_ATTR_KEY) or {})
            for key, value in attr.items():
                current.setdefault(key, value)
            st.session_state[SESSION_ATTR_KEY] = current
            return current
        except Exception:
            pass
    return attr


def current_attribution() -> dict[str, str]:
    if st is not None:
        try:
            return dict(st.session_state.get(SESSION_ATTR_KEY) or capture_attribution())
        except Exception:
            pass
    return capture_attribution({})


def user_hash(username: object | None) -> str | None:
    user = str(username or "").strip().lower()
    if not user:
        return None
    return hashlib.sha256(user.encode("utf-8")).hexdigest()[:24]


@schema_once
def _ensure_schema(conn: Any) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS acquisition_events (
            id SERIAL PRIMARY KEY,
            event_name TEXT NOT NULL,
            source TEXT DEFAULT 'direct',
            utm_source TEXT,
            utm_medium TEXT,
            utm_campaign TEXT,
            utm_content TEXT,
            utm_term TEXT,
            referrer_domain TEXT,
            user_hash TEXT,
            plan TEXT,
            metadata JSONB DEFAULT '{}'::jsonb,
            occurred_at TIMESTAMPTZ DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_acquisition_events_name_time "
        "ON acquisition_events (event_name, occurred_at DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_acquisition_events_source_time "
        "ON acquisition_events (source, occurred_at DESC)"
    )
    conn.commit()
    cur.close()


def track_event(
    event_name: str,
    *,
    username: object | None = None,
    plan: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> bool:
    """Record an acquisition event. Returns True only when persisted."""
    name = str(event_name or "").strip()
    if name not in ALLOWED_EVENTS:
        return False
    attr = current_attribution()
    clean_meta = {
        _clean(k, max_len=60): _clean(v, max_len=160)
        for k, v in dict(metadata or {}).items()
        if _clean(k, max_len=60)
    }
    try:
        from db.engine import get_neon_conn

        conn = get_neon_conn()
        if conn is None:
            return False
        _ensure_schema(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO acquisition_events
                (event_name, source, utm_source, utm_medium, utm_campaign, utm_content,
                 utm_term, referrer_domain, user_hash, plan, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                name,
                attr.get("source") or "direct",
                attr.get("utm_source"),
                attr.get("utm_medium"),
                attr.get("utm_campaign"),
                attr.get("utm_content"),
                attr.get("utm_term"),
                attr.get("referrer_domain"),
                user_hash(username),
                _clean(plan, max_len=40) if plan else None,
                clean_meta,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


def track_landing_visit_once() -> None:
    if st is None:
        return
    try:
        capture_attribution()
        if not st.session_state.get(LANDING_VISIT_KEY):
            st.session_state[LANDING_VISIT_KEY] = True
            track_event("landing_visit")
    except Exception:
        pass


def track_authenticated_session_once(username: object, plan: object | None = None) -> None:
    if st is None:
        return
    try:
        key = str(username or "").strip().lower()
        if not key:
            return
        prior = st.session_state.get(AUTH_SESSION_KEY)
        if prior == key:
            return
        st.session_state[AUTH_SESSION_KEY] = key
        event = "return_session" if st.session_state.pop("hsf_restored_session", False) else "first_authenticated_session"
        track_event(event, username=username, plan=plan)
    except Exception:
        pass


def track_scanner_view_once(username: object, plan: object | None = None) -> None:
    if st is None:
        return
    try:
        key = str(username or "").strip().lower()
        if key and st.session_state.get(SCANNER_VIEW_KEY) != key:
            st.session_state[SCANNER_VIEW_KEY] = key
            track_event("first_scanner_view", username=username, plan=plan)
    except Exception:
        pass
