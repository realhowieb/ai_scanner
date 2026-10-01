"""Safe query-parameter routing for signed-out acquisition links."""

from __future__ import annotations

from typing import Any

SIGNUP_VIEW_KEY = "hsf_signup_view_requested"
POST_AUTH_TARGET_KEY = "hsf_post_auth_target"
ALLOWED_POST_AUTH_TARGETS = {"scanner"}


def _query_param(st: Any, name: str) -> str:
    try:
        value = st.query_params.get(name) or ""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value).strip().lower()
    except (RuntimeError, OSError, TypeError, ValueError, KeyError, AttributeError):
        return ""


def capture_entry_intent(st: Any) -> bool:
    """Persist a whitelisted landing-page auth intent across reruns."""
    if _query_param(st, "view") == "signup":
        st.session_state[SIGNUP_VIEW_KEY] = True
    target = _query_param(st, "next")
    if target in ALLOWED_POST_AUTH_TARGETS:
        st.session_state[POST_AUTH_TARGET_KEY] = target
    return bool(st.session_state.get(SIGNUP_VIEW_KEY))


def apply_post_auth_target(st: Any) -> bool:
    """Apply and consume the whitelisted destination after authentication."""
    target = str(st.session_state.pop(POST_AUTH_TARGET_KEY, "") or "").strip().lower()
    st.session_state.pop(SIGNUP_VIEW_KEY, None)
    for key in ("view", "next"):
        try:
            st.query_params.pop(key, None)
        except (RuntimeError, OSError, TypeError, ValueError, KeyError, AttributeError):
            pass
    if target == "scanner":
        st.session_state["hsf_start_scanner_after_auth"] = True
        return True
    return False
