"""Sign-in helpers kept out of ui/auth.py (size-capped): P2-20 and P2-40."""
from __future__ import annotations

from ui.auth_sessions import COOKIE_NAME


def clear_session_cookie(cookies) -> None:
    """Blank the session cookie (P2-20).

    streamlit-cookies-manager's delete is a no-op when a prefix is set (it looks
    for the unprefixed name among prefixed cookies), so pop() never removed the
    cookie and every later visit said "Your session has expired". Overwriting it
    with an empty value goes through the working set path; an empty sid is
    treated as no session.
    """
    cookies[COOKIE_NAME] = ""


def deactivated_with_password(login_key: str, candidates) -> bool:
    """P2-40: True only for a deactivated account AND the correct password."""
    try:
        from db.inactive_login import inactive_password, password_matches

        return password_matches(inactive_password(login_key), candidates)
    except Exception:
        return False
