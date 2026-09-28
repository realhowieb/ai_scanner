"""P1-41 — unsubscribe from HSF emails via the link in every email.

Works without signing in: the link's token identifies the account. It never
acts on page load (email scanners open links); the person clicks a button.
"""
from __future__ import annotations

import streamlit as st


def _save(ok: bool) -> None:
    """Redraw with the saved state (so the button disappears), or say it failed."""
    if ok:
        st.rerun()
    st.error("Couldn't save that right now. Please try again in a minute.")


def main() -> None:
    st.set_page_config(page_title="Email preferences | HSFinest.AI", page_icon="✉️")
    from ui.chrome import hide_developer_chrome  # noqa: E402

    hide_developer_chrome()  # Run 62/P2: before anything else renders
    try:
        from ui.header import render_page_logo

        render_page_logo()
    except Exception:
        pass
    from ui.design_system import render_page_header

    render_page_header("Email preferences", "Choose which HSF emails you get.")

    token = str(st.query_params.get("t", "") or "").strip()
    kind = str(st.query_params.get("k", "") or "").strip().lower()
    try:
        from db.email_prefs import KINDS, LABELS, get_prefs, set_prefs, user_for_token
        from ui.log_privacy import mask_email

        user = user_for_token(token) if token else None
    except Exception:
        user = None
    if not user:
        st.error("This unsubscribe link isn't valid. Sign in and open **Settings** to change your emails.")
        st.page_link("app.py", label="Go to sign in")
        return

    st.markdown(f"Emails for `{mask_email(user)}`")  # code style: the *** mask would break bold
    prefs = get_prefs(user)
    if not any(prefs.values()):
        st.success("You're unsubscribed from all HSF emails. In-app alerts still work.")
    elif kind in KINDS and not prefs.get(kind, True):
        st.success(f"You're unsubscribed from the {LABELS[kind].lower()}.")
    elif kind in KINDS and st.button(f"Unsubscribe from the {LABELS[kind].lower()}", key="unsub_one", type="primary"):
        _save(set_prefs(user, **{kind: False}))
    if any(prefs.values()) and st.button("Unsubscribe from all HSF emails", key="unsub_all"):
        _save(set_prefs(user, **{k: False for k in KINDS}))
    st.caption(" · ".join(f"{LABELS[k]}: {'on' if prefs[k] else 'off'}" for k in KINDS))
    st.caption("Changed your mind? Sign in and turn emails back on in **Settings**. "
               "Account emails like password resets always go out.")
    st.page_link("app.py", label="Go to HSF")


main()
