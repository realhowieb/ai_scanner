"""Settings > Delete account (P2-82). Same rules and code as DELETE /v1/me
(db.account_deletion); the password check is the API's bcrypt check, which
matches the web sign-in. Never raises."""
from __future__ import annotations

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def render_delete_account(username: str) -> None:
    if st is None or not username:
        return
    try:
        with st.expander("Delete account", expanded=False):
            st.caption("Permanently deletes your account and its data: watchlists, alerts, journal, "
                       "paper-trading connection, settings and saved scans. This can't be undone. "
                       "If you have a paid plan, cancel it on the Billing page first.")
            password = st.text_input("Password", type="password", key="del_acct_pw")
            typed = st.text_input('Type DELETE to confirm', key="del_acct_confirm")
            if not st.button("Delete my account", key="del_acct_btn", type="primary",
                             disabled=typed.strip() != "DELETE" or not password):
                return
            from api.store import _conn, check_password, get_account
            from db.account_deletion import DeletionBlocked, delete_account

            account = get_account(username)
            if not account or not check_password(account, password):
                st.error("Your password is incorrect.")
                return
            conn = _conn()
            try:
                delete_account(conn, username)
            except DeletionBlocked as e:
                st.warning(str(e))
                return
            finally:
                conn.close()
        from ui.auth import logout_and_reset_session

        st.success("Your account has been deleted.")
        logout_and_reset_session()
    except Exception as exc:
        if any(c.__name__ == "ScriptControlException" for c in type(exc).__mro__):
            raise
        from ui.safe_errors import show_error

        show_error("account deletion", exc)
