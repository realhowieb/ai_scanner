"""Admin Users Management UI."""

import streamlit as st

from db.engine import get_neon_conn
from db.schema import ensure_neon_users_schema
from db.users import fetch_all_users, load_users
from ui.arrow_safe import arrow_safe


def render_admin_users_panel(username, ADMIN_USERS, db_status):
    """Render the full admin panel UI."""
    # Admin by the ADMIN_USERS secret or by the database flag resolved at sign-in
    # (P1-38): renaming an account to its email must not hide the panel.
    if username not in ADMIN_USERS and not bool(st.session_state.get("is_admin")):
        return

    with st.expander("👑 Admin: Manage Users", expanded=False):
        enable_admin = st.checkbox(
            "Enable admin user management",
            value=False,
            key="enable_admin_users",
        )

        if not enable_admin:
            st.caption("Toggle the switch above to load and manage Neon users.")
            return

        _render_create_user()

        # --- Manage Existing Users ---
        users_df = fetch_all_users()
        if users_df is None or users_df.empty:
            st.caption("No users found in Neon users table.")
            return

        st.caption("View and edit user tiers. Changes apply to Neon-backed accounts.")

        status = _account_status()
        if status is not None and not status.empty:
            st.dataframe(arrow_safe(status), width="stretch", height=260)
        else:
            desired_cols = ["id", "username", "full_name", "tier", "is_active", "created_at"]
            display_cols = [c for c in desired_cols if c in users_df.columns]
            st.dataframe(arrow_safe(users_df[display_cols]), width="stretch", height=260)

        usernames_list = users_df["username"].tolist()
        selected_user = st.selectbox("Select user to edit", usernames_list)
        row = users_df[users_df["username"] == selected_user].iloc[0]

        # Keep the stored tier (including "admin") selected so "Update User"
        # never silently downgrades an account nobody meant to change.
        current_tier = str(row["tier"] or "basic").strip().lower()
        tier_options = ["basic", "pro", "premium"] + (
            [current_tier] if current_tier not in ("basic", "pro", "premium") else []
        )
        new_tier = st.selectbox("Tier", tier_options, index=tier_options.index(current_tier))
        new_active = st.checkbox("Active", value=bool(row["is_active"]))

        if st.button("Update User"):
            try:
                conn = get_neon_conn()
                if conn is None:
                    st.error("Neon connection unavailable; cannot update user.")
                else:
                    ensure_neon_users_schema(conn)
                    cur = conn.cursor()
                    cur.execute(
                        """
                        UPDATE users
                        SET tier = %s,
                            is_active = %s
                        WHERE username = %s
                        """,
                        (new_tier, new_active, selected_user),
                    )
                    conn.commit()
                    cur.close()
                    conn.close()

                    try:
                        load_users.clear()  # type: ignore
                    except Exception:
                        pass

                    st.success(f"User '{selected_user}' updated successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to update user: {e}")

        _render_account_actions(selected_user, status, acting_user=username)


def _render_create_user() -> None:
    """P1-37: the username must be a valid email (stored lowercase); the
    password is bcrypt-hashed; an existing account is reported, not overwritten."""
    st.subheader("➕ Create New User")
    new_username = st.text_input("Email address (username)", key="create_user_email",
                                 help="Every HSF email (verification, reset, digest, alerts) goes here.")
    new_full_name = st.text_input("Full Name", key="create_user_name")
    new_password = st.text_input("Password", type="password", key="create_user_password")
    new_tier_create = st.selectbox("Tier", ["basic", "pro", "premium"], key="create_user_tier")
    new_active_create = st.checkbox("Active", value=True, key="create_user_active")
    if st.button("Create User"):
        try:
            from db.user_admin import create_user
        except ImportError:
            st.error("Account creation is updating; try again in a moment.")
            return
        ok, msg = create_user(new_username, new_full_name, new_password,
                              tier=new_tier_create, active=new_active_create)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)


def _account_status():
    """Accounts with admin and verification columns, or None if unavailable."""
    try:
        import pandas as pd

        from db.user_admin import fetch_users_for_admin

        rows = fetch_users_for_admin()
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def _render_account_actions(selected_user, status, *, acting_user) -> None:
    """P1-38: verification and admin status for the selected account."""
    try:
        from db.user_admin import is_valid_email, mark_email_verified, set_admin
    except ImportError:
        return
    info = {}
    if status is not None and not status.empty:
        match = status[status["username"] == selected_user]
        if not match.empty:
            info = match.iloc[0].to_dict()
    verified, admin = bool(info.get("email_verified")), bool(info.get("is_admin"))
    st.markdown(f"**{selected_user}** · email {'verified ✅' if verified else 'not verified'}"
                f" · {'admin 👑' if admin else 'not admin'}")
    if not is_valid_email(selected_user):
        st.caption("This username isn't an email address, so the account can't receive email. "
                   "Rename it to the person's email (P2-28) before verifying.")
    else:
        c1, c2 = st.columns(2)
        if not verified and c1.button("Mark email verified", key="admin_mark_verified"):
            ok, msg = mark_email_verified(selected_user)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()
        if not verified and c2.button("Resend verification email", key="admin_resend_verify"):
            try:
                from ui.email_verification_gate import _resend_verification

                sent = _resend_verification(selected_user)
            except Exception:
                sent = False
            if sent:
                st.success(f"Verification email sent to {selected_user}.")
            else:
                st.error("Couldn't send the email. Check the email setup (Resend test mode only "
                         "delivers to the Resend account owner).")
    confirm = st.checkbox(f"Confirm: {'remove admin role from' if admin else 'make'} {selected_user}"
                          f"{'' if admin else ' an admin'}", key=f"admin_role_confirm_{selected_user}")
    if st.button("Revoke admin" if admin else "Grant admin", key="admin_role_btn", disabled=not confirm):
        ok, msg = set_admin(selected_user, not admin, acting_user=acting_user)
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()
