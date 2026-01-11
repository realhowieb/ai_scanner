"""Admin Users Management UI."""

import streamlit as st

from db.users import fetch_all_users, load_users
from db.engine import get_neon_conn
from db.schema import ensure_neon_users_schema

try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None


def render_admin_users_panel(username, ADMIN_USERS, db_status):
    """Render the full admin panel UI."""
    if username not in ADMIN_USERS:
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

        # --- Create New User ---
        st.subheader("➕ Create New User")

        new_username = st.text_input("New Username")
        new_full_name = st.text_input("Full Name")
        new_password = st.text_input("Password", type="password")
        new_tier_create = st.selectbox("Tier", ["basic", "pro", "premium"], key="create_user_tier")
        new_active_create = st.checkbox("Active", value=True, key="create_user_active")

        if st.button("Create User"):
            if not new_username or not new_full_name or not new_password:
                st.error("All fields are required.")
            else:
                try:
                    conn = get_neon_conn()
                    if conn is None:
                        st.error("Neon connection unavailable; cannot create user.")
                    else:
                        ensure_neon_users_schema(conn)
                        cur = conn.cursor()

                        # Hash if possible
                        pwd_to_store = new_password
                        try:
                            if stauth is not None:
                                pwd_to_store = stauth.Hasher([new_password]).generate()[0]
                        except Exception:
                            pwd_to_store = new_password

                        cur.execute(
                            """
                            INSERT INTO users (username, full_name, password, tier, is_active)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (username) DO NOTHING
                            """,
                            (new_username, new_full_name, pwd_to_store, new_tier_create, new_active_create),
                        )
                        conn.commit()
                        cur.close()
                        conn.close()

                        try:
                            load_users.clear()  # type: ignore
                        except Exception:
                            pass

                        st.success(f"User '{new_username}' created successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to create user: {e}")

        # --- Manage Existing Users ---
        users_df = fetch_all_users()
        if users_df is None or users_df.empty:
            st.caption("No users found in Neon users table.")
            return

        st.caption("View and edit user tiers. Changes apply to Neon-backed accounts.")

        desired_cols = ["id", "username", "full_name", "tier", "is_active", "created_at"]
        display_cols = [c for c in desired_cols if c in users_df.columns]
        st.dataframe(users_df[display_cols], width="stretch", height=260)

        usernames_list = users_df["username"].tolist()
        selected_user = st.selectbox("Select user to edit", usernames_list)
        row = users_df[users_df["username"] == selected_user].iloc[0]

        new_tier = st.selectbox(
            "Tier",
            ["basic", "pro", "premium"],
            index=["basic", "pro", "premium"].index(
                row["tier"] if row["tier"] in ["basic", "pro", "premium"] else "basic"
            ),
        )
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