"""Admin Console — dedicated, authorization-gated operational workspace."""
from __future__ import annotations

import streamlit as st

from ui.showcase import initial_sidebar_state

st.set_page_config(
    page_title="Admin Console — HSF",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state=initial_sidebar_state(),
)

from ui.chrome import hide_developer_chrome  # noqa: E402

hide_developer_chrome()

username = str(st.session_state.get("username") or "").strip().lower()
if not username:
    st.info("Please log in on the main page to open the Admin Console.")
    st.page_link("app.py", label="Go to login", icon="🔐")
    st.stop()

try:
    from ui.nav import render_sidebar_nav

    render_sidebar_nav()
except Exception:
    pass

if not bool(st.session_state.get("is_admin")):
    st.error("Admin Console is only available to administrators.")
    st.page_link("app.py", label="Return to Scanner", icon="🔎")
    st.stop()

from db.core import get_conn  # noqa: E402
from ui.admin_results_tab import render_admin_page  # noqa: E402
from ui.admin_users import render_admin_users_panel  # noqa: E402
from ui.db_status import render_db_status_badge  # noqa: E402

render_admin_page(
    username=username,
    db_status=render_db_status_badge(show_badge=False),
    admin_users=(),
    render_admin_users_panel=render_admin_users_panel,
    get_db_conn=get_conn,
)
