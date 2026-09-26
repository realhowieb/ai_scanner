"""How HSF works — public methodology page (readable signed in or out)."""
from __future__ import annotations

import streamlit as st

from ui.product_copy import PRODUCT_NAME

st.set_page_config(page_title=f"How HSF works · {PRODUCT_NAME}", page_icon="📈", layout="centered")

try:
    from ui.chrome import hide_developer_chrome

    hide_developer_chrome()
except Exception:
    pass

from ui.methodology import render_methodology  # noqa: E402

if (st.session_state.get("username") or "").strip():
    try:
        from ui.nav import render_sidebar_nav

        render_sidebar_nav()
    except Exception:
        pass

render_methodology()

st.page_link("app.py", label="← Back to HSF", icon="🏠")
