"""Run 62 — hide Streamlit developer chrome that reads as a demo, not a product.

`.streamlit/config.toml` already sets `toolbarMode = "minimal"`, but Streamlit
Community Cloud still injects its own toolbar actions ("Fork" and the GitHub
link) for public-repository apps. Those render inside the app frame in
`[data-testid="stToolbarActions"]`, which holds nothing else, so hiding exactly
that container is safe: the sidebar toggle, page navigation and Streamlit
Cloud's owner tools ("Manage app") live elsewhere and are unaffected.

The source is still public on GitHub; this only removes the in-app shortcut.
"""
from __future__ import annotations

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

CHROME_CSS = (
    "<style>"
    '[data-testid="stToolbarActions"]{display:none !important;}'
    "</style>"
)


def hide_developer_chrome() -> None:
    """Inject the CSS once per run. Never raises."""
    if st is None:
        return
    try:
        st.markdown(CHROME_CSS, unsafe_allow_html=True)
    except Exception:
        pass
