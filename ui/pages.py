from __future__ import annotations

# UI loader that exposes `render_app()` for app.py
# Tries to import a concrete page renderer from `ui/pages_main.py`
# and falls back to a simple stub if not available.
import importlib
import os
import sys
import traceback

import streamlit as st


def render_app() -> None:
    """Entry point used by app.py to render the UI."""
    # Ensure project root is on sys.path so local packages like `ui/` are importable
    app_root = os.path.dirname(os.path.abspath(__file__))
    if app_root not in sys.path:
        sys.path.insert(0, app_root)

    candidates = (
        "ui.pages_main",                 # local package (ui/__init__.py present)
        "ai_scanner.ui.pages_main",      # installed/namespace package style
        "pages_main",                    # flat module next to this file
    )
    for candidate in candidates:
        try:
            mod = importlib.import_module(candidate)
            render_fn = getattr(mod, "render", None)
            if callable(render_fn):
                return render_fn()
            else:
                st.warning(f"Module `{candidate}` found but has no callable `render()`.")
        except Exception as e:
            # Show a concise error in the UI; full traceback if needed.
            st.error(f"Failed to load `{candidate}`: {e}")
            st.caption("Traceback:")
            st.code(''.join(traceback.format_exc()))

    # Fallback stub UI
    st.title("AI Scanner")
    st.info(
        "UI stub loaded: create `ui/pages_main.py` with a function `render()` "
        "to customize the layout."
    )
    st.caption(
        "Tip: Inside `pages_main.py`, define `def render(): ...` and build your tabs/pages there."
    )