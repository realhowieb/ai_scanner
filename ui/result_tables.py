"""Table rendering helpers for results UI."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


BASIC_RESULTS_TABLE_CSS = """
<style>
.basic-results-wrap {
  max-height: 420px;
  overflow-x: auto;
  overflow-y: auto;
  border: 1px solid rgba(49, 51, 63, 0.25);
  border-radius: 10px;
  padding: 6px;
}

/* Prevent vertical letter stacking on mobile */
.basic-results-wrap table {
  width: max-content;
  min-width: 100%;
  border-collapse: collapse;
}

.basic-results-wrap th,
.basic-results-wrap td {
  white-space: nowrap;
  padding: 6px 10px;
}

/* Sticky header */
.basic-results-wrap th {
  position: sticky;
  top: 0;
  background: rgba(15, 17, 22, 0.98);
  z-index: 2;
}
</style>
"""


def _to_static_html(table_like: Any) -> str:
    if hasattr(table_like, "hide"):
        try:
            table_like = table_like.hide(axis="index")
        except (AttributeError, TypeError, ValueError):
            pass

    if hasattr(table_like, "to_html"):
        try:
            return str(table_like.to_html(index=False))
        except TypeError:
            return str(table_like.to_html())

    raise TypeError("table object does not support to_html")


def render_static_results_table(table_like: Any, fallback_df: pd.DataFrame) -> None:
    """Render a non-interactive, horizontally scrollable results table."""
    try:
        table_html = _to_static_html(table_like)
        st.markdown(BASIC_RESULTS_TABLE_CSS, unsafe_allow_html=True)
        st.markdown(
            f"<div class='basic-results-wrap'>{table_html}</div>",
            unsafe_allow_html=True,
        )
    except (AttributeError, TypeError, ValueError):
        try:
            st.table(fallback_df)
        except Exception:
            st.markdown(fallback_df.to_html(index=False), unsafe_allow_html=True)
