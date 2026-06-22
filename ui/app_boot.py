"""Application boot helpers for Streamlit compatibility and noisy dependency output."""
from __future__ import annotations

import contextlib
import io
import sys
from collections.abc import Iterator

import streamlit as st


def patch_use_container_width() -> None:
    """Map deprecated use_container_width calls to Streamlit's width argument."""

    def _wrap(fn):
        if getattr(fn, "_ucw_patched", False):
            return fn

        def _inner(*args, **kwargs):
            if "use_container_width" in kwargs:
                ucw = kwargs.pop("use_container_width")
                if "width" not in kwargs:
                    kwargs["width"] = "stretch" if bool(ucw) else "content"
            return fn(*args, **kwargs)

        setattr(_inner, "_ucw_patched", True)
        return _inner

    chart_api_names = (
        "dataframe",
        "table",
        "data_editor",
        "plotly_chart",
        "altair_chart",
        "pyplot",
        "line_chart",
        "bar_chart",
        "area_chart",
        "scatter_chart",
    )
    for name in chart_api_names:
        try:
            fn = getattr(st, name, None)
            if callable(fn):
                setattr(st, name, _wrap(fn))
        except Exception:
            pass

    try:
        from streamlit.delta_generator import DeltaGenerator  # type: ignore

        for name in chart_api_names:
            try:
                meth = getattr(DeltaGenerator, name, None)
                if callable(meth):
                    setattr(DeltaGenerator, name, _wrap(meth))
            except Exception:
                pass
    except Exception:
        pass


class FilteredStderr(io.TextIOBase):
    """Drop known noisy yfinance messages from stderr while keeping real errors visible."""

    def __init__(self, underlying: io.TextIOBase):
        self._underlying = underlying

    def write(self, s: str) -> int:
        try:
            text = str(s)
        except Exception:
            return 0

        noisy_patterns = (
            "HTTP Error 401",
            "Invalid Crumb",
            "quoteSummary",
            "No fundamentals data found",
            "HTTP Error 404",
            "No earnings dates found, symbol may be delisted",
        )
        if any(pattern in text for pattern in noisy_patterns):
            return len(text)

        try:
            return self._underlying.write(text)
        except Exception:
            return len(text)

    def flush(self) -> None:
        try:
            self._underlying.flush()
        except Exception:
            pass


def install_stderr_filter() -> None:
    """Install the global stderr filter once."""
    try:
        if not isinstance(sys.stderr, FilteredStderr):
            sys.stderr = FilteredStderr(sys.stderr)  # type: ignore[assignment]
    except Exception:
        pass


@contextlib.contextmanager
def quiet_external_calls() -> Iterator[None]:
    """Silence stdout/stderr for noisy third-party libraries."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


def install_streamlit_compat() -> None:
    """Install Streamlit compatibility shims and warning suppression."""
    install_stderr_filter()
    patch_use_container_width()
    try:
        st.cache = st.cache_data  # type: ignore[attr-defined]
    except Exception:
        pass


def configure_page() -> None:
    """Apply Streamlit page config for the scanner app."""
    st.set_page_config(
        page_title="Breakout Stock Scanner",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
