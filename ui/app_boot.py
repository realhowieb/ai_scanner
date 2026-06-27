"""Application boot helpers for Streamlit compatibility and noisy dependency output."""
from __future__ import annotations

import contextlib
import io
import logging
import sys
import warnings
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
            "`st.cache` is deprecated",
            "Please use one of Streamlit's new",
            "The behavior of `st.cache` was updated",
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


def install_warning_filters() -> None:
    """Suppress known third-party warning spam without hiding app warnings."""
    warnings.filterwarnings(
        "ignore",
        message=r"The 'generic' unit for NumPy timedelta is deprecated.*",
        category=DeprecationWarning,
        module=r"yfinance\.utils",
    )


class _DropStCacheDeprecation(logging.Filter):
    """Drop the st.cache deprecation log emitted by older third-party libs.

    Streamlit logs this via the logging module (not stderr), so the stderr
    filter can't catch it. Our own code uses st.cache_data/st.cache_resource;
    the call comes from a dependency (e.g. streamlit-authenticator).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "st.cache` is deprecated" not in msg and "`st.cache`" not in msg


def install_logging_filters() -> None:
    """Suppress the st.cache deprecation log emitted by older third-party libs.

    A logging.Filter on a *logger* only filters records logged directly through
    it — not records propagated up from child loggers. So we attach the filter
    to the root logger's *handlers* (where every propagated record is actually
    written) as well as to the candidate Streamlit loggers, to catch it
    regardless of which logger emits it.
    """
    flt = _DropStCacheDeprecation()
    # Direct loggers (best-effort).
    for name in ("streamlit", "streamlit.runtime.caching", "streamlit.deprecation_util", ""):
        try:
            logging.getLogger(name).addFilter(flt)
        except Exception:
            pass
    # Root handlers — this is the reliable catch-all for propagated records.
    try:
        for handler in logging.getLogger().handlers:
            handler.addFilter(flt)
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
    install_warning_filters()
    install_logging_filters()
    install_stderr_filter()
    patch_use_container_width()


def configure_page() -> None:
    """Apply Streamlit page config for the scanner app."""
    st.set_page_config(
        page_title="Breakout Stock Scanner",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
