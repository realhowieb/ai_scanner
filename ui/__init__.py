"""Shared helpers from the old Streamlit app that the API and scheduled jobs still use.

The Streamlit app itself was retired on 2026-10-21; `st` in these modules comes from
utils/st_compat.py, not from Streamlit.
"""

import importlib

__all__ = ["universe", "auth", "charts"]


def __getattr__(name: str):
    """Lazily expose submodules."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
