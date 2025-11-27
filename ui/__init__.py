"""UI package for ai_scanner.

This package contains all user-interface related modules:
    - pages:     Streamlit page definitions / page router
    - universe:  Universe loaders + filtering + liquidity helpers
    - auth:      Authentication UI
    - pricing:   Pricing / upgrade sidebar UI
    - charts:    UI-level wrappers around chart rendering (if needed)
"""

import importlib

__all__ = []


def _try_import(name: str):
    """Safely import a ui submodule if it exists."""
    try:
        module = importlib.import_module(f"{__name__}.{name}")
        __all__.append(name)
        return module
    except ImportError:
        return None


# Auto-detect and expose available UI modules
pages = _try_import("pages")
universe = _try_import("universe")
auth = _try_import("auth")
pricing = _try_import("pricing")
charts = _try_import("charts")
ai_notes = _try_import("ai_notes")
