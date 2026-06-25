"""UI package for ai_scanner.

This package contains all user-interface related modules:
    - pages:     Streamlit page definitions / page router
    - universe:  Universe loaders + filtering + liquidity helpers
    - auth:      Authentication UI
    - charts:    UI-level wrappers around chart rendering (if needed)
"""

import importlib

__all__ = ["pages", "universe", "auth", "charts", "ai_notes"]


def __getattr__(name: str):
    """Lazily expose UI submodules without import-time Streamlit side effects."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
