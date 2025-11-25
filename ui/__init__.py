"""UI package for ai_scanner."""

# Expose pages module explicitly if available
try:
    from . import pages
except ImportError:
    pages = None

__all__ = ["pages"]
