"""Compatibility import; implementation lives in services.billing."""
import sys

from services import billing as _implementation

sys.modules[__name__] = _implementation
