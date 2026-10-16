"""Compatibility import; implementation lives in domain.password_policy."""
import sys

from domain import password_policy as _implementation

sys.modules[__name__] = _implementation
