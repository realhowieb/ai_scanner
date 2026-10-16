"""Compatibility import; implementation lives in services.account_tokens."""
import sys

from services import account_tokens as _implementation

sys.modules[__name__] = _implementation
