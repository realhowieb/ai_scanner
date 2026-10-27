"""Compatibility import; implementation lives in domain.plan_labels."""
import sys

from domain import plan_labels as _implementation

sys.modules[__name__] = _implementation
