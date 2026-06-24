"""Scheduler package initializer."""
try:
    from .jobs import (
        run_nasdaq_now,
        run_postmarket_now,
        run_premarket_now,
        run_sp500_now,
    )  # type: ignore
except ImportError:
    pass