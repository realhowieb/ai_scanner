"""Scan package initializer."""
# Re-export common scan entrypoints if present
try:
    from .breakout import run_sp500_scan, run_nasdaq_scan, breakout_scanner  # type: ignore
except ImportError:
    try:
        from .breakout import breakout_scanner  # type: ignore
    except ImportError:
        pass