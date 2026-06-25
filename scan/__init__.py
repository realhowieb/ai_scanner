"""Scan package initializer."""
__all__: list[str] = []

# Re-export common scan entrypoints if dependencies are installed.
try:
    from .breakout import breakout_scanner, run_breakout_scan  # type: ignore
except ImportError:
    pass
else:
    __all__ = [
        "breakout_scanner",
        "run_breakout_scan",
    ]
