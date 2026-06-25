"""Data package initializer."""
try:
    from .fetch import fetch_hot_stocks, fetch_most_active_stocks, fetch_trending_stocks  # type: ignore
except Exception:
    pass