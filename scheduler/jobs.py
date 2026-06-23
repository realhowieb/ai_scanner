"""Compatibility entrypoints for manual and scheduled scanner jobs."""

from __future__ import annotations

from typing import Any

from .cron_runner import run_and_save


_UNIVERSE_ALIASES = {
    "S&P 500": "SP500",
    "SP500": "SP500",
    "SP 500": "SP500",
    "NASDAQ": "NASDAQ",
    "NASDAQ100": "NASDAQ",
    "NASDAQ 100": "NASDAQ",
    "COMBO": "COMBO",
}


def _normalize_universe(value: str | None, default: str) -> str:
    if not value:
        return default
    key = str(value).strip().upper()
    return _UNIVERSE_ALIASES.get(key, key)


def _coerce_positive_int(value: Any, default: int | None = None) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _run_now(
    universe: str,
    *,
    username: str = "scheduler",
    profile: str = "regular",
    premarket: bool = False,
    afterhours: bool = False,
    unusual_volume: bool = False,
    min_gap: float | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    top_n: int | None = None,
    **_: Any,
) -> int:
    """Run a headless scan and return 0 on success, -1 on failure."""
    ok = run_and_save(
        _normalize_universe(universe, "SP500"),
        username=username,
        premarket=premarket,
        afterhours=afterhours,
        unusual_volume=unusual_volume,
        min_gap=min_gap,
        min_price=min_price,
        max_price=max_price,
        top_n=_coerce_positive_int(top_n),
        profile=profile,
    )
    success = bool(getattr(ok, "ok", ok))
    return 0 if success else -1


def run_sp500_now(**kwargs: Any) -> int:
    return _run_now("SP500", **kwargs)


def run_nasdaq_now(**kwargs: Any) -> int:
    return _run_now("NASDAQ", **kwargs)


def run_premarket_now(universe: str = "SP500", **kwargs: Any) -> int:
    return _run_now(
        _normalize_universe(universe, "SP500"),
        profile="premarket",
        premarket=True,
        **kwargs,
    )


def run_postmarket_now(universe: str = "SP500", **kwargs: Any) -> int:
    return _run_now(
        _normalize_universe(universe, "SP500"),
        profile="postmarket",
        afterhours=True,
        **kwargs,
    )


def run_premarket(settings: dict[str, Any] | None = None) -> int:
    settings = dict(settings or {})
    universe = _normalize_universe(settings.pop("universe_name", None), "SP500")
    settings.pop("profile", None)
    settings.pop("premarket", None)
    return run_premarket_now(universe=universe, **settings)


def run_postmarket(settings: dict[str, Any] | None = None) -> int:
    settings = dict(settings or {})
    universe = _normalize_universe(settings.pop("universe_name", None), "SP500")
    settings.pop("profile", None)
    settings.pop("afterhours", None)
    return run_postmarket_now(universe=universe, **settings)


__all__ = [
    "run_sp500_now",
    "run_nasdaq_now",
    "run_premarket_now",
    "run_postmarket_now",
    "run_premarket",
    "run_postmarket",
]
