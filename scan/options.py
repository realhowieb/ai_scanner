"""Pure scan option helpers shared by Streamlit UI and tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_MARKET = "SP500"
DEFAULT_STRATEGY = "gap_up"
DEFAULT_PROFILE = "aggressive"

ADMIN_MIN_NASDAQ_SCAN = 100_000
ADMIN_MIN_COMBO_SCAN = 150_000
ADMIN_MIN_TOP_N = 10_000


@dataclass(frozen=True)
class ScanProfileOptions:
    min_gap: float
    unusual_volume: bool


@dataclass(frozen=True)
class ScanRunOptions:
    min_gap: float
    min_price: float
    max_price: float
    top_n: int
    premarket: bool
    afterhours: bool
    unusual_volume: bool


def normalize_market(value: object) -> str:
    market = str(value or "").strip().upper()
    if market in {"SP500", "NASDAQ", "COMBO"}:
        return market
    return DEFAULT_MARKET


def normalize_strategy(value: object) -> str:
    strategy = str(value or "").strip().lower()
    if strategy in {"gap_up", "gap_down", "most_active", "unusual_vol", "momentum", "breakout_only"}:
        return strategy
    return DEFAULT_STRATEGY


def normalize_profile(value: object) -> str:
    profile = str(value or "").strip().lower()
    if profile in {"aggressive", "regular", "conservative"}:
        return profile
    return DEFAULT_PROFILE


def apply_admin_caps(
    max_nasdaq_scan: int,
    max_combo_scan: int,
    top_n: int,
    *,
    is_admin: bool,
) -> tuple[int, int, int]:
    """Return scan caps, expanding them for admin users."""
    max_nasdaq_scan = int(max_nasdaq_scan)
    max_combo_scan = int(max_combo_scan)
    top_n = int(top_n)
    if not is_admin:
        return max_nasdaq_scan, max_combo_scan, top_n
    return (
        max(max_nasdaq_scan, ADMIN_MIN_NASDAQ_SCAN),
        max(max_combo_scan, ADMIN_MIN_COMBO_SCAN),
        max(top_n, ADMIN_MIN_TOP_N),
    )


def profile_options(profile: object, values: Mapping[str, Any]) -> ScanProfileOptions:
    """Resolve profile-specific scan settings from a mapping-like state object."""
    profile_name = normalize_profile(profile)
    if profile_name == "aggressive":
        return ScanProfileOptions(
            min_gap=float(values.get("min_gap_pct_aggressive", 0.0)),
            unusual_volume=bool(values.get("unusual_vol_aggressive", False)),
        )
    if profile_name == "conservative":
        return ScanProfileOptions(
            min_gap=float(values.get("min_gap_pct_conservative", 3.0)),
            unusual_volume=bool(values.get("unusual_vol_conservative", True)),
        )
    return ScanProfileOptions(
        min_gap=float(values.get("min_gap_pct", 1.0)),
        unusual_volume=bool(values.get("unusual_vol", True)),
    )


def build_scan_run_options(
    profile: object,
    values: Mapping[str, Any],
    *,
    is_admin: bool,
) -> ScanRunOptions:
    """Resolve scan runtime options from UI/session values."""
    top_n = int(values.get("top_n", 150))
    max_nasdaq_scan = int(values.get("max_nasdaq_scan", 2000))
    max_combo_scan = int(values.get("max_combo_scan", 4000))
    _max_nasdaq_scan, _max_combo_scan, top_n = apply_admin_caps(
        max_nasdaq_scan,
        max_combo_scan,
        top_n,
        is_admin=is_admin,
    )
    profile_cfg = profile_options(profile, values)
    return ScanRunOptions(
        min_gap=profile_cfg.min_gap,
        min_price=float(values.get("min_price", 1.0)),
        max_price=float(values.get("max_price", 1000.0)),
        top_n=top_n,
        premarket=bool(values.get("premarket", False)),
        afterhours=bool(values.get("afterhours", False)),
        unusual_volume=profile_cfg.unusual_volume,
    )
