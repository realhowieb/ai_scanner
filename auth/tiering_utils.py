

"""Tier capability helpers.

This module converts a Tier object into a set of capability flags that the
UI and scan engine can use. It keeps the logic for interpreting `features`
centralized so app.py stays slim.
"""

from typing import Dict, Any


def derive_tier_flags(tier: Any) -> Dict[str, bool]:
    """Return all capability flags based on tier attributes or features list.

    This helper looks first for explicit boolean attributes on the Tier
    instance (e.g. `can_scan_sp500`), and falls back to the `features`
    list if those attributes are missing. This means older Basic/Pro/
    Premium configs that only use `features` still work.
    """

    # Some Tier implementations may not have `features` at all.
    features = getattr(tier, "features", []) or []

    def _flag(attr: str, feature_name: str) -> bool:
        # Prefer explicit attribute on Tier class
        if hasattr(tier, attr):
            try:
                return bool(getattr(tier, attr))
            except Exception:
                pass

        # Fallback: feature list check
        if feature_name == "SP500 Scan":
            return ("SP500 Scan" in features) or ("SP500" in features)

        return feature_name in features

    return {
        "can_scan_sp500": _flag("can_scan_sp500", "SP500 Scan"),
        "can_scan_nasdaq": _flag("can_scan_nasdaq", "NASDAQ"),
        "can_premarket": _flag("can_premarket", "Premarket"),
        "can_afterhours": _flag("can_afterhours", "AfterHours"),
        "can_unusual_volume": _flag("can_unusual_volume", "UnusualVolume"),
        "can_export_csv": _flag("can_export_csv", "ExportCSV"),
        "can_ai_notes": _flag("can_ai_notes", "AI Notes"),
    }