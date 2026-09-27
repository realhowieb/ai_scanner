"""Presentation guards for entitlement-sensitive evidence.

These helpers never rescore or mutate canonical scanner observations. They only
remove paid evidence from a copy before a customer-facing renderer receives it.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

PREBREAKOUT_KEYS = ("PreBreakoutProb%", "PreBreakoutProb", "PreBreakoutProbRaw", "PreBreakoutScore")


def can(entitlements: Any, capability: str) -> bool:
    return bool((entitlements or {}).get(capability)) if isinstance(entitlements, dict) else False


def redact_prebreakout_opportunity(value: Dict[str, Any], *, allowed: bool) -> Dict[str, Any]:
    """Return a display copy without Premium PreBreakout evidence when blocked."""
    out = dict(value or {})
    if allowed:
        return out
    signals = [s for s in (out.get("signals") or []) if str(s).lower() != "prebreakout"]
    out["signals"] = signals
    if "n_signals" in out:
        out["n_signals"] = len(signals)
    if "prebreakout" in str(out.get("primary_setup") or "").lower():
        labels = {
            "golden_cross": "Golden Cross", "breakout": "Breakout",
            "gapper": "Gapper", "gainer": "Momentum",
        }
        out["primary_setup"] = next((labels[s] for s in signals if s in labels), "Signal")
    out["prob"] = None
    model = dict(out.get("model") or {})
    model["prebreakout_prob"] = None
    out["model"] = model
    out["reasons"] = [r for r in (out.get("reasons") or []) if "prebreakout" not in str(r).lower()]
    out["positive_reasons"] = [
        r for r in (out.get("positive_reasons") or []) if "prebreakout" not in str(r).lower()
    ]
    if "Why" in out:
        out["Why"] = " · ".join(
            part.strip() for part in str(out.get("Why") or "").split("·")
            if "prebreakout" not in part.lower()
        )
    lifecycle = []
    for event in out.get("lifecycle") or []:
        item = dict(event)
        for key in ("signals", "signals_added", "signals_removed"):
            item[key] = [s for s in (item.get(key) or []) if str(s).lower() != "prebreakout"]
        lifecycle.append(item)
    if lifecycle:
        out["lifecycle"] = lifecycle
    return out


def redact_prebreakout_rows(rows: Iterable[Dict[str, Any]], *, allowed: bool) -> list[Dict[str, Any]]:
    return [redact_prebreakout_opportunity(row, allowed=allowed) for row in rows]


def redact_prebreakout_frame(df: Any, *, allowed: bool) -> Any:
    """Drop Premium model columns from a display copy; leave caller data intact."""
    if allowed or df is None:
        return df
    try:
        return df.drop(columns=[c for c in PREBREAKOUT_KEYS if c in df.columns], errors="ignore")
    except Exception:
        return df
