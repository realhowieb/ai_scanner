"""Run 32 — deterministic Day Trader signal intelligence (pure, no ML, no I/O).

Synthesizes the validated intraday fields build_day_trader_metrics already
produces (chg_pct, gap_pct, rvol, vs_vwap_pct, adx, supertrend_direction, ewo)
into an interpretable summary:

    {direction, score, quality, reasons, conflicts, evidence}

Design principles:
  * DIRECTION (bullish/bearish/neutral) and STRENGTH (DT Score 0-100) are
    SEPARATE — a bearish setup can score high (a strong, coherent short setup).
  * ADX and RVOL are CONFIRMATION/strength only; they never vote direction.
  * Missing fields are honestly excluded (never assumed 0); the score
    renormalizes over the evidence that IS present. Too little evidence ->
    Neutral / "Insufficient data".
  * Extreme values are capped (5x RVOL is not 5x a 1x setup).
  * No execution language (no buy/sell). Descriptive intelligence only.

Pure functions — Streamlit-free — so they are unit tested directly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Scoring version. v2.0 = Run 33 recalibration (C1 count-weighted agreement,
# C2 raised normalization ceilings to break the ceiling saturation, C4 tiers).
DT_SCORE_VERSION = "2.0"

# --- Normalization knobs (documented, centralized). C2: ceilings raised so a
# typical liquid mover no longer maxes every sub-score (which pinned scores at
# ~100). Values chosen to spread the mid-range, not to fit any single regime. ---
_ADX_FLOOR, _ADX_FULL = 15.0, 55.0     # <15 weak; ceiling 40 -> 55
_RVOL_ORDINARY, _RVOL_STRONG = 1.0, 5.0  # 1x ordinary; ceiling 3x -> 5x
_VWAP_FULL = 2.0                        # 1% -> 2% for full contribution
_MOM_FULL = 5.0                         # 3% -> 5% for full contribution
_GAP_FULL = 5.0                         # 3% -> 5% for full contribution
_DEADBAND = 0.05                        # % noise band around 0 for direction

# DT Score weights (must sum to 1.0 across all factors; renormalized when a
# factor's input is missing).
_WEIGHTS = {
    "agreement": 0.35,   # directional signal agreement
    "adx": 0.20,         # trend strength
    "rvol": 0.20,        # participation
    "vwap": 0.10,        # VWAP relationship
    "momentum": 0.10,    # intraday momentum
    "gap": 0.05,         # gap context (only when it supports direction)
}
_CONFLICT_PENALTY = 5.0   # points removed per conflict
_CONFLICT_PENALTY_CAP = 25.0
_MIN_DIRECTIONAL = 2      # fewer available directional signals -> insufficient

# C4: quality-tier gates re-fit to the C1/C2 spread distribution. Once the
# ceiling saturation is broken a strong, complete, all-agree setup lands ~55-60
# (was ~100), so Strong gates at 55 and Weak at 35 (were 65 / 40). Held-out
# validation confirms/adjusts these before ship (docs/DT_SCORE_RECALIBRATION.md).
_STRONG_THRESHOLD = 55.0
_WEAK_THRESHOLD = 35.0


def _num(v: Any) -> Optional[float]:
    """Float or None (NaN/blank/unparseable -> None, never a silent 0)."""
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _supertrend_sign(value: Any) -> Optional[int]:
    """+1 green/bullish, -1 red/bearish, None when unknown/absent."""
    t = str(value or "").strip().lower()
    if t in ("green", "bullish", "up", "true"):
        return 1
    if t in ("red", "bearish", "down", "false"):
        return -1
    return None


def _sign(v: Optional[float]) -> Optional[int]:
    if v is None or abs(v) < _DEADBAND:
        return 0 if v is not None else None
    return 1 if v > 0 else -1


def _direction_votes(row: Dict[str, Any]) -> Dict[str, int]:
    """Per-signal directional votes (+1 bull / -1 bear); only present signals."""
    chg = _num(row.get("chg_pct"))
    gap = _num(row.get("gap_pct"))
    vsvwap = _num(row.get("vs_vwap_pct"))
    ewo = _num(row.get("ewo"))
    st = _supertrend_sign(row.get("supertrend_direction"))

    votes: Dict[str, int] = {}
    if vsvwap is not None:
        votes["vwap"] = 1 if vsvwap > _DEADBAND else -1 if vsvwap < -_DEADBAND else 0
    if st is not None:
        votes["supertrend"] = st
    if ewo is not None:
        votes["ewo"] = 1 if ewo > 0 else -1 if ewo < 0 else 0
    if chg is not None:
        votes["momentum"] = 1 if chg > _DEADBAND else -1 if chg < -_DEADBAND else 0
    # Gap only votes when the current move supports it (gap same sign as chg).
    if gap is not None and chg is not None and abs(gap) >= _DEADBAND:
        if (gap > 0 and chg > 0) or (gap < 0 and chg < 0):
            votes["gap"] = 1 if gap > 0 else -1
    return votes


def classify_day_trade_direction(row: Dict[str, Any]) -> str:
    votes = _direction_votes(row)
    directional = [v for v in votes.values() if v != 0]
    if len(directional) < _MIN_DIRECTIONAL:
        return "neutral"
    net = sum(directional)
    if net > 0:
        return "bullish"
    if net < 0:
        return "bearish"
    return "neutral"


def _agreement(row: Dict[str, Any]) -> Optional[float]:
    """Fraction of directional signals agreeing with the majority (0.5 tie ..
    1.0 unanimous). None when too few directional signals exist."""
    votes = [v for v in _direction_votes(row).values() if v != 0]
    if len(votes) < _MIN_DIRECTIONAL:
        return None
    bull = sum(1 for v in votes if v > 0)
    bear = len(votes) - bull
    return max(bull, bear) / len(votes)


def day_trade_conflicts(row: Dict[str, Any]) -> List[str]:
    """Descriptive conflict/context flags that reduce setup confidence."""
    out: List[str] = []
    chg = _num(row.get("chg_pct"))
    gap = _num(row.get("gap_pct"))
    vsvwap = _num(row.get("vs_vwap_pct"))
    ewo = _num(row.get("ewo"))
    rvol = _num(row.get("rvol"))
    adx = _num(row.get("adx"))
    st = _supertrend_sign(row.get("supertrend_direction"))

    # Mixed trend: SuperTrend disagrees with the VWAP side.
    if st is not None and vsvwap is not None and abs(vsvwap) >= _DEADBAND:
        if (vsvwap > 0) != (st > 0):
            out.append("Mixed trend signals")
    # Losing VWAP: positive momentum/gap but price is below VWAP (or vice-versa).
    if vsvwap is not None and chg is not None and abs(vsvwap) >= _DEADBAND and abs(chg) >= _DEADBAND:
        if (chg > 0 and vsvwap < 0) or (chg < 0 and vsvwap > 0):
            out.append("Losing VWAP")
    # Momentum disagreement: EWO and intraday change point opposite ways.
    if ewo is not None and chg is not None and abs(chg) >= _DEADBAND and abs(ewo) > 0:
        if (ewo > 0) != (chg > 0):
            out.append("Momentum disagreement")
    # Gap fading: a real gap but the current move is against it.
    if gap is not None and chg is not None and abs(gap) >= 0.5:
        if (gap > 0 and chg < -_DEADBAND) or (gap < 0 and chg > _DEADBAND):
            out.append("Gap fading")
    # Low participation.
    if rvol is not None and rvol < _RVOL_ORDINARY:
        out.append("Low participation")
    # Weak trend strength.
    if adx is not None and adx < _ADX_FLOOR:
        out.append("Weak trend strength")
    return out


def score_day_trade_setup(row: Dict[str, Any]) -> Optional[float]:
    """DT Score 0-100 (setup coherence/strength, not price prediction). None when
    there is too little evidence to score. Missing factors renormalize the
    weights; conflicts apply a bounded penalty; extremes are capped."""
    agreement = _agreement(row)
    if agreement is None:
        return None

    adx = _num(row.get("adx"))
    rvol = _num(row.get("rvol"))
    vsvwap = _num(row.get("vs_vwap_pct"))
    chg = _num(row.get("chg_pct"))
    gap = _num(row.get("gap_pct"))

    # C1: agreement is count-weighted so unanimity with few signals no longer
    # auto-maxes — 5/5 agreeing signals score higher than 3/3. Completeness is
    # the share of the 5 possible directional signals that are present.
    n_dir = len([v for v in _direction_votes(row).values() if v != 0])
    completeness = 0.4 + 0.6 * _clamp01(n_dir / 5.0)
    subs: Dict[str, float] = {"agreement": _clamp01((agreement - 0.5) / 0.5) * completeness}
    if adx is not None:
        subs["adx"] = _clamp01((adx - _ADX_FLOOR) / (_ADX_FULL - _ADX_FLOOR))
    if rvol is not None:
        subs["rvol"] = _clamp01((rvol - _RVOL_ORDINARY) / (_RVOL_STRONG - _RVOL_ORDINARY))
    if vsvwap is not None:
        subs["vwap"] = _clamp01(abs(vsvwap) / _VWAP_FULL)
    if chg is not None:
        subs["momentum"] = _clamp01(abs(chg) / _MOM_FULL)
    # Gap only rewards when it supports the move (not fading).
    if gap is not None and chg is not None and abs(gap) >= _DEADBAND and (gap > 0) == (chg > 0):
        subs["gap"] = _clamp01(abs(gap) / _GAP_FULL)

    total_w = sum(_WEIGHTS[k] for k in subs)
    if total_w <= 0:
        return None
    raw = sum(_WEIGHTS[k] * subs[k] for k in subs) / total_w * 100.0
    penalty = min(len(day_trade_conflicts(row)) * _CONFLICT_PENALTY, _CONFLICT_PENALTY_CAP)
    return round(max(0.0, min(100.0, raw - penalty)), 1)


def classify_setup_quality(row: Dict[str, Any]) -> str:
    """Strong / Developing / Weak / insufficient — from score AND agreement AND
    confirmation, never score alone."""
    agreement = _agreement(row)
    if agreement is None:
        return "insufficient"
    score = score_day_trade_setup(row)
    if score is None:
        return "insufficient"
    adx = _num(row.get("adx"))
    rvol = _num(row.get("rvol"))
    conflicts = day_trade_conflicts(row)
    confirmed = (adx is not None and adx >= 20.0) or (rvol is not None and rvol >= 1.5)

    # C4: tiers re-fit to the spread (C2) distribution — the old 65/40 gates left
    # everything "developing" once scores no longer pinned at ~100.
    if score >= _STRONG_THRESHOLD and agreement >= 0.7 and confirmed and len(conflicts) <= 1:
        return "strong"
    if score < _WEAK_THRESHOLD or agreement < 0.55 or (rvol is not None and rvol < _RVOL_ORDINARY) or len(conflicts) >= 3:
        return "weak"
    return "developing"


def day_trade_signal_reasons(row: Dict[str, Any]) -> List[str]:
    """Concise, descriptive reasons from present signals only."""
    out: List[str] = []
    vsvwap = _num(row.get("vs_vwap_pct"))
    st = _supertrend_sign(row.get("supertrend_direction"))
    ewo = _num(row.get("ewo"))
    adx = _num(row.get("adx"))
    rvol = _num(row.get("rvol"))
    gap = _num(row.get("gap_pct"))

    if vsvwap is not None and abs(vsvwap) >= _DEADBAND:
        out.append(f"{'Above' if vsvwap > 0 else 'Below'} VWAP ({vsvwap:+.2f}%)")
    if st is not None:
        out.append(f"SuperTrend {'green' if st > 0 else 'red'}")
    if ewo is not None:
        out.append(f"EWO {'positive' if ewo > 0 else 'negative'} ({ewo:+.2f})")
    if adx is not None:
        out.append(f"ADX {adx:.0f} confirms trend" if adx >= 20 else f"ADX only {adx:.0f}")
    if rvol is not None:
        out.append(f"RVOL {rvol:.1f}x confirms participation" if rvol >= 1.5 else f"RVOL {rvol:.1f}x")
    if gap is not None and abs(gap) >= 0.5:
        out.append(f"Gap {gap:+.1f}%")
    return out


_DIRECTION_ICON = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def day_trade_intelligence(row: Dict[str, Any]) -> Dict[str, Any]:
    """One structured, deterministic summary for a Day Trader row."""
    agreement = _agreement(row)
    if agreement is None:
        return {
            "direction": "neutral", "score": None, "quality": "insufficient",
            "reasons": day_trade_signal_reasons(row), "conflicts": [],
            "evidence": len([v for v in _direction_votes(row).values() if v != 0]),
        }
    return {
        "direction": classify_day_trade_direction(row),
        "score": score_day_trade_setup(row),
        "quality": classify_setup_quality(row),
        "reasons": day_trade_signal_reasons(row),
        "conflicts": day_trade_conflicts(row),
        "evidence": len([v for v in _direction_votes(row).values() if v != 0]),
    }


def direction_icon(direction: str) -> str:
    return _DIRECTION_ICON.get(str(direction or "").lower(), "⚪")
