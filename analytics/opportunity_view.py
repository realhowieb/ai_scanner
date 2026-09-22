"""Run 40 — Intelligent Alerts & Opportunity Feed presentation layer (pure).

Aggregates HSF's EXISTING scanner/model outputs into a clear, deterministic
"OpportunityView": what deserves attention, why it matters, what changed, and how
urgently to surface it. It reads a canonical Run 36 observation (as produced by
production capture) and computes primary setup, scanner agreement, evidence-based
positive/risk reasons, meaningful changes vs a prior observation, a transparent
alert priority, a lifecycle state, and a dedup decision.

Hard boundaries (Run 40 non-goals):
  * This is NOT Opportunity Score and NOT a predictive model. `alert_priority` is
    an attention/urgency signal, never a return/probability/edge claim.
  * No scanner, model, or DT behavior is changed or re-run here.
  * Scanner agreement is presented as CONFIRMATION/CONTEXT, never proven edge.
  * Every reason/risk is backed by a value actually present on the observation;
    nothing is invented (no news/earnings/catalyst unless real data exists).

Pure and deterministic: identical inputs → identical OpportunityView.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "hsf-opportunity-view-1.0"

# Product-semantic display labels for scanners (reuses ui.opportunities vocab).
SCANNER_LABELS = {
    "prebreakout": "PreBreakout", "breakout": "Breakout", "breakout_only": "Breakout",
    "momentum": "Momentum", "unusual_vol": "Unusual Volume", "gap_up": "Gapper",
    "gap_down": "Gap Down", "most_active": "Most Active", "ai_confidence": "AI-ranked",
    "golden_cross": "Golden Cross",
}
# Deterministic primary-setup priority by product specificity (NOT by historical
# performance — Run 40 must not use unvalidated performance weighting). Most
# specific / model-driven setups first.
PRIMARY_ORDER = ["prebreakout", "golden_cross", "breakout", "breakout_only",
                 "momentum", "unusual_vol", "gap_up", "gap_down", "most_active"]

PRIORITY_LEVELS = ("HIGH", "MEDIUM", "LOW")
LIFECYCLE_STATES = ("NEW", "ACTIVE", "STRENGTHENING", "WEAKENING", "RESOLVED")

# Change-detection thresholds (meaningful transitions only).
_PROB_DELTA = 10.0        # PreBreakout probability points
_RVOL_DELTA = 0.5         # RVOL multiple


def _num(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _triggered(scanners: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in (scanners or []) if s.get("triggered", True)]


def select_primary_setup(scanners: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Deterministic primary reason a symbol appears. Returns (name, label)."""
    names = {str(s.get("name")) for s in _triggered(scanners)}
    for n in PRIMARY_ORDER:
        if n in names:
            return n, SCANNER_LABELS.get(n, n)
    for s in _triggered(scanners):  # fall back to first triggered
        n = str(s.get("name"))
        return n, SCANNER_LABELS.get(n, n)
    return None, None


def scanner_agreement(scanners: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Confirmation/context — how many distinct scanners fired (NOT an edge claim)."""
    trig = _triggered(scanners)
    names = []
    for s in trig:
        n = str(s.get("name"))
        if n not in names:
            names.append(n)
    return {"count": len(names), "names": names,
            "labels": [SCANNER_LABELS.get(n, n) for n in names]}


def overall_direction(scanners: Sequence[Dict[str, Any]]) -> str:
    votes = [str(s.get("direction") or "long").lower() for s in _triggered(scanners)]
    longs = sum(1 for v in votes if v == "long")
    shorts = sum(1 for v in votes if v == "short")
    if longs and shorts:
        return "mixed"
    if shorts and not longs:
        return "bearish"
    if longs:
        return "bullish"
    return "neutral"


def _models(obs: Dict[str, Any]) -> Dict[str, Any]:
    return obs.get("models") or {}


def _prebreakout_prob(obs: Dict[str, Any]) -> Optional[float]:
    pb = _models(obs).get("prebreakout") or {}
    p = _num(pb.get("probability"))
    if p is None:
        return None
    return p * 100 if p <= 1.0 else p  # accept fraction or percent


def positive_reasons(obs: Dict[str, Any]) -> List[str]:
    """Evidence-based reasons from present fields only (Task 5)."""
    reasons: List[str] = []
    ind = obs.get("indicators") or {}
    prob = _prebreakout_prob(obs)
    if prob is not None:
        reasons.append(f"PreBreakout probability {prob:.0f}%")
    ai = _num((_models(obs).get("ai_confidence") or {}).get("confidence"))
    if ai is not None:
        reasons.append(f"AI confidence {ai * 100 if ai <= 1 else ai:.0f}%")
    rvol = _num(ind.get("rvol"))
    if rvol is not None and rvol >= 1.5:
        reasons.append(f"RVOL {rvol:.1f}x")
    vsv = _num(ind.get("vs_vwap_pct"))
    if vsv is not None and vsv > 0:
        reasons.append("Above VWAP")
    adx = _num(ind.get("adx"))
    if adx is not None and adx >= 20:
        reasons.append(f"ADX {adx:.0f} (trending)")
    chg = _num(ind.get("chg_pct"))
    if chg is not None and chg > 0:
        reasons.append(f"Up {chg:+.1f}%")
    gap = _num(ind.get("gap_pct"))
    if gap is not None and abs(gap) >= 1:
        reasons.append(f"Gap {gap:+.1f}%")
    agree = scanner_agreement(obs.get("scanners") or [])
    if agree["count"] >= 2:
        reasons.append(f"{agree['count']} scanners agree")
    return reasons


def risk_reasons(obs: Dict[str, Any]) -> List[str]:
    """Complementary caution flags — only data-supported risks (Task 6)."""
    risks: List[str] = []
    ind = obs.get("indicators") or {}
    dq = obs.get("data_quality") or {}
    vsv = _num(ind.get("vs_vwap_pct"))
    if vsv is not None:
        if vsv < 0:
            risks.append("Below VWAP")
        elif vsv >= 4:
            risks.append(f"Extended above VWAP ({vsv:+.1f}%)")
    rvol = _num(ind.get("rvol"))
    if rvol is not None and rvol < 1.0:
        risks.append(f"Low participation (RVOL {rvol:.1f}x)")
    adx = _num(ind.get("adx"))
    if adx is not None and adx < 15:
        risks.append(f"Weak trend (ADX {adx:.0f})")
    gap = _num(ind.get("gap_pct"))
    if gap is not None and abs(gap) >= 5:
        risks.append(f"Large opening gap ({gap:+.1f}%)")
    vol = _num(ind.get("atr_pct"))
    if vol is not None and vol >= 6:
        risks.append(f"High volatility ({vol:.0f}%)")
    if overall_direction(obs.get("scanners") or []) == "mixed":
        risks.append("Conflicting scanner directions")
    if str(obs.get("session") or "").lower() in ("afternoon", "afterhours", "late"):
        risks.append("Late-session setup")
    # Data-quality risks (Task 14) — real from Run 37/36 metadata.
    if dq.get("stale"):
        risks.append("Stale data")
    if dq.get("fallback_used"):
        risks.append("Incomplete data (fallback)")
    elif dq.get("feature_completeness") is not None and dq["feature_completeness"] < 0.6:
        risks.append("Partial data")
    return risks


def detect_changes(current: Dict[str, Any], prior: Optional[Dict[str, Any]]) -> List[str]:
    """Meaningful state transitions vs a prior observation (Task 7). Empty when no
    prior. Only prioritized transitions, not every numeric wiggle."""
    if not prior:
        return []
    changes: List[str] = []
    ci, pi = current.get("indicators") or {}, prior.get("indicators") or {}
    cp, pp = _prebreakout_prob(current), _prebreakout_prob(prior)
    if cp is not None and pp is not None and abs(cp - pp) >= _PROB_DELTA:
        changes.append(f"PreBreakout probability: {pp:.0f}% → {cp:.0f}%")
    cr, pr = _num(ci.get("rvol")), _num(pi.get("rvol"))
    if cr is not None and pr is not None and abs(cr - pr) >= _RVOL_DELTA:
        changes.append(f"RVOL: {pr:.1f}x → {cr:.1f}x")
    cv, pv = _num(ci.get("vs_vwap_pct")), _num(pi.get("vs_vwap_pct"))
    if cv is not None and pv is not None and (cv > 0) != (pv > 0):
        changes.append("Moved above VWAP" if cv > 0 else "Lost VWAP")
    ca = scanner_agreement(current.get("scanners") or [])
    pa = scanner_agreement(prior.get("scanners") or [])
    new_scanners = [SCANNER_LABELS.get(n, n) for n in ca["names"] if n not in pa["names"]]
    if new_scanners:
        changes.append("New scanner: " + ", ".join(new_scanners))
    if ca["count"] != pa["count"]:
        changes.append(f"Scanner agreement: {pa['count']} → {ca['count']}")
    cd, pd = overall_direction(current.get("scanners") or []), overall_direction(prior.get("scanners") or [])
    if cd != pd:
        changes.append(f"Direction: {pd.title()} → {cd.title()}")
    return changes


def _has(changes: List[str], *keys: str) -> bool:
    return any(any(k.lower() in c.lower() for k in keys) for c in changes)


def alert_priority(obs: Dict[str, Any], changes: Optional[List[str]] = None) -> Tuple[str, str]:
    """Transparent HIGH/MEDIUM/LOW urgency — NOT a prediction (Task 8).

    Rules (deterministic, documented in docs/INTELLIGENT_ALERTS.md). Stale or
    fallback data caps priority so untrustworthy setups are never surfaced as
    HIGH urgency.
    """
    changes = changes or []
    ind = obs.get("indicators") or {}
    dq = obs.get("data_quality") or {}
    agree = scanner_agreement(obs.get("scanners") or [])["count"]
    prob = _prebreakout_prob(obs)
    rvol = _num(ind.get("rvol"))
    capped = bool(dq.get("stale") or dq.get("fallback_used"))

    high = (agree >= 3 or (prob is not None and prob >= 70)
            or _has(changes, "Direction:", "Moved above VWAP", "New scanner"))
    medium = (agree == 2 or (prob is not None and prob >= 40)
              or (rvol is not None and rvol >= 2) or bool(changes))

    if high and not capped:
        parts = []
        if agree >= 3:
            parts.append(f"{agree} scanners agree")
        if prob is not None and prob >= 70:
            parts.append(f"PreBreakout {prob:.0f}%")
        if _has(changes, "Direction:", "Moved above VWAP", "New scanner"):
            parts.append("significant change")
        return "HIGH", "; ".join(parts) or "multiple strong confirmations"
    if high and capped:
        return "MEDIUM", "strong signals but data quality is limited"
    if medium:
        parts = []
        if agree == 2:
            parts.append("2 scanners agree")
        if prob is not None and prob >= 40:
            parts.append(f"PreBreakout {prob:.0f}%")
        if rvol is not None and rvol >= 2:
            parts.append(f"RVOL {rvol:.1f}x")
        if changes:
            parts.append("state changed")
        return "MEDIUM", "; ".join(parts) or "moderate confirmation"
    return "LOW", "single setup, limited confirmation"


def lifecycle_state(
    *, prior_state: Optional[str], changes: List[str], present: bool = True,
) -> str:
    """NEW/ACTIVE/STRENGTHENING/WEAKENING/RESOLVED from history (Task 10)."""
    if not present:
        return "RESOLVED"
    if prior_state is None:
        return "NEW"
    if _has(changes, "Moved above VWAP", "New scanner", "→") and _strengthening(changes):
        return "STRENGTHENING"
    if _weakening(changes):
        return "WEAKENING"
    return "ACTIVE"


def _strengthening(changes: List[str]) -> bool:
    for c in changes:
        if "Moved above VWAP" in c or "New scanner" in c:
            return True
        if "→" in c and _increased(c):
            return True
    return False


def _weakening(changes: List[str]) -> bool:
    for c in changes:
        if "Lost VWAP" in c:
            return True
        if "Direction:" in c and "Neutral" in c.split("→")[-1]:
            return True
        if "→" in c and _decreased(c):
            return True
    return False


def _two_numbers(c: str) -> Optional[Tuple[float, float]]:
    import re
    nums = re.findall(r"-?\d+\.?\d*", c.split("→")[0]), re.findall(r"-?\d+\.?\d*", c.split("→")[-1])
    if nums[0] and nums[1]:
        try:
            return float(nums[0][-1]), float(nums[1][-1])
        except ValueError:
            return None
    return None


def _increased(c: str) -> bool:
    tn = _two_numbers(c)
    return tn is not None and tn[1] > tn[0]


def _decreased(c: str) -> bool:
    tn = _two_numbers(c)
    return tn is not None and tn[1] < tn[0]


_PRIORITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def should_alert(view: Dict[str, Any], prior_alert: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Dedup/cooldown decision (Task 9). A new alert requires something
    meaningful; otherwise the existing alert is updated, not re-fired."""
    if prior_alert is None:
        return True, "first trigger"
    changes = view.get("changes_since_prior") or []
    cur_rank = _PRIORITY_RANK.get(view.get("alert_priority"), 0)
    prev_rank = _PRIORITY_RANK.get(prior_alert.get("alert_priority"), 0)
    if cur_rank > prev_rank:
        return True, "priority increased"
    if prior_alert.get("lifecycle_state") == "RESOLVED":
        return True, "setup reappeared"
    triggers = ("New scanner", "Direction:", "Moved above VWAP", "Lost VWAP",
                "PreBreakout probability", "RVOL")
    if _has(changes, *triggers):
        return True, "material change"
    return False, "no material change (cooldown)"


def freshness(obs: Dict[str, Any]) -> str:
    """Fresh / Partial Data / Delayed / Stale from Run 37/36 metadata (Task 14)."""
    dq = obs.get("data_quality") or {}
    if dq.get("stale"):
        return "Stale"
    if dq.get("fallback_used"):
        return "Partial Data"
    fc = dq.get("feature_completeness")
    if fc is not None and fc < 0.6:
        return "Partial Data"
    return "Fresh"


def build_opportunity_view(
    obs: Dict[str, Any],
    *,
    prior_obs: Optional[Dict[str, Any]] = None,
    prior_alert: Optional[Dict[str, Any]] = None,
    watchlist: Optional[Sequence[str]] = None,
    present: bool = True,
) -> Dict[str, Any]:
    """Assemble the canonical OpportunityView (Task 2). Presentation only."""
    scanners = obs.get("scanners") or []
    ind = obs.get("indicators") or {}
    mkt = obs.get("market") or {}
    primary_name, primary_label = select_primary_setup(scanners)
    agree = scanner_agreement(scanners)
    changes = detect_changes(obs, prior_obs)
    priority, priority_reason = alert_priority(obs, changes)
    wl = {str(x).upper() for x in (watchlist or [])}
    symbol = str(obs.get("symbol") or "").upper()

    view = {
        "schema_version": SCHEMA_VERSION,
        "observation_id": obs.get("observation_id"),
        "symbol": symbol,
        "timestamp": obs.get("timestamp"),
        "session": obs.get("session"),
        "price": mkt.get("price"),
        "change_pct": _num(ind.get("chg_pct")),
        "primary_setup": primary_label,
        "primary_setup_name": primary_name,
        "secondary_setups": [lbl for n, lbl in zip(agree["names"], agree["labels"])
                             if n != primary_name],
        "scanner_count": agree["count"],
        "scanner_names": agree["labels"],
        "direction": overall_direction(scanners),
        "scores": {
            "prebreakout_probability": _prebreakout_prob(obs),
            "ai_confidence": _num((_models(obs).get("ai_confidence") or {}).get("confidence")),
        },
        "positive_reasons": positive_reasons(obs),
        "risk_reasons": risk_reasons(obs),
        "changes_since_prior": changes,
        "data_quality": obs.get("data_quality"),
        "freshness": freshness(obs),
        "is_watchlist": symbol in wl,
        "alert_priority": priority,
        "priority_reason": priority_reason,
    }
    view["lifecycle_state"] = lifecycle_state(
        prior_state=(prior_alert or {}).get("lifecycle_state") if prior_alert else (
            "ACTIVE" if prior_obs else None),
        changes=changes, present=present)
    fire, dedup_reason = should_alert(view, prior_alert)
    view["should_alert"] = fire
    view["dedup_reason"] = dedup_reason
    return view


FEED_FILTERS = ("All", "High Priority", "PreBreakout", "Momentum", "Unusual Volume",
                "Gappers", "Bullish", "Bearish", "Watchlist", "New", "Strengthening")


def filter_feed(views: Sequence[Dict[str, Any]], flt: str) -> List[Dict[str, Any]]:
    """Apply a feed filter (Task 12). Unknown/`All` returns everything."""
    f = str(flt or "All")
    if f in ("All", ""):
        return list(views)
    def _match(v: Dict[str, Any]) -> bool:
        labels = {str(n) for n in (v.get("scanner_names") or [])}
        if f == "High Priority":
            return v.get("alert_priority") == "HIGH"
        if f == "PreBreakout":
            return "PreBreakout" in labels
        if f == "Momentum":
            return "Momentum" in labels
        if f == "Unusual Volume":
            return "Unusual Volume" in labels
        if f == "Gappers":
            return "Gapper" in labels or "Gap Down" in labels
        if f == "Bullish":
            return v.get("direction") == "bullish"
        if f == "Bearish":
            return v.get("direction") == "bearish"
        if f == "Watchlist":
            return bool(v.get("is_watchlist"))
        if f == "New":
            return v.get("lifecycle_state") == "NEW"
        if f == "Strengthening":
            return v.get("lifecycle_state") == "STRENGTHENING"
        return True
    return [v for v in views if _match(v)]


def empty_state_message(*, scanned: Optional[int], detected: Optional[int],
                        high_priority: int = 0) -> str:
    """Real-count empty state (Task 17). Only shows numbers actually provided."""
    if high_priority > 0:
        return ""
    parts = ["No high-priority opportunities right now."]
    if scanned is not None:
        parts.append(f"HSF scanned {scanned:,} symbols.")
    if detected is not None:
        parts.append(f"{detected:,} setups were detected, but none met the "
                     f"current High Priority criteria.")
    return " ".join(parts)


def rank_feed(views: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic feed ordering: priority, then scanner agreement, then
    PreBreakout probability, then symbol. NOT a predictive ranking."""
    def _key(v: Dict[str, Any]):
        return (
            -_PRIORITY_RANK.get(v.get("alert_priority"), 0),
            -(v.get("scanner_count") or 0),
            -((v.get("scores") or {}).get("prebreakout_probability") or 0),
            str(v.get("symbol") or ""),
        )
    return sorted(views, key=_key)
