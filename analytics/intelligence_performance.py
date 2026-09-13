"""Run 26 — HSF Intelligence Performance (read-only descriptive analytics).

Composes the persisted evidence from two SEPARATE systems into understandable,
deterministic findings. It adds NO new pipeline and writes nothing:

  * Opportunity Intelligence  -> db.opportunity_outcomes (Run 25/25A)
  * Alert Intelligence        -> db.intelligence_alerts.get_alert_quality_summary
                                 (Run 24/24A)

The two systems keep separate denominators and semantics — they are presented
side by side, never merged. No scans, freezes, maturation, evaluation, delivery,
writes, or Claude. Every finding is traceable to persisted Run 24A/25A rows.

Nothing here feeds back into HSF Score, ranking, thresholds, or alert behavior.
It measures HSF-STATE history, never price / investment return.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

# Sample floor reused from the measurement layers (one definition).
from db.opportunity_outcomes import MIN_OUTCOME_SAMPLE as MIN_SAMPLE

# Conservative, centralized, NEVER tuned to results. A cohort comparison only
# becomes a finding when rates differ by at least this much (5 percentage points).
MIN_MEANINGFUL_RATE_DELTA = 0.05
# Horizon-decay magnitude thresholds (percentage-point drop first->last horizon).
_SHARP_DECAY_DROP = 0.20
_GRADUAL_DECAY_DROP = MIN_MEANINGFUL_RATE_DELTA

# Canonical score-band order (high -> low); mirrors analytics.opportunity_outcomes.
_BAND_ORDER = ["75+", "60-74", "50-59", "<50"]
_SIGNAL_ORDER = ["3+", "2", "1"]  # "0" is malformed per 25A — never a strength cohort


def _rows_by_key(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("key")): r for r in (rows or [])}


def _ready(row: Optional[Dict[str, Any]]) -> bool:
    return bool(row) and int(row.get("comparable") or 0) >= MIN_SAMPLE


def evidence_strength(*samples: int) -> str:
    """EARLY / MODERATE / STRONG from the smallest comparable sample involved."""
    n = min(samples) if samples else 0
    if n >= 5 * MIN_SAMPLE:
        return "STRONG"
    if n >= 2 * MIN_SAMPLE:
        return "MODERATE"
    return "EARLY"


def evidence_readiness(opp: Dict[str, Any], aq: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, documented readiness tier — NOT a marketing threshold.

    EARLY       no dimension has a comparable sample >= MIN.
    DEVELOPING  some cohort(s) ready, but fewer than 3 major cohorts.
    SUFFICIENT  >= 3 major cohorts ready AND >= 3 horizons ready.
    ROBUST      all 4 horizons ready AND >= 2 statuses AND >= 3 bands ready AND
                overall comparable >= 10x MIN (intentionally hard to reach now).
    """
    ready_h = sum(1 for r in opp.get("by_horizon", []) if _ready(r))
    ready_s = sum(1 for r in opp.get("by_status", []) if _ready(r))
    ready_b = sum(1 for r in opp.get("by_score_band", []) if _ready(r))
    major_ready = ready_h + ready_s + ready_b
    opp_comparable = int(opp.get("comparable") or 0)
    aq_comparable = int(aq.get("evaluable") or 0)

    if major_ready == 0 and opp_comparable < MIN_SAMPLE and aq_comparable < MIN_SAMPLE:
        tier = "EARLY"
    elif (ready_h == 4 and ready_s >= 2 and ready_b >= 3 and opp_comparable >= 10 * MIN_SAMPLE):
        tier = "ROBUST"
    elif major_ready >= 3 and ready_h >= 3:
        tier = "SUFFICIENT"
    elif major_ready >= 1:
        tier = "DEVELOPING"
    else:
        tier = "EARLY"
    return {"tier": tier, "ready_horizons": ready_h, "ready_statuses": ready_s,
            "ready_score_bands": ready_b, "opportunity_comparable": opp_comparable,
            "alert_comparable": aq_comparable, "min_sample": MIN_SAMPLE}


def horizon_decay(opp: Dict[str, Any]) -> Dict[str, Any]:
    """Does persistence follow-through weaken as the horizon grows? Deterministic,
    no statistical modelling. Only ready horizons are considered."""
    order = {"NEXT": 0, "H24": 1, "H72": 2, "H120": 3}
    pts = [(order.get(r["key"], 9), r["follow_through_rate"], r)
           for r in opp.get("by_horizon", []) if _ready(r) and r.get("follow_through_rate") is not None]
    pts.sort(key=lambda p: p[0])
    if len(pts) < 2:
        return {"state": "INSUFFICIENT_SAMPLE", "points": len(pts)}
    first, last = pts[0][1], pts[-1][1]
    drop = first - last
    rising = last - first
    non_increasing = all(pts[i][1] >= pts[i + 1][1] - 1e-9 for i in range(len(pts) - 1))
    non_decreasing = all(pts[i][1] <= pts[i + 1][1] + 1e-9 for i in range(len(pts) - 1))
    if drop >= _SHARP_DECAY_DROP:
        state = "SHARP_DECAY"
    elif non_increasing and drop >= _GRADUAL_DECAY_DROP:
        state = "GRADUAL_DECAY"
    elif non_decreasing and rising >= _GRADUAL_DECAY_DROP:
        state = "IMPROVING"
    elif abs(drop) < _GRADUAL_DECAY_DROP:
        state = "STABLE"
    else:
        state = "MIXED"
    return {"state": state, "first": first, "last": last,
            "horizons": [p[2]["key"] for p in pts]}


def _monotonicity(rows_by_key: Dict[str, Dict[str, Any]], order: List[str]) -> Dict[str, Any]:
    """Are follow-through rates non-increasing down the ordered cohorts (highest
    first)? MONOTONIC / MOSTLY_MONOTONIC / NON_MONOTONIC / INSUFFICIENT_SAMPLE.
    An inversion counts only when a lower cohort exceeds a higher one by more than
    MIN_MEANINGFUL_RATE_DELTA (tiny wiggles are not inversions)."""
    seq = [(k, rows_by_key[k]["follow_through_rate"]) for k in order
           if k in rows_by_key and _ready(rows_by_key[k])
           and rows_by_key[k].get("follow_through_rate") is not None]
    if len(seq) < 2:
        return {"state": "INSUFFICIENT_SAMPLE", "ordered": [k for k, _ in seq]}
    inversions = 0
    for i in range(len(seq) - 1):
        if seq[i + 1][1] > seq[i][1] + MIN_MEANINGFUL_RATE_DELTA:
            inversions += 1
    state = "MONOTONIC" if inversions == 0 else ("MOSTLY_MONOTONIC" if inversions == 1 else "NON_MONOTONIC")
    return {"state": state, "inversions": inversions,
            "ordered": [{"key": k, "rate": r} for k, r in seq]}


# Pref-key -> event type (mirrors analytics.opportunity_events._EVENT_PREF).
_PREF_TO_EVENT = {
    "new": "NEW_OPPORTUNITY", "upgrade": "STATUS_UPGRADE", "downgrade": "STATUS_DOWNGRADE",
    "fading": "FADING", "dropped": "DROPPED", "rising": "RISING", "falling": "FALLING",
    "signal_added": "SIGNAL_ADDED", "signal_removed": "SIGNAL_REMOVED",
}
_NOISE_MAP = {"Promising": "HIGH_VALUE", "High reversal": "POTENTIALLY_NOISY",
              "Mixed": "MIXED", "INSUFFICIENT_SAMPLE": "INSUFFICIENT_SAMPLE"}


def alert_noise_assessment(aq: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-event noise label, REUSING Run 24A's assessment (no new thresholds).
    Maps Promising->HIGH_VALUE, High reversal->POTENTIALLY_NOISY, Mixed->MIXED."""
    dist = (aq.get("frequency") or {}).get("event_type_distribution") or {}
    out = []
    for e in aq.get("by_event_type", []):
        et = e.get("event_type")
        out.append({
            "event_type": et, "matured": e.get("matured"), "evaluable": e.get("evaluable"),
            "confirmation_rate": e.get("confirmation_rate"), "reversed": e.get("reversed"),
            "detected_volume": dist.get(et),
            "assessment": _NOISE_MAP.get(e.get("assessment"), "MIXED"),
        })
    return out


def default_preference_cohorts(aq: Dict[str, Any]) -> Dict[str, Any]:
    """Compare default-ENABLED vs default-DISABLED alert event cohorts (Run 22
    defaults discovered from code). SUPPORTED / MIXED / NOT_SUPPORTED /
    INSUFFICIENT_SAMPLE. Descriptive only — never changes defaults."""
    try:
        from analytics.opportunity_events import DEFAULT_PREFERENCES
    except Exception:
        DEFAULT_PREFERENCES = {}
    enabled_events = {_PREF_TO_EVENT[k] for k, v in DEFAULT_PREFERENCES.items() if v and k in _PREF_TO_EVENT}
    by_event = {e.get("event_type"): e for e in aq.get("by_event_type", [])}

    def _agg(events):
        evaluable = sum(int(by_event[e].get("evaluable") or 0) for e in events if e in by_event)
        confirmed = sum(int(by_event[e].get("confirmed") or 0) for e in events if e in by_event)
        reversed_ = sum(int(by_event[e].get("reversed") or 0) for e in events if e in by_event)
        return evaluable, confirmed, reversed_

    all_events = set(by_event) | set(enabled_events)
    disabled_events = {e for e in all_events if e not in enabled_events and e != "VERSION_CHANGED"}
    en_eval, en_conf, en_rev = _agg(enabled_events)
    di_eval, di_conf, di_rev = _agg(disabled_events)
    result = {"enabled_events": sorted(enabled_events), "disabled_events": sorted(disabled_events),
              "enabled_evaluable": en_eval, "disabled_evaluable": di_eval,
              "enabled_confirmation": (en_conf / en_eval) if en_eval else None,
              "disabled_confirmation": (di_conf / di_eval) if di_eval else None,
              "assessment": "INSUFFICIENT_SAMPLE"}
    if en_eval < MIN_SAMPLE or di_eval < MIN_SAMPLE:
        return result
    en_c, di_c = en_conf / en_eval, di_conf / di_eval
    if en_c - di_c >= MIN_MEANINGFUL_RATE_DELTA:
        result["assessment"] = "SUPPORTED"
    elif di_c - en_c >= MIN_MEANINGFUL_RATE_DELTA:
        result["assessment"] = "NOT_SUPPORTED"
    else:
        result["assessment"] = "MIXED"
    return result


def derive_supported_findings(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic, template-based findings (NO Claude). A finding is emitted
    only when samples are adequate and a cohort difference exceeds the centralized
    meaningful threshold; contradictory evidence yields a 'MIXED' note, not a
    cherry-picked claim. Empty list is a valid, honest result."""
    findings: List[Dict[str, Any]] = []
    opp = summary.get("opportunity") or {}
    by_band = _rows_by_key(opp.get("by_score_band"))
    by_status = _rows_by_key(opp.get("by_status"))

    # Score-band monotonicity.
    mono = summary.get("score_monotonicity") or {}
    if mono.get("state") in ("MONOTONIC", "MOSTLY_MONOTONIC"):
        top, bot = _BAND_ORDER[0], None
        for b in reversed(_BAND_ORDER):
            if b in by_band and _ready(by_band[b]):
                bot = b
                break
        if top in by_band and bot and bot != top and _ready(by_band[top]):
            hi, lo = by_band[top], by_band[bot]
            if (hi["follow_through_rate"] - lo["follow_through_rate"]) >= MIN_MEANINGFUL_RATE_DELTA:
                findings.append({
                    "statement": f"Higher HSF score bands show stronger persistence "
                                 f"({top}: {hi['follow_through_rate']*100:.0f}% vs {bot}: "
                                 f"{lo['follow_through_rate']*100:.0f}% persisted/strengthened).",
                    "evidence_strength": evidence_strength(hi["comparable"], lo["comparable"]),
                    "detail": f"N={hi['comparable']} vs {lo['comparable']}", "dimension": "score_band",
                })
    elif mono.get("state") == "NON_MONOTONIC":
        findings.append({"statement": "HSF score-band persistence is not consistently ordered across bands.",
                         "evidence_strength": "EARLY", "detail": "Mixed cohort ordering.",
                         "dimension": "score_band"})

    # STRONG vs WATCH stability.
    if _ready(by_status.get("STRONG")) and _ready(by_status.get("WATCH")):
        s, w = by_status["STRONG"], by_status["WATCH"]
        delta = (s["follow_through_rate"] or 0) - (w["follow_through_rate"] or 0)
        if delta >= MIN_MEANINGFUL_RATE_DELTA:
            findings.append({
                "statement": "STRONG opportunities persist more than WATCH opportunities.",
                "evidence_strength": evidence_strength(s["comparable"], w["comparable"]),
                "detail": f"STRONG {s['follow_through_rate']*100:.0f}% vs WATCH "
                          f"{w['follow_through_rate']*100:.0f}% (N={s['comparable']} vs {w['comparable']}).",
                "dimension": "status"})

    # Confirming-signal monotonicity.
    smono = summary.get("signal_monotonicity") or {}
    if smono.get("state") in ("MONOTONIC", "MOSTLY_MONOTONIC") and len(smono.get("ordered", [])) >= 2:
        findings.append({
            "statement": "More confirming signals correspond to stronger HSF-state persistence.",
            "evidence_strength": "MODERATE", "detail": "Persistence rises with confirming-signal count.",
            "dimension": "confirming_signals"})

    # Horizon decay.
    decay = summary.get("horizon_decay") or {}
    if decay.get("state") in ("GRADUAL_DECAY", "SHARP_DECAY"):
        findings.append({
            "statement": f"HSF-state persistence decays over longer horizons ({decay['state'].lower()}).",
            "evidence_strength": "MODERATE",
            "detail": f"{decay.get('first', 0)*100:.0f}% -> {decay.get('last', 0)*100:.0f}% "
                      f"across {'/'.join(decay.get('horizons', []))}.", "dimension": "horizon"})

    # Alert event value (HIGH_VALUE only — conservative).
    for n in summary.get("alert_noise") or []:
        if n.get("assessment") == "HIGH_VALUE" and (n.get("evaluable") or 0) >= MIN_SAMPLE:
            findings.append({
                "statement": f"{n['event_type']} alerts show strong subsequent follow-through.",
                "evidence_strength": evidence_strength(int(n.get("evaluable") or 0)),
                "detail": f"N={n.get('evaluable')}, confirmation "
                          f"{(n.get('confirmation_rate') or 0)*100:.0f}%.", "dimension": "alert"})
    return findings


def get_intelligence_performance_summary(*, days_back: int = 90) -> Dict[str, Any]:
    """Top-level read-only performance snapshot composing persisted evidence.
    Stable, testable shape. Never raises (returns a safe structure)."""
    try:
        from db.intelligence_alerts import get_alert_quality_summary
        from db.opportunity_outcomes import (
            get_degraded_cohort_outcomes,
            get_intelligence_evidence_freshness,
            get_opportunity_outcome_summary,
        )
    except Exception:
        return {"available": False, "reason": "evidence layer unavailable"}

    opp = get_opportunity_outcome_summary(days_back=days_back)
    aq = get_alert_quality_summary(days_back=days_back)
    degraded = get_degraded_cohort_outcomes(days_back=days_back)
    freshness = get_intelligence_evidence_freshness()

    readiness = evidence_readiness(opp, aq)
    decay = horizon_decay(opp)
    score_mono = _monotonicity(_rows_by_key(opp.get("by_score_band")), _BAND_ORDER)
    signal_mono = _monotonicity(_rows_by_key(opp.get("by_signal_count")), _SIGNAL_ORDER)
    noise = alert_noise_assessment(aq)
    defaults = default_preference_cohorts(aq)

    # Regime coverage (never reconstructed — 25A never froze regime).
    regime_comparable = sum(int(r.get("comparable") or 0) for r in opp.get("by_regime", [])
                            if str(r.get("key")) != "UNKNOWN")
    regime_state = "AVAILABLE" if regime_comparable >= MIN_SAMPLE else "INSUFFICIENT_COVERAGE"

    summary = {
        "available": True, "generated_at": _dt.datetime.now(_dt.timezone.utc),
        "readiness": readiness, "freshness": freshness,
        "opportunity": opp, "alert_quality": aq, "degraded_recovery": degraded,
        "horizon_decay": decay, "score_monotonicity": score_mono,
        "signal_monotonicity": signal_mono, "alert_noise": noise,
        "default_preferences": defaults,
        "regime": {"state": regime_state, "comparable": regime_comparable},
        "min_sample": MIN_SAMPLE, "min_meaningful_delta": MIN_MEANINGFUL_RATE_DELTA,
    }
    summary["findings"] = derive_supported_findings(summary)
    return summary
