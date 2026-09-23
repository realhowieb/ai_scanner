"""Run 49 — canonical HSF IntelligenceView (pure adapter, presentation only).

ONE presentation-layer representation of HSF's EXISTING intelligence. It adapts
already-computed outputs — it recomputes and reinterprets nothing that would
change production semantics. It reuses the Run 40 `opportunity_view` engine (the
shared source of direction/lifecycle/priority/reasons already consumed by Market
Brief, Watchlist, and Historical Replay) and layers structured, traceable
explanation factors, existing scores (surfaced, not combined), scan-health,
research-evidence status, provenance and staleness.

Hard rules (Run 49): no new score, no recalibration, no threshold change.
`opportunity_score`, `prebreakout`, `ml_probability`, `alert_priority` are shown
side-by-side and NEVER merged into a composite. Unknown is `None`, never 0/False.
Scan health describes the SCAN, not per-stock prediction confidence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from analytics import opportunity_view as ov

SCHEMA_VERSION = "hsf-intelligence-view-1.0"

# Canonical product terminology (Task 4). Maps aliases → canonical; documents,
# does not merge distinct concepts.
TERMINOLOGY = {
    "bullish": "LONG", "bearish": "SHORT", "neutral": "NEUTRAL", "mixed": "MIXED",
    "strong": "STRONG", "high": "STRONG", "developing": "DEVELOPING",
    "watching": "DEVELOPING", "confirming": "CONFIRMING", "confirmed": "CONFIRMED",
    "warning": "CONFLICT", "supporting factor": "CONFIRMATION",
}

# What each existing number means (Task 11) — descriptive only, no combining.
SCORE_METADATA = {
    "opportunity_score": {"label": "Opportunity Score", "range": "0–100",
                          "higher_means": "stronger HSF composite setup (coherence, not a return prediction)",
                          "source": "ui.opportunities.build_opportunity_score", "version_key": "hsf_score"},
    "prebreakout": {"label": "PreBreakout", "range": "0–100 (%)",
                    "higher_means": "higher model-estimated pre-breakout likelihood",
                    "source": "ml_prebreakout", "calibrated": True, "version_key": "prebreakout_model"},
    "ml_probability": {"label": "AI Confidence", "range": "0–1",
                       "higher_means": "higher model confidence", "source": "scan.ai_confidence",
                       "version_key": "ai_confidence_model"},
    "alert_priority": {"label": "Alert Priority", "range": "HIGH/MEDIUM/LOW",
                       "higher_means": "how urgently to surface — an attention signal, NOT a return/probability claim",
                       "source": "analytics.opportunity_view", "predictive": False},
}

RESEARCH_EVIDENCE_LEVELS = ("UNVALIDATED", "PRELIMINARY", "MODERATE", "STRONG")
# Evidence promotion gate (Task 15) — reuses the Run 48 evidence framework.
EVIDENCE_PROMOTION_GATE = {
    "INSUFFICIENT": "no production recommendation",
    "UNVALIDATED": "no production recommendation",
    "PRELIMINARY": "hypothesis only",
    "MODERATE": "eligible for controlled experiment",
    "STRONG": "eligible for production-change proposal",
}


def _num(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _factor(code, label, source, value, severity):
    return {"code": code, "label": label, "source": source, "value": value,
            "severity": severity}


def build_factors(obs: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Structured supporting/caution factors traceable to real values (Task 5/6).

    Mirrors the Run 40 positive/risk reason thresholds but emits {code,label,
    source,value,severity} so every factor points at an actual system value.
    Nothing is emitted without a backing value — no unsupported explanations."""
    ind = obs.get("indicators") or {}
    models = obs.get("models") or {}
    dq = obs.get("data_quality") or {}
    supporting: List[Dict[str, Any]] = []
    caution: List[Dict[str, Any]] = []

    rvol = _num(ind.get("rvol"))
    if rvol is not None and rvol >= 1.5:
        supporting.append(_factor("RVOL_EXPANSION", f"RVOL {rvol:.1f}x", "indicators.rvol", rvol, "info"))
    elif rvol is not None and rvol < 1.0:
        caution.append(_factor("LOW_PARTICIPATION", f"Low RVOL {rvol:.1f}x", "indicators.rvol", rvol, "warn"))
    vsv = _num(ind.get("vs_vwap_pct"))
    if vsv is not None and vsv > 0:
        supporting.append(_factor("ABOVE_VWAP", "Above VWAP", "indicators.vs_vwap_pct", vsv, "info"))
    elif vsv is not None and vsv < 0:
        caution.append(_factor("BELOW_VWAP", "Below VWAP", "indicators.vs_vwap_pct", vsv, "warn"))
    if vsv is not None and vsv >= 4:
        caution.append(_factor("EXTENDED_ABOVE_VWAP", f"Extended above VWAP ({vsv:+.1f}%)",
                               "indicators.vs_vwap_pct", vsv, "warn"))
    adx = _num(ind.get("adx"))
    if adx is not None and adx >= 20:
        supporting.append(_factor("TRENDING", f"ADX {adx:.0f}", "indicators.adx", adx, "info"))
    elif adx is not None and adx < 15:
        caution.append(_factor("WEAK_TREND", f"Weak trend (ADX {adx:.0f})", "indicators.adx", adx, "warn"))
    chg = _num(ind.get("chg_pct"))
    if chg is not None and chg > 0:
        supporting.append(_factor("POSITIVE_MOMENTUM", f"Up {chg:+.1f}%", "indicators.chg_pct", chg, "info"))
    gap = _num(ind.get("gap_pct"))
    if gap is not None and abs(gap) >= 5:
        caution.append(_factor("LARGE_GAP", f"Large gap ({gap:+.1f}%)", "indicators.gap_pct", gap, "warn"))
    e9, e21 = _num(ind.get("ema9")), _num(ind.get("ema21"))
    if e9 is not None and e21 is not None:
        if e9 > e21:
            supporting.append(_factor("EMA_ALIGNMENT", "EMA9 above EMA21", "indicators.ema9/ema21", e9 - e21, "info"))
        else:
            caution.append(_factor("EMA_MISALIGNMENT", "EMA9 below EMA21", "indicators.ema9/ema21", e9 - e21, "warn"))
    prob = _num((models.get("prebreakout") or {}).get("probability"))
    if prob is not None:
        pct = prob * 100 if prob <= 1 else prob
        supporting.append(_factor("PREBREAKOUT_MODEL", f"PreBreakout {pct:.0f}%",
                                  "models.prebreakout.probability", pct,
                                  "info" if pct >= 60 else "low"))
    agree = ov.scanner_agreement(obs.get("scanners") or [])
    if agree["count"] >= 2:
        supporting.append(_factor("SCANNER_AGREEMENT", f"{agree['count']} scanners agree",
                                  "scanners", agree["count"], "info"))
    if ov.overall_direction(obs.get("scanners") or []) == "mixed":
        caution.append(_factor("CONFLICTING_DIRECTIONS", "Conflicting scanner directions",
                               "scanners.direction", None, "warn"))
    if dq.get("stale"):
        caution.append(_factor("STALE_DATA", "Stale data", "data_quality.stale", True, "warn"))
    if dq.get("fallback_used"):
        caution.append(_factor("PARTIAL_DATA", "Incomplete data (fallback)", "data_quality.fallback_used", True, "warn"))
    return {"supporting": supporting, "caution": caution}


def build_intelligence_view(
    obs: Dict[str, Any], *, prior_obs: Optional[Dict[str, Any]] = None,
    opportunity_score: Optional[float] = None, tier: Optional[str] = None,
    research_evidence: str = "UNVALIDATED",
) -> Dict[str, Any]:
    """Assemble the canonical IntelligenceView from an existing observation.

    Reuses opportunity_view for direction/lifecycle/priority/agreement/changes
    (identical semantics to Market Brief/Watchlist/Replay). Scores are surfaced
    from existing values only; absent values are None (UNKNOWN), never 0."""
    view = ov.build_opportunity_view(obs, prior_obs=prior_obs)
    models = obs.get("models") or {}
    ctx = obs.get("market_context") or {}
    factors = build_factors(obs)
    direction = view.get("direction")

    scores = {
        "opportunity_score": _num(opportunity_score),
        "prebreakout": _num((models.get("prebreakout") or {}).get("probability")),
        "ml_probability": _num((models.get("ai_confidence") or {}).get("confidence")),
        "alert_priority": view.get("alert_priority"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": view.get("symbol"),
        "timestamp": view.get("timestamp"),
        # WHAT IS HAPPENING
        "direction": TERMINOLOGY.get(str(direction).lower(), str(direction).upper()) if direction else None,
        "lifecycle": view.get("lifecycle_state"),
        "tier": tier,  # surfaced from existing status only; None when unknown
        # HOW IMPORTANT / scores (side-by-side, never combined)
        "scores": scores,
        "score_metadata": SCORE_METADATA,
        # WHY / WHAT RISK
        "agreement": {"count": view.get("scanner_count"), "scanners": view.get("scanner_names")},
        "confirmations": factors["supporting"],
        "conflicts": factors["caution"],
        "supporting_factors": factors["supporting"],
        "caution_factors": factors["caution"],
        "primary_setup": view.get("primary_setup"),
        "changes_since_prior": view.get("changes_since_prior"),
        "next_confirmation": _next_confirmation(factors, view),
        # HOW FRESH / TRUSTWORTHY (scan-level, NOT prediction confidence)
        "scan_health": {
            "coverage_health": ctx.get("coverage_health"),
            "note": "Describes the SCAN's market coverage — not the reliability of "
                    "this stock's signal.",
        },
        "research_evidence": {
            "level": research_evidence if research_evidence in RESEARCH_EVIDENCE_LEVELS else "UNVALIDATED",
            "note": EVIDENCE_PROMOTION_GATE.get(research_evidence, EVIDENCE_PROMOTION_GATE["UNVALIDATED"]),
        },
        "freshness": view.get("freshness"),
        "staleness": {
            "intelligence_as_of": view.get("timestamp"),
            "market_data_as_of": obs.get("scan_timestamp") or view.get("timestamp"),
            "scan_run_id": ctx.get("scan_id"),
        },
        "provenance": obs.get("versions"),
    }


def _next_confirmation(factors, view) -> Optional[str]:
    """A single, deterministic 'what to watch next' from existing state — the
    strongest missing confirmation, else None. No fabricated guidance."""
    have = {f["code"] for f in factors["supporting"]}
    if view.get("direction") in ("bullish", "long") and "ABOVE_VWAP" not in have:
        return "Reclaim VWAP"
    if "RVOL_EXPANSION" not in have:
        return "Volume expansion (RVOL)"
    if "TRENDING" not in have:
        return "Trend strength (ADX ≥ 20)"
    return None


def to_dict(view: Dict[str, Any]) -> Dict[str, Any]:
    """Stable, deterministic, null-safe serialization (Task 24). The view is
    already a plain dict; this guarantees the schema key and ordering are stable."""
    out = {"schema_version": view.get("schema_version", SCHEMA_VERSION)}
    out.update({k: view[k] for k in sorted(view) if k != "schema_version"})
    return out


# --- Run 48 research-readiness check (Task 14/15) ----------------------------
def run48_readiness(observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Determine whether accumulated research data is sufficient to rerun Run 48
    meaningfully. Reuses the Run 48 evidence framework (no second statistics)."""
    from analytics.signal_effectiveness import (
        HORIZONS,
        dataset_evidence_summary,
        observation_to_record,
    )

    records = [observation_to_record(o) for o in observations
               if (o.get("market_context") or {}).get("coverage_health") == "HEALTHY"]
    summary = dataset_evidence_summary(records)
    days = len({str(o.get("timestamp"))[:10] for o in observations if o.get("timestamp")})
    paired = {h: min(
        sum(1 for r in records if r["cohort"] == "CANDIDATE" and (r["outcomes"] or {}).get(h)),
        sum(1 for r in records if r["cohort"] == "CONTROL" and (r["outcomes"] or {}).get(h)),
    ) for h in HORIZONS}
    level = summary["evidence_level"]
    rec = ("RERUN_RUN48" if level in ("MODERATE", "STRONG")
           else "CONTINUE_ACCUMULATING")
    return {
        "schema": "hsf-run48-readiness-1.0",
        "trading_days": days,
        "total_research_observations": len(observations),
        "by_cohort": summary["by_cohort"],
        "matured_paired_by_horizon": paired,
        "healthy_records": len(records),
        "evidence_level": level,
        "promotion_gate": EVIDENCE_PROMOTION_GATE.get(level, "no production recommendation"),
        "recommendation": rec,
    }
