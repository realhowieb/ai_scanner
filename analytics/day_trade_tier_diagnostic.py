"""Read-only quality-tier diagnostics for historical Day Trader observations."""
from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from analytics import day_trade_intel as di

PROFILES = {
    "production_v1": (65.0, 0.75),
    "rejected_v2": (55.0, 0.70),
}
INPUTS = ("chg_pct", "gap_pct", "rvol", "vs_vwap_pct", "adx", "supertrend_direction", "ewo")
CONFLICT_INPUTS = {
    "Mixed trend signals": ("supertrend_direction", "vs_vwap_pct"),
    "Losing VWAP": ("chg_pct", "vs_vwap_pct"),
    "Momentum disagreement": ("ewo", "chg_pct"),
    "Gap fading": ("gap_pct", "chg_pct"),
    "Low participation": ("rvol",),
    "Weak trend strength": ("adx",),
}
SCORE_FACTORS = {
    "chg_pct": "momentum", "gap_pct": "gap", "rvol": "rvol",
    "vs_vwap_pct": "vwap", "adx": "adx",
    "supertrend_direction": "agreement", "ewo": "agreement",
}


def score_components(features: dict[str, Any], *, profile: str) -> dict[str, Any]:
    """Explain the score formula without calling or changing production scoring."""
    agreement = di._agreement(features)
    if agreement is None:
        return {}
    v2 = profile == "rejected_v2"
    adx_full, rvol_full, vwap_full, mom_full, gap_full = (55, 5, 2, 5, 5) if v2 else (40, 3, 1, 3, 3)
    votes = [v for v in di._direction_votes(features).values() if v]
    completeness = 0.4 + 0.6 * di._clamp01(len(votes) / 5) if v2 else 1.0
    subs = {"agreement": di._clamp01((agreement - 0.5) / 0.5) * completeness}
    values = {name: di._num(features.get(name)) for name in ("adx", "rvol", "vs_vwap_pct", "chg_pct", "gap_pct")}
    if values["adx"] is not None:
        subs["adx"] = di._clamp01((values["adx"] - 15) / (adx_full - 15))
    if values["rvol"] is not None:
        subs["rvol"] = di._clamp01((values["rvol"] - 1) / (rvol_full - 1))
    if values["vs_vwap_pct"] is not None:
        subs["vwap"] = di._clamp01(abs(values["vs_vwap_pct"]) / vwap_full)
    if values["chg_pct"] is not None:
        subs["momentum"] = di._clamp01(abs(values["chg_pct"]) / mom_full)
    gap, chg = values["gap_pct"], values["chg_pct"]
    if gap is not None and chg is not None and abs(gap) >= 0.05 and (gap > 0) == (chg > 0):
        subs["gap"] = di._clamp01(abs(gap) / gap_full)
    total_weight = sum(di._WEIGHTS[name] for name in subs)
    raw = sum(di._WEIGHTS[name] * value for name, value in subs.items()) / total_weight * 100
    penalty = min(len(di.day_trade_conflicts(features)) * 5, 25)
    return {"subscores": subs, "capped": [name for name, value in subs.items() if value == 1],
            "available_weight": total_weight, "raw_score": raw, "conflict_penalty": penalty,
            "calculated_score": round(max(0.0, min(100.0, raw - penalty)), 1),
            "signal_pattern": {name: di._direction_votes(features).get(name) for name in ("vwap", "supertrend", "ewo", "momentum", "gap")}}


def diagnose_row(features: dict[str, Any], *, profile: str = "production_v1",
                 score: float | None = None, quality: str | None = None) -> dict[str, Any]:
    """Use the scored observation and its original inputs; no future outcomes."""
    threshold, agreement_threshold = PROFILES[profile]
    votes = [vote for vote in di._direction_votes(features).values() if vote]
    direction = di.classify_day_trade_direction(features)
    agreement = di._agreement(features)
    components = score_components(features, profile=profile)
    score = (components.get("calculated_score") if profile == "rejected_v2"
             else di.score_day_trade_setup(features)) if score is None else score
    conflicts = di.day_trade_conflicts(features)
    adx, rvol = di._num(features.get("adx")), di._num(features.get("rvol"))
    gates = {
        "score_gate": score is not None and score >= threshold,
        "agreement_gate": agreement is not None and agreement >= agreement_threshold,
        "confirmation_gate": (adx is not None and adx >= 20) or (rvol is not None and rvol >= 1.5),
        "conflict_gate": len(conflicts) <= 1,
    }
    if quality is None and profile == "rejected_v2":
        quality = ("insufficient" if agreement is None or score is None else
                   "strong" if all(gates.values()) else
                   "weak" if score < 35 or agreement < 0.55 or
                   (rvol is not None and rvol < 1) or len(conflicts) >= 3 else "developing")
    return {
        "score": score, "direction": direction, "agreement": agreement,
        "directional_signals": len(votes),
        "agreeing_signals": max(sum(v > 0 for v in votes), sum(v < 0 for v in votes)) if votes else 0,
        "adx": adx, "rvol": rvol,
        "adx_confirmed": adx is not None and adx >= 20,
        "rvol_confirmed": rvol is not None and rvol >= 1.5,
        "confirmation_status": ("both" if adx is not None and adx >= 20 and rvol is not None and rvol >= 1.5
                                else "adx_only" if adx is not None and adx >= 20
                                else "rvol_only" if rvol is not None and rvol >= 1.5 else "neither"),
        "conflict_count": len(conflicts), "conflicts": conflicts,
        "quality": quality if quality is not None else di.classify_setup_quality(features),
        "gates": gates,
        "score_components": components,
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(min(r["conflict_count"], 4) for r in rows)
    return {str(k) if k < 4 else "4+": {"count": counts[k],
            "pct": counts[k] / len(rows) if rows else None} for k in range(5)}


def _agreement_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.55:
        return "<0.55"
    if value < 0.70:
        return "0.55-0.69"
    if value < 0.80:
        return "0.70-0.79"
    if value < 1:
        return "0.80-0.99"
    return "1.00"


def summarize(rows: list[dict[str, Any]], *, profile: str = "production_v1") -> dict[str, Any]:
    """Aggregate precomputed row diagnostics without selecting observations by outcome."""
    directional = [r for r in rows if r["direction"] in ("bullish", "bearish")]
    n = len(directional)
    names = ("score_gate", "agreement_gate", "confirmation_gate", "conflict_gate")
    if not n:
        return {"status": "NO_OBSERVATIONS", "profile": profile, "directional_n": 0,
                "reason": "Original per-observation indicator inputs are required."}
    gate_pass = {name: sum(r["gates"][name] for r in directional) for name in names}
    blockers = {name: {"count": n - count, "pct": (n - count) / n}
                for name, count in gate_pass.items()}
    combinations = {
        "score_agreement_pass_conflict_fails": sum(r["gates"]["score_gate"] and r["gates"]["agreement_gate"] and not r["gates"]["conflict_gate"] for r in directional),
        "score_confirmation_pass_conflict_fails": sum(r["gates"]["score_gate"] and r["gates"]["confirmation_gate"] and not r["gates"]["conflict_gate"] for r in directional),
        "only_conflict_fails": sum(all(r["gates"][k] for k in names if k != "conflict_gate") and not r["gates"]["conflict_gate"] for r in directional),
        "only_confirmation_fails": sum(all(r["gates"][k] for k in names if k != "confirmation_gate") and not r["gates"]["confirmation_gate"] for r in directional),
        "multiple_fail": sum(sum(not r["gates"][k] for k in names) > 1 for r in directional),
    }
    funnel = {}
    for i, name in enumerate(names, 1):
        funnel[name] = sum(all(r["gates"][k] for k in names[:i]) for r in directional)
    sensitivity = {"current": sum(all(r["gates"].values()) for r in directional)}
    for removed in names:
        sensitivity[f"without_{removed}"] = sum(all(r["gates"][k] for k in names if k != removed) for r in directional)
    for limit in range(4):
        sensitivity[f"conflicts_le_{limit}"] = sum(
            all(r["gates"][k] for k in names if k != "conflict_gate") and r["conflict_count"] <= limit
            for r in directional)
    groups = {"all": directional, "score_40_59": [r for r in directional if r["score"] is not None and 40 <= r["score"] < 60],
              "score_60_69": [r for r in directional if r["score"] is not None and 60 <= r["score"] < 70],
              "score_70_79": [r for r in directional if r["score"] is not None and 70 <= r["score"] < 80],
              "bullish": [r for r in directional if r["direction"] == "bullish"],
              "bearish": [r for r in directional if r["direction"] == "bearish"]}
    conflict_frequency = {}
    for label, inputs in CONFLICT_INPUTS.items():
        matched = [r for r in directional if label in r["conflicts"]]
        conflict_frequency[label] = {
            "count": len(matched), "pct": len(matched) / n,
            "score_ge_60": sum(r["score"] is not None and r["score"] >= 60 for r in matched),
            "score_ge_70": sum(r["score"] is not None and r["score"] >= 70 for r in matched),
            "bullish": sum(r["direction"] == "bullish" for r in matched),
            "bearish": sum(r["direction"] == "bearish" for r in matched),
            "indicator_inputs": inputs,
            "also_score_factors": sorted({SCORE_FACTORS[i] for i in inputs}),
        }
    agreement = {}
    for bucket in ("<0.55", "0.55-0.69", "0.70-0.79", "0.80-0.99", "1.00"):
        selected = [r for r in directional if _agreement_bucket(r["agreement"]) == bucket]
        agreement[bucket] = {"count": len(selected), "conflicts": _distribution(selected),
                             "direction": dict(Counter(r["direction"] for r in selected)),
                             "confirmation": dict(Counter(r["confirmation_status"] for r in selected)),
                             "score_bucket": dict(Counter(int(r["score"] // 10) * 10 for r in selected if r["score"] is not None))}
    confirmation = {}
    for status in ("adx_only", "rvol_only", "both", "neither"):
        selected = [r for r in directional if r["confirmation_status"] == status]
        confirmation[status] = {"count": len(selected),
                                "median_score": median(r["score"] for r in selected if r["score"] is not None) if any(r["score"] is not None for r in selected) else None,
                                "median_conflicts": median(r["conflict_count"] for r in selected) if selected else None,
                                "quality": dict(Counter(r["quality"] for r in selected))}
    pileup = [r for r in directional if r["score"] is not None and abs(r["score"] - 77.1) <= 0.05]
    patterns = Counter(str(r["score_components"].get("signal_pattern")) for r in pileup)
    subscore_summary = {}
    for name in di._WEIGHTS:
        vals = [r["score_components"]["subscores"][name] for r in pileup
                if name in r["score_components"].get("subscores", {})]
        subscore_summary[name] = {"present": len(vals), "median": median(vals) if vals else None,
                                  "capped": sum(value == 1 for value in vals)}
    return {"status": "OK", "profile": profile, "directional_n": n,
            "quality": dict(Counter(r["quality"] for r in directional)),
            "gate_pass": gate_pass, "gate_funnel": funnel,
            "rejection_reasons": blockers, "rejection_combinations": combinations,
            "conflict_distribution": {name: _distribution(group) for name, group in groups.items()},
            "conflict_frequency": dict(sorted(conflict_frequency.items(), key=lambda kv: -kv[1]["count"])),
            "agreement": agreement, "confirmation": confirmation,
            "gate_sensitivity": sensitivity,
            "score_77_1": {"count": len(pileup),
                           "directional_signal_count": dict(Counter(r["directional_signals"] for r in pileup)),
                           "agreement": dict(Counter(str(r["agreement"]) for r in pileup)),
                           "conflicts": _distribution(pileup),
                           "subscores": subscore_summary,
                           "available_weight": dict(Counter(str(r["score_components"].get("available_weight")) for r in pileup)),
                           "raw_score_median": median(r["score_components"]["raw_score"] for r in pileup) if pileup else None,
                           "penalties": dict(Counter(r["score_components"].get("conflict_penalty") for r in pileup)),
                           "common_signal_patterns": patterns.most_common(10)}}
