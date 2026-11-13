"""Read-only: reconcile missing ML v4 core inputs with saved prediction-time values.

Part 1 (development slice, the 1,003 Sep 14-Oct 21 training signal-days of
hsf-ml-2026-11-13-v1, verified by hash before anything else) checks the rows
where the canonical scan join left a core input empty but the PreBreakout
prediction record saved an observed (not defaulted) value:
  * timing: was the saved value computed before the pick was saved, from the
    pick's own scan?
  * units: same scale as the canonical column (overlap ratios, independent
    cross-checks against values stored on the pick itself)?
  * clipping: did the model's preprocessing clip the saved value?
  * the disagreements in the overlap: size and cause.

Part 2 (Sep 14 to today, features only) reports the same recoverability by
week plus RSI-14 and SPY/SMA20 regime coverage from the joined scan record.

Prints counts, rates and quantiles only: no tickers, no labels, no outcome
aggregates. Never reads on or after the reserved forward window (Dec 23).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESERVED_START = dt.date(2026, 12, 23)
# PreBreakout vector name -> canonical research column.
CORE = {
    "Trend10D%": "trend_10d_pct", "Trend20D%": "trend_20d_pct", "VolRel20": "rvol_20",
    "DollarVol20": "dollar_vol_20", "BreakoutScore": "scanner_breakout_score", "GapPct": "scan_gap_pct",
}
EXTRA_META = ("'rsi_14', record->'research_metadata'->'rsi_14', "
              "'market_regime_at_observation', record->'research_metadata'->'market_regime_at_observation', "
              "'market_regime_source', record->'research_metadata'->'market_regime_source', "
              "'market_regime_version', record->'research_metadata'->'market_regime_version', "
              "'market_regime_inputs', record->'research_metadata'->'market_regime_inputs', ")


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) else math.nan


def _json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return {}
    return v if isinstance(v, dict) else {}


def _q(values, qs=(0.01, 0.1, 0.5, 0.9, 0.99)):
    v = sorted(x for x in values if math.isfinite(x))
    if not v:
        return None
    return {f"p{int(q*100)}": round(v[min(len(v)-1, int(q*(len(v)-1)+0.5))], 6) for q in qs} | {"n": len(v)}


def _vector(raw_row):
    """Observed (non-defaulted) PreBreakout inputs + their provenance, or None."""
    sig = _json(raw_row.get("raw_signal"))
    p = (sig.get("models") or {}).get("prebreakout") or {}
    names, values, mask = (p.get(k) for k in ("feature_names", "feature_values", "default_mask"))
    if p.get("status") != "captured" or not all(isinstance(x, list) for x in (names, values, mask)) \
            or not len(names) == len(values) == len(mask):
        return None
    observed = {n: _num(v) for n, v, m in zip(names, values, mask) if m is False}
    return {"values": observed, "p": p, "row_source_scan_id": sig.get("source_scan_id")}


def _clip_bounds(p):
    plan = p.get("preprocessing") or {}
    return {c: b for c, b in (plan.get("clip_thresholds") or {}).items() if c in CORE}


def _minutes(a, b):
    from analytics import research_dataset as rd
    a, b = rd.to_dt(a), rd.to_dt(b)
    if a is None or b is None:
        return None
    return (a - b).total_seconds() / 60.0


def _lag_bucket(m):
    if m is None:
        return "unavailable"
    if m < -0.5:
        return "AFTER_PICK"
    for lim, name in ((5, "0-5m"), (60, "5-60m"), (180, "1-3h")):
        if m <= lim:
            return name
    return ">3h"


def reconcile(rows, raw_rows, scans_by_id):
    """rows: audit rows; raw_rows: opportunity rows (same ids)."""
    from analytics import research_dataset as rd
    raw_by_id = {int(r["id"]): r for r in raw_rows}
    out = {"rows": len(rows), "with_captured_vector": 0, "fields": {}, "recoverable_rows": 0,
           "timing": {}, "cross_checks": {}, "clip": {}, "disagreements": []}
    per = {c: Counter() for c in CORE.values()}
    ratio = defaultdict(list)
    rec_vals, can_vals = defaultdict(list), defaultdict(list)
    timing = {k: Counter() for k in ("input_ts_vs_pick", "inferred_vs_pick", "source_scan_vs_pick",
                                     "source_scan_matches_pick_scan", "join_status", "entry_week")}
    cross = {"gap_vs_pick_indicator": Counter(), "breakout_vs_pick_setup_score": Counter()}
    clip_seen, at_bound = Counter(), Counter()
    recoverable_ids = set()
    case = 0
    for r in rows:
        raw = raw_by_id.get(r.observation_id)
        vec = _vector(raw) if raw else None
        if vec is None:
            for c in CORE.values():
                per[c]["no_vector_" + ("present" if math.isfinite(_num(r.features.get(c))) else "missing")] += 1
            continue
        out["with_captured_vector"] += 1
        bounds = _clip_bounds(vec["p"])
        for c in bounds:
            clip_seen[c] += 1
        row_recoverable = False
        for name, col in CORE.items():
            v, cur = vec["values"].get(name, math.nan), _num(r.features.get(col))
            if not math.isfinite(v):
                per[col]["vector_defaulted_or_absent"] += 1
                continue
            b = bounds.get(name)
            if b and (math.isclose(v, float(b["lower"])) or math.isclose(v, float(b["upper"]))):
                at_bound[col] += 1
            if not math.isfinite(cur):
                per[col]["recoverable"] += 1
                rec_vals[col].append(v)
                row_recoverable = True
                continue
            per[col]["overlap"] += 1
            can_vals[col].append(cur)
            if cur != 0:
                ratio[col].append(v / cur)
            if not math.isclose(v, cur, rel_tol=1e-4, abs_tol=1e-6):
                per[col]["disagree"] += 1
                case += 1
                scan_ts = (r.join_status, r.extra.get("scan_ts"))
                out["disagreements"].append({
                    "case": case, "field": col, "entry_week": "%d-W%02d" % r.entry_day.isocalendar()[:2],
                    "rel_diff_pct": round(100 * (v - cur) / abs(cur), 3) if cur else None,
                    "abs_diff": round(v - cur, 4),
                    "vector_scan_minus_joined_scan_min": _minutes(vec["p"].get("source_scan_id"), r.extra.get("scan_ts")),
                    "joined_scan_lag_min": round(r.join_lag_s / 60, 1) if r.join_lag_s is not None else None,
                    "join_link": r.extra.get("join_link"), "join_status": scan_ts[0],
                    "vector_input_minus_pick_min": _minutes(vec["p"].get("input_timestamp"), r.observed_at),
                    "at_clip_bound": bool(b and (math.isclose(v, float(b["lower"])) or math.isclose(v, float(b["upper"])))),
                })
        if not row_recoverable:
            continue
        recoverable_ids.add(r.observation_id)
        p = vec["p"]
        timing["input_ts_vs_pick"][_lag_bucket(_minutes(r.observed_at, p.get("input_timestamp")))] += 1
        timing["inferred_vs_pick"][_lag_bucket(_minutes(r.observed_at, p.get("inferred_at")))] += 1
        timing["source_scan_vs_pick"][_lag_bucket(_minutes(r.observed_at, p.get("source_scan_id")))] += 1
        same = rd.to_dt(p.get("source_scan_id")) is not None and \
            rd.to_dt(p.get("source_scan_id")) == rd.to_dt(vec["row_source_scan_id"])
        timing["source_scan_matches_pick_scan"]["same" if same else (
            "pick_has_no_scan_id" if vec["row_source_scan_id"] is None else "different")] += 1
        timing["join_status"][str(r.join_status)] += 1
        timing["entry_week"]["%d-W%02d" % r.entry_day.isocalendar()[:2]] += 1
        ind = _json(raw.get("indicators"))
        g, gv = _num(ind.get("gap_pct")), vec["values"].get("GapPct", math.nan)
        cross["gap_vs_pick_indicator"]["no_pick_value" if not math.isfinite(g) else (
            "agree" if math.isclose(g, gv, rel_tol=1e-3, abs_tol=1e-3) else "differ")] += 1
        s, sv = _num(raw.get("setup_score")), vec["values"].get("BreakoutScore", math.nan)
        cross["breakout_vs_pick_setup_score"]["no_pick_value" if not math.isfinite(s) else (
            "agree" if math.isclose(s, sv, rel_tol=1e-3, abs_tol=1e-3) else "differ")] += 1
    out["recoverable_rows"] = len(recoverable_ids)
    out["fields"] = {c: dict(per[c]) for c in CORE.values()}
    out["units"] = {c: {"overlap_ratio_vector_over_canonical": _q(ratio[c]),
                        "recoverable_values": _q(rec_vals[c]), "canonical_overlap_values": _q(can_vals[c])}
                    for c in CORE.values()}
    out["timing"] = {k: dict(v) for k, v in timing.items()}
    out["cross_checks"] = {k: dict(v) for k, v in cross.items()}
    out["clip"] = {"vectors_with_clip_rule": dict(clip_seen), "values_at_clip_bound": dict(at_bound)}
    return out, recoverable_ids


def _annotate_scan(rows, records):
    by_id = {int(rec["observation"]["observation_id"]): rec for rec in records}
    for r in rows:
        j = (by_id.get(r.observation_id) or {}).get("features")
        j = j.join if j is not None else {}
        r.extra["scan_ts"] = j.get("scan_timestamp")
        r.extra["join_link"] = j.get("link")
        r.extra["scan_observation_id"] = j.get("scan_observation_id")


def meta_coverage(rows, scans):
    """RSI / regime from the joined scheduled scan record (signal-day rows)."""
    from analytics import research_dataset as rd
    meta_by_scan = {}
    for s in scans:
        v = rd.scan_record_view(s)
        meta_by_scan[v["scan_observation_id"]] = (v["record"].get("research_metadata") or {})
    weekly = defaultdict(Counter)
    rsi_vals, regime_labels, sign_ok, versions = [], Counter(), Counter(), Counter()
    by_scan_regime = defaultdict(set)
    for r in rows:
        wk = "%d-W%02d" % r.entry_day.isocalendar()[:2]
        weekly[wk]["rows"] += 1
        m = meta_by_scan.get(str(r.extra.get("scan_observation_id") or "")) if r.extra.get("scan_observation_id") else None
        if m is None:
            weekly[wk]["no_joined_scan"] += 1
            continue
        rsi = _num(m.get("rsi_14"))
        if math.isfinite(rsi):
            weekly[wk]["rsi"] += 1
            rsi_vals.append(rsi)
        lab = m.get("market_regime_at_observation")
        if lab:
            weekly[wk]["regime"] += 1
            regime_labels[str(lab)] += 1
            versions[str(m.get("market_regime_version"))] += 1
            pct = _num((m.get("market_regime_inputs") or {}).get("spy_vs_sma20_pct"))
            if math.isfinite(pct):
                sign_ok["consistent" if (pct >= 0) == (lab == "SPY_ABOVE_SMA20") else "inconsistent"] += 1
            else:
                sign_ok["no_pct"] += 1
            by_scan_regime[str(r.extra.get("scan_ts"))].add(str(lab))
    return {"weekly": {k: dict(v) for k, v in sorted(weekly.items())},
            "rsi_quantiles": _q(rsi_vals), "rsi_out_of_0_100": sum(1 for x in rsi_vals if not 0 <= x <= 100),
            "regime_labels": dict(regime_labels), "regime_versions": dict(versions),
            "regime_label_vs_pct_sign": dict(sign_ok),
            "scans_with_mixed_regime_labels": sum(1 for v in by_scan_regime.values() if len(v) > 1),
            "distinct_scans": len(by_scan_regime)}


def main():
    import db.research_datasets as D
    from analytics import ml_v3_audit as A
    from analytics import research_dataset as rd
    from scripts.ml_development_quality import load_scope

    # Read RSI / regime with the scan record (diagnostic-only projection).
    D._SCAN_JSON_PG = D._SCAN_JSON_PG.replace("'row_features',", EXTRA_META + "'row_features',", 1)

    result = {"labels_or_outcomes_printed": False, "reserved_start": RESERVED_START.isoformat()}
    scope = json.loads((ROOT / "docs/ml_development_quality_scope_2026-11-13.json").read_text())
    rows, raw = load_scope(scope)  # verifies membership hash + slice fingerprint
    lo = dt.datetime.fromisoformat(scope["development_start"]).replace(tzinfo=dt.timezone.utc)
    hi = dt.datetime.fromisoformat(scope["development_end_exclusive"]).replace(tzinfo=dt.timezone.utc)
    scans = D.fetch_scan_records({r["ticker"] for r in raw}, lo - rd.MAX_SCAN_LAG - dt.timedelta(hours=1), hi)
    _annotate_scan(rows, rd.build_records(raw, scans))
    result["development_slice"], _ = reconcile(rows, raw, None)

    # Part 2: Sep 14 -> today, feature-only, signal-day.
    end = min(dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1), RESERVED_START)
    lo2 = dt.datetime(2026, 9, 14, tzinfo=dt.timezone.utc)
    hi2 = dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc)
    raw2 = D.fetch_opportunity_rows(lo2, hi2)
    if any(rd.to_dt(r["fired_at"]).date() >= RESERVED_START for r in raw2):
        raise SystemExit("reserved window reached; refusing")
    scans2 = D.fetch_scan_records({r["ticker"] for r in raw2}, lo2 - rd.MAX_SCAN_LAG - dt.timedelta(hours=1), hi2)
    recs2 = rd.build_records(raw2, scans2)
    rows2 = A.build_audit_rows(recs2, raw2, unit="signal_day")
    _annotate_scan(rows2, recs2)
    full, _ = reconcile(rows2, raw2, None)
    full.pop("disagreements_detail", None)
    full["disagreement_count"] = len(full.pop("disagreements"))
    weekly = defaultdict(Counter)
    for r in rows2:
        wk = "%d-W%02d" % r.entry_day.isocalendar()[:2]
        weekly[wk]["rows"] += 1
        weekly[wk]["all6_canonical"] += all(math.isfinite(_num(r.features.get(c))) for c in CORE.values())
    result["sep14_to_today"] = {"window": [lo2.date().isoformat(), end.isoformat()], **full,
                                "weekly_all6_canonical": {k: dict(v) for k, v in sorted(weekly.items())},
                                "rsi_regime": meta_coverage([r for r in rows2 if r.entry_day >= dt.date(2026, 10, 26)], scans2)}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
