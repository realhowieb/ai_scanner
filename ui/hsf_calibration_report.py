"""Admin/internal HSF Score calibration report (read-only).

Renders the calibration diagnostics from real stored outcomes. No control here
changes production weights — this is evidence, not a tuning knob. Shows honest
'insufficient data' states while outcomes are still maturing.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _rate(r) -> str:
    return f"{r*100:.0f}%" if r is not None else "—"


def render_hsf_calibration_report(days_back: int = 365) -> None:
    """Read-only calibration report. Safe when DB/data is unavailable."""
    if st is None:
        return
    try:
        from analytics import hsf_calibration as hc

        ds = hc.build_calibration_dataset(days_back=days_back)
    except Exception:
        st.caption("Calibration data is unavailable right now.")
        return

    st.markdown("#### 🔬 HSF Score calibration")
    lo, hi = ds["date_range"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Frozen", ds["n_total"])
    c2.metric("Matured", ds["n_matured"])
    c3.metric("Pending", ds["n_pending"])
    st.caption(f"Range: {lo} → {hi} · versions: {ds['versions']}")
    for w in ds["quality"]:
        st.caption(f"⚠ {w}")

    matured = ds["matured"]
    if not matured:
        st.info("No matured HSF opportunities yet — the report populates as "
                "forward windows complete (infrastructure is in place).")
        return

    st.markdown("**Score buckets** — positive outcome = reached +4% within 5D")
    buckets = hc.summarize_score_buckets(matured)
    st.dataframe(
        [{"Bucket": b["bucket"], "Matured": b["n_matured"],
          "Positive %": _rate(b["positive_rate"]),
          "Median MFE": b["median_mfe_5d"], "Median MAE": b["median_mae_5d"],
          "Confidence": b["confidence"]} for b in buckets],
        hide_index=True, width="stretch",
    )
    mono = hc.evaluate_monotonicity(buckets)
    st.caption(f"Higher score → better outcome? **{mono['monotonic']}** "
               f"(rank corr {mono.get('rank_correlation')}) · {mono['confidence']}")

    st.markdown("**Status**")
    st.dataframe(
        [{"Status": s["status"], "Matured": s["n_matured"],
          "Positive %": _rate(s["positive_rate"]),
          "Median MFE": s["median_mfe_5d"], "Confidence": s["confidence"]}
         for s in hc.summarize_status_performance(matured)],
        hide_index=True, width="stretch",
    )

    sig = hc.summarize_signal_types(matured)
    if sig:
        st.markdown("**Signal types**")
        st.dataframe(
            [{"Signal": s["signal_type"], "Matured": s["n_matured"],
              "Positive %": _rate(s["positive_rate"]), "Confidence": s["confidence"]}
             for s in sig],
            hide_index=True, width="stretch",
        )

    combos = hc.analyze_signal_combinations(matured)
    if combos:
        st.markdown("**Common combinations** (n≥10)")
        st.dataframe(
            [{"Combination": c["combination"], "Matured": c["n_matured"],
              "Positive %": _rate(c["positive_rate"])} for c in combos],
            hide_index=True, width="stretch",
        )

    st.markdown("**Baseline discrimination** (AUC vs positive outcome)")
    st.dataframe(
        [{"Ranker": b["ranker"], "AUC": b["auc"], "n": b["n"], "Confidence": b["confidence"]}
         for b in hc.compare_baselines(matured)],
        hide_index=True, width="stretch",
    )

    cal = hc.evaluate_calibration(matured)
    st.caption(f"Calibration Brier: **{cal['brier']}** · n={cal['n']} · {cal['confidence']}")
    dist = hc.score_distribution(ds["records"])
    if dist.get("n"):
        st.caption(f"Score distribution — min {dist['min']} · median {dist['median']} · "
                   f"mean {dist['mean']} · max {dist['max']} · buckets {dist['bucket_counts']}")
