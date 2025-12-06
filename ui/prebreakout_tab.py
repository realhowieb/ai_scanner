"""
UI components for the 'Early Breakout Candidates' tab.

This tab surfaces model-based pre-breakout probabilities for the latest scan
stored in `st.session_state.results_df`, assuming the scan results were
scored by ml_prebreakout.score_prebreakout and contain a 'PreBreakoutProb'
column.
"""

from typing import Optional

import pandas as pd
import streamlit as st
from ml_prebreakout import load_prebreakout_model, train_prebreakout_model


def _get_latest_results_df() -> Optional[pd.DataFrame]:
    """Helper to safely fetch the latest results dataframe from session_state."""
    df = st.session_state.get("results_df")
    if df is None:
        return None
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def render_prebreakout_tab() -> None:
    """
    Render the 'Early Breakout Candidates' tab.

    Expects that:
      - st.session_state["results_df"] is a pandas DataFrame of the latest scan
      - If a model is available, that DataFrame includes a 'PreBreakoutProb' column
        with values in [0.0, 1.0].
    """
    st.markdown("### 🔮 Early Breakout Candidates (Model-based)")

    # --- DEBUG: Check run history ---
    if st.button("🔍 Debug: Check DB history"):
        from ml_prebreakout import load_run_history

        df_debug = load_run_history(days_back=365)
        st.write(f"Rows loaded from history: {len(df_debug)}")

        if len(df_debug) == 0:
            st.warning("⚠️ No runs found in DB. Run SP500/NASDAQ/Combo scans first.")
        else:
            st.success("✅ History loaded successfully!")
            st.dataframe(df_debug.head(50), use_container_width=True)

    # --- DEBUG: Check raw runs from db.runs.list_runs ---
    if st.button("🔎 Debug: Raw runs from DB"):
        from db.runs import list_runs

        runs = list_runs(limit=5)
        st.write(f"Raw runs returned: {len(runs)}")
        st.json(runs)

    # --- Model status + training controls ---
    with st.expander("🧠 Model status & training", expanded=False):
        bundle = load_prebreakout_model()
        if bundle:
            features = bundle.get("features", [])
            feature_preview = ", ".join(features[:8])
            if len(features) > 8:
                feature_preview += ", ..."

            st.success(
                f"Current model loaded.\n\n"
                f"- AUC: **{bundle.get('auc', 0):.3f}**\n"
                f"- Trained at: **{bundle.get('trained_at', 'unknown')}**\n"
                f"- Features: `{feature_preview}`"
            )
        else:
            st.warning(
                "No pre-breakout model is currently loaded. "
                "Train a new model using your historical runs."
            )

        if st.button("🚀 Train / Refresh model from DB history", use_container_width=True):
            with st.spinner("Training pre-breakout model from DB history..."):
                trained_bundle = train_prebreakout_model(
                    days_back=90,
                    horizon_scans=3,
                )
            if trained_bundle:
                st.success(
                    f"Model trained! AUC={trained_bundle.get('auc', 0):.3f}, "
                    f"trained_at={trained_bundle.get('trained_at', 'unknown')}"
                )
            else:
                st.error(
                    "Training failed. Check the app logs for details "
                    "(e.g., missing history or missing IsBreakout column)."
                )

    st.caption(
        "These candidates are ranked by a model trained on your historical scans "
        "to detect pre-breakout patterns (rising trend, volume pressure, and "
        "increasing breakout scores before actual breakouts)."
    )

    df = _get_latest_results_df()
    if df is None:
        st.info("Run a scan first to see model-based early breakout candidates.")
        return

    if "PreBreakoutProb" not in df.columns:
        st.info(
            "No pre-breakout predictions are available yet. "
            "Train the model and rerun a scan to populate 'PreBreakoutProb'."
        )
        return

    # Controls row: threshold + limit
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        threshold = st.slider(
            "Minimum pre-breakout probability",
            0.0,
            1.0,
            0.60,
            0.01,
            help=(
                "Only show tickers whose model-based pre-breakout probability "
                "is at or above this value."
            ),
        )
    with c2:
        max_rows = st.number_input(
            "Max rows to display",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )
    with c3:
        sort_desc = st.toggle(
            "Sort by highest probability",
            value=True,
            help="If enabled, highest pre-breakout probabilities appear first.",
        )

    work_df = df.copy()

    # Format probability as percentage for display
    work_df["PreBreakoutProb"] = work_df["PreBreakoutProb"].astype(float)
    work_df["PreBreakoutProb%"] = (work_df["PreBreakoutProb"] * 100.0).round(1)

    filtered = work_df[work_df["PreBreakoutProb"] >= threshold]

    if sort_desc:
        filtered = filtered.sort_values("PreBreakoutProb", ascending=False)
    else:
        filtered = filtered.sort_values("PreBreakoutProb", ascending=True)

    filtered = filtered.head(int(max_rows)).reset_index(drop=True)

    st.caption(
        f"{len(filtered)} ticker(s) at or above {threshold:.2f} probability. "
        "Higher values indicate a stronger historical pre-breakout pattern."
    )

    if filtered.empty:
        st.warning(
            "No symbols meet the current probability threshold. "
            "Try lowering the threshold or running a different scan."
        )
        return

    # Choose a concise set of columns for display if the DF is wide
    preferred_cols = [
        col
        for col in [
            "Symbol",
            "Ticker",
            "Name",
            "Last",
            "Change",
            "% Change",
            "BreakoutScore",
            "Trend10D%",
            "Trend20D%",
            "VolRel20",
            "DollarVol20",
            "PreBreakoutProb%",
        ]
        if col in filtered.columns
    ]
    if preferred_cols:
        display_df = filtered[preferred_cols]
    else:
        display_df = filtered

    st.dataframe(display_df, use_container_width=True)
