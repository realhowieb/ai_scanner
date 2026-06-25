"""Scan history UI module."""

import traceback
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from auth.tiering import has_min_tier
from db.runs import list_runs, load_run_results


def render_history_expander(db_status: str) -> None:
    """Render the Scan History expander and allow loading past runs."""
    # Pro+ only: Basic users cannot view Scan History. Silent guard: no banner for lower tiers.
    tier = st.session_state.get("tier")
    if not has_min_tier(tier, "pro"):
        return

    with st.expander("📜 Scan History", expanded=False):
        runs_list: List[Dict[str, Any]] = []
        try:
            runs_list = list_runs()
        except Exception as e:
            st.error(f"History unavailable (DB error): {e}")
            try:
                st.code(traceback.format_exc())
            except Exception:
                pass
            runs_list = []

        if runs_list:
            options = []
            for r in runs_list:
                # Expect dict-like rows from list_runs
                rid = r.get("id") if isinstance(r, dict) else None
                name = r.get("name") if isinstance(r, dict) else str(r)
                ts = r.get("timestamp") if isinstance(r, dict) else None

                if hasattr(ts, "strftime"):
                    ts_str = ts.strftime("%Y-%m-%d %H:%M")
                elif ts is not None:
                    ts_str = str(ts)
                else:
                    ts_str = ""

                label_str = f"#{rid} — {name}"
                if ts_str:
                    label_str += f" — {ts_str}"
                options.append((label_str, rid))

            if options:
                labels = [lbl for (lbl, _rid) in options]
                selected_label = st.selectbox("Select a past scan to load:", labels, index=0)
                selected_id = None
                for lbl, _rid in options:
                    if lbl == selected_label:
                        selected_id = _rid
                        break

                col_hist1, col_hist2 = st.columns([1, 1])
                with col_hist1:
                    if st.button("Load Selected Scan") and selected_id is not None:
                        try:
                            payload = load_run_results(int(selected_id))
                            hist_df = pd.read_json(payload)
                            st.session_state.results_df = hist_df
                            st.success(f"Loaded scan #{selected_id} from history with {len(hist_df)} rows.")
                        except Exception as e:
                            st.error(f"Failed to load scan #{selected_id}: {e}")
                with col_hist2:
                    if db_status == "neon":
                        st.caption(
                            "History is stored in Neon (cloud Postgres). Local scanner.sqlite is used as a fallback."
                        )
                    elif db_status == "sqlite":
                        st.caption(
                            "History is stored in a local scanner.sqlite file next to app.py."
                        )
                    else:
                        st.caption(
                            "History storage backend is currently unavailable."
                        )
            else:
                st.caption("No past scans saved yet.")
        else:
            st.caption("No past scans saved yet.")