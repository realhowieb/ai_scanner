# ai_scanner/ui/components.py
from __future__ import annotations
import pandas as pd
import streamlit as st

# Optional diagnostics (no-op if not available)
try:
    from .diagnostics import show_runner_info, run_with_diagnostics  # type: ignore
except Exception:  # pragma: no cover
    def show_runner_info(label, target):
        return None
    def run_with_diagnostics(label, fn):
        return fn()

def pill(label: str, value: str):
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:#EEF2FF;color:#1F2937;font-size:12px;margin-right:6px'>"
        f"<b>{label}</b>: {value}</span>",
        unsafe_allow_html=True,
    )

def runs_table(list_runs, load_run_results, max_rows: int = 200):
    if list_runs is None:
        st.info("Database not available yet — list_runs() missing.")
        return
    try:
        runs_df = list_runs(limit=max_rows)
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return
    if runs_df is None or runs_df.empty:
        st.write("No runs saved yet.")
        return
    view = runs_df.copy()
    for c in ["started_at", "finished_at"]:
        if c in view.columns:
            view[c] = pd.to_datetime(view[c], errors="ignore")
    if "elapsed_s" in view.columns:
        view["elapsed_s"] = pd.to_numeric(view["elapsed_s"], errors="coerce").round(2)
    st.dataframe(view, use_container_width=True)
    if "id" in view.columns and load_run_results:
        ids = [i for i in view["id"].tolist() if pd.notna(i)]
        if ids:
            run_id = st.selectbox("Inspect run id:", ids, index=0)
            if run_id is not None:
                try:
                    details = load_run_results(run_id)
                    if isinstance(details, pd.DataFrame) and not details.empty:
                        st.markdown("### Results for selected run")
                        st.dataframe(details, use_container_width=True)
                    else:
                        st.write("This run has no saved rows.")
                except Exception as e:
                    st.error(f"Failed to load run #{run_id} details: {e}")


# Enhanced run_button with diagnostics and flexible result preview
def run_button(label: str, fn):
    disabled = fn is None
    target = getattr(fn, "_target_fn", fn)

    # Show which function is bound (diagnostic caption)
    try:
        show_runner_info(label, fn)
    except Exception:
        pass

    if st.button(label, type="primary", disabled=disabled):
        with st.status(f"Running: {label}", expanded=True):
            try:
                # Time + run summary via diagnostics (no-op fallback if unavailable)
                res = run_with_diagnostics(label, fn)
                st.success("Completed")

                # Extract a DataFrame to preview when possible
                df_to_show = None
                if isinstance(res, pd.DataFrame):
                    df_to_show = res
                elif isinstance(res, dict):
                    for k in ("df", "data", "results", "table"):
                        v = res.get(k)
                        if isinstance(v, pd.DataFrame):
                            df_to_show = v
                            break
                elif isinstance(res, (list, tuple)):
                    for item in res:
                        if isinstance(item, pd.DataFrame):
                            df_to_show = item
                            break

                if isinstance(df_to_show, pd.DataFrame) and not df_to_show.empty:
                    st.dataframe(df_to_show.head(50), use_container_width=True)
                else:
                    st.write("No tabular results returned.")
            except Exception as e:  # pragma: no cover
                st.error(str(e))

    if disabled:
        st.caption("Function not wired yet.")