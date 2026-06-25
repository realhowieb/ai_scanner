from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from ui.admin_results_tab import render_admin_tab

RESULTS_TAB_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    OSError,
)


def render_results_tabs(
    *,
    df: pd.DataFrame | None,
    flags: dict[str, bool],
    scan_ran_at: Any,
    username: str,
    db_status: str,
    admin_users: object,
    list_runs: Callable[..., Any] | None,
    load_run_results: Callable[..., Any] | None,
    render_results: Callable[..., Any],
    render_prebreakout_tab: Callable[..., Any],
    render_admin_users_panel: Callable[..., Any],
    render_chart_for_ticker: Callable[..., Any],
    generate_ai_note: Callable[..., Any],
    get_db_conn: Callable[[], Any],
    normalize_results_to_df: Callable[[object], pd.DataFrame | None],
) -> None:
    rows = 0 if df is None else len(df)

    if flags.get("can_scan_history") or flags.get("can_early_breakout") or flags.get("can_admin_panel"):
        tab_names = [f"📊 Latest scan results ({rows} rows)"]

        if flags.get("can_early_breakout"):
            tab_names.append("🔮 Early Breakout Candidates")
        if flags.get("can_scan_history"):
            tab_names.append("📚 Scan History")
        if flags.get("can_admin_panel"):
            tab_names.append("🛠 Admin")

        tabs = st.tabs(tab_names)

        tab_latest = tabs[0]
        idx = 1

        tab_early = None
        if flags.get("can_early_breakout"):
            tab_early = tabs[idx]
            idx += 1

        tab_history = None
        if flags.get("can_scan_history"):
            tab_history = tabs[idx]
            idx += 1

        tab_admin = None
        if flags.get("can_admin_panel"):
            tab_admin = tabs[idx]
    else:
        (tab_latest,) = st.tabs([f"📊 Latest scan results ({rows} rows)"])
        tab_early = None
        tab_history = None
        tab_admin = None

    _render_latest_results_tab(
        tab_latest=tab_latest,
        df=df,
        rows=rows,
        flags=flags,
        scan_ran_at=scan_ran_at,
        render_results=render_results,
        render_chart_for_ticker=render_chart_for_ticker,
        generate_ai_note=generate_ai_note,
    )

    if tab_early is not None:
        _render_early_breakout_tab(
            tab_early=tab_early,
            username=username,
            render_prebreakout_tab=render_prebreakout_tab,
        )

    if tab_history is not None:
        _render_scan_history_tab(
            tab_history=tab_history,
            username=username,
            flags=flags,
            list_runs=list_runs,
            load_run_results=load_run_results,
            render_results=render_results,
            render_chart_for_ticker=render_chart_for_ticker,
            generate_ai_note=generate_ai_note,
            normalize_results_to_df=normalize_results_to_df,
        )

    if tab_admin is not None:
        render_admin_tab(
            tab_admin=tab_admin,
            username=username,
            db_status=db_status,
            admin_users=admin_users,
            render_admin_users_panel=render_admin_users_panel,
            get_db_conn=get_db_conn,
        )


def _render_latest_results_tab(
    *,
    tab_latest: Any,
    df: pd.DataFrame | None,
    rows: int,
    flags: dict[str, bool],
    scan_ran_at: Any,
    render_results: Callable[..., Any],
    render_chart_for_ticker: Callable[..., Any],
    generate_ai_note: Callable[..., Any],
) -> None:
    with tab_latest:
        if scan_ran_at:
            try:
                st.caption(f"🕒 Scan run at {scan_ran_at.strftime('%Y-%m-%d %H:%M UTC')}")
            except RESULTS_TAB_ERRORS:
                st.caption("🕒 Scan run time available")

        if rows == 0:
            with st.expander(f"📊 Latest scan results ({rows} rows)", expanded=False):
                render_results(
                    df,
                    flags["can_export_csv"],
                    flags["can_ai_notes"],
                    render_chart_for_ticker,
                    generate_ai_note,
                )
        else:
            render_results(
                df,
                flags["can_export_csv"],
                flags["can_ai_notes"],
                render_chart_for_ticker,
                generate_ai_note,
            )

        # Premium: AI-generated summary of the strongest setups.
        if rows > 0 and flags.get("can_ai_notes"):
            try:
                st.divider()
                from ui.ai_summary import render_ai_summary
                render_ai_summary(df)
            except RESULTS_TAB_ERRORS:
                pass


def _render_early_breakout_tab(
    *,
    tab_early: Any,
    username: str,
    render_prebreakout_tab: Callable[..., Any],
) -> None:
    with tab_early:
        try:
            render_prebreakout_tab()
        except TypeError:
            try:
                render_prebreakout_tab(username=username)
            except RESULTS_TAB_ERRORS:
                st.info("Early Breakout Candidates panel is not available in this build.")
        except RESULTS_TAB_ERRORS as e:
            st.error("Early Breakout Candidates failed to render.")
            try:
                st.exception(e)
            except RESULTS_TAB_ERRORS:
                st.caption(f"{type(e).__name__}: {e}")


def _render_scan_history_tab(
    *,
    tab_history: Any,
    username: str,
    flags: dict[str, bool],
    list_runs: Callable[..., Any] | None,
    load_run_results: Callable[..., Any] | None,
    render_results: Callable[..., Any],
    render_chart_for_ticker: Callable[..., Any],
    generate_ai_note: Callable[..., Any],
    normalize_results_to_df: Callable[[object], pd.DataFrame | None],
) -> None:
    with tab_history:
        st.markdown("## 📚 Scan History")

        if not callable(list_runs) or not callable(load_run_results):
            st.info("Scan history is not available (DB runs module not configured).")
            return

        runs = None
        try:
            runs = list_runs(username=username)
        except TypeError:
            try:
                runs = list_runs(user_id=username)
            except TypeError:
                runs = list_runs()
        except RESULTS_TAB_ERRORS as e:
            st.error("Failed to load scan history.")
            try:
                st.exception(e)
            except RESULTS_TAB_ERRORS:
                st.caption(f"{type(e).__name__}: {e}")

        if not runs:
            st.info("No saved scans yet. Run a scan and make sure it saves to history.")
            return

        try:
            runs_df = pd.DataFrame(runs)
        except RESULTS_TAB_ERRORS:
            runs_df = pd.DataFrame([{"run": r} for r in runs])

        id_col = None
        for col in ("run_id", "id", "runId", "uuid"):
            if col in runs_df.columns:
                id_col = col
                break

        st.dataframe(runs_df, width="stretch")

        if id_col is None:
            st.caption("Scan history loaded, but no run id column was found to load details.")
            return

        try:
            options = runs_df[id_col].astype(str).tolist()
        except RESULTS_TAB_ERRORS:
            options = [str(x) for x in runs_df[id_col].tolist()]

        picked = st.selectbox(
            "Select a run to view",
            options,
            index=0,
            key="scan_history_pick_run",
        )

        if not picked:
            return

        run_df = None
        try:
            run_df = load_run_results(picked)
        except TypeError:
            try:
                run_df = load_run_results(run_id=picked)
            except TypeError:
                run_df = load_run_results(str(picked))
        except RESULTS_TAB_ERRORS as e:
            st.error("Failed to load results for the selected run.")
            try:
                st.exception(e)
            except RESULTS_TAB_ERRORS:
                st.caption(f"{type(e).__name__}: {e}")

        run_df_norm = normalize_results_to_df(run_df)

        if run_df_norm is None:
            st.info("No results found for this run (or could not parse results).")
            if bool(st.session_state.get("show_diagnostics_ui", False)):
                st.caption(f"run_df_raw type: {type(run_df).__name__}")
            return

        st.markdown("### Results for selected run")
        try:
            render_results(
                run_df_norm,
                flags.get("can_export_csv", False),
                flags.get("can_ai_notes", False),
                render_chart_for_ticker,
                generate_ai_note,
                key_prefix=f"history_results_{_safe_widget_key(picked)}",
            )
        except RESULTS_TAB_ERRORS:
            st.dataframe(run_df_norm, width="stretch")


def _safe_widget_key(value: object) -> str:
    text = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    return safe[:80] or "selected"
