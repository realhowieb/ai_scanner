from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st


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
    fetch_earnings_this_week: Callable[..., Any],
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
        _render_admin_tab(
            tab_admin=tab_admin,
            username=username,
            db_status=db_status,
            admin_users=admin_users,
            render_admin_users_panel=render_admin_users_panel,
            fetch_earnings_this_week=fetch_earnings_this_week,
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
            except Exception:
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
            except Exception:
                st.info("Early Breakout Candidates panel is not available in this build.")
        except Exception as e:
            st.error("Early Breakout Candidates failed to render.")
            try:
                st.exception(e)
            except Exception:
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
        except Exception as e:
            st.error("Failed to load scan history.")
            try:
                st.exception(e)
            except Exception:
                st.caption(f"{type(e).__name__}: {e}")

        if not runs:
            st.info("No saved scans yet. Run a scan and make sure it saves to history.")
            return

        try:
            runs_df = pd.DataFrame(runs)
        except Exception:
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
        except Exception:
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
        except Exception as e:
            st.error("Failed to load results for the selected run.")
            try:
                st.exception(e)
            except Exception:
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
            )
        except Exception:
            st.dataframe(run_df_norm, width="stretch")


def _render_admin_tab(
    *,
    tab_admin: Any,
    username: str,
    db_status: str,
    admin_users: object,
    render_admin_users_panel: Callable[..., Any],
    fetch_earnings_this_week: Callable[..., Any],
    get_db_conn: Callable[[], Any],
) -> None:
    with tab_admin:
        st.markdown("## 🛠 Admin Panel")

        try:
            render_admin_users_panel(
                username=username,
                ADMIN_USERS=admin_users,
                db_status=db_status,
            )
        except Exception as e:
            st.error("Admin panel failed to render.")
            try:
                st.exception(e)
            except Exception:
                st.caption(f"{type(e).__name__}: {e}")

        st.markdown("---")
        st.markdown("### 🧰 Admin Tools")

        if not bool(st.session_state.get("is_admin")):
            st.info("🔒 Admin tools are only available to admin users.")
        else:
            c1, c2, c3 = st.columns(3)
            _render_force_tier_resync(c1)
            _render_earnings_refresh(c2, fetch_earnings_this_week, get_db_conn)
            _render_db_integrity_checks(c3, get_db_conn)

        st.caption(
            "Tip: Earnings refresh is intentionally admin-only and never runs automatically during scans."
        )


def _render_force_tier_resync(column: Any) -> None:
    with column:
        if st.button("🔄 Force tier resync", key="admin_force_tier_resync", width="stretch"):
            for key in (
                "entitlements",
                "tier",
                "tier_key",
                "profile_loaded_for_user",
                "pricing_focus",
            ):
                st.session_state.pop(key, None)
            st.success("Tier resync requested (session caches cleared). Reloading...")
            st.rerun()


def _render_earnings_refresh(
    column: Any,
    fetch_earnings_this_week: Callable[..., Any],
    get_db_conn: Callable[[], Any],
) -> None:
    with column:
        if not st.button("📅 Refresh earnings now", key="admin_refresh_earnings", width="stretch"):
            return

        with st.spinner("Refreshing earnings calendar (admin)..."):
            conn = None
            refreshed = 0
            try:
                earnings_list = []
                try:
                    earnings_list = fetch_earnings_this_week() or []
                except Exception:
                    earnings_list = []

                if not earnings_list:
                    st.info(
                        "Earnings refresh completed, but the upstream source returned 0 items. "
                        "(This can happen on weekends/holidays or if the provider is unavailable.)"
                    )
                    _clear_earnings_result_cache()
                    st.stop()

                conn = get_db_conn()
                try:
                    from db.earnings import ensure_earnings_table  # type: ignore

                    ensure_earnings_table(conn)
                except Exception:
                    pass

                try:
                    import db.earnings as earn_mod  # type: ignore

                    populate = getattr(earn_mod, "populate_earnings_calendar", None)
                    if not callable(populate):
                        raise TypeError("populate_earnings_calendar is not callable")

                    try:
                        populate(conn, earnings_list)
                    except TypeError:
                        try:
                            populate(earnings_list)
                        except TypeError:
                            try:
                                populate(conn=conn, earnings_info_list=earnings_list)
                            except TypeError:
                                populate(conn=conn, earnings_list=earnings_list)

                    refreshed = len(earnings_list)
                except Exception as e:
                    st.warning(f"Earnings refresh ran, but upsert failed: {type(e).__name__}: {e}")

                if refreshed > 0:
                    st.success(f"✅ Earnings refreshed: upsert attempted for {refreshed} symbols.")
                else:
                    st.info(
                        "Earnings refresh completed, but no rows were upserted. "
                        "(This can happen if the DB helper is unavailable or rejected all symbols.)"
                    )

                _clear_earnings_result_cache()

            except Exception as e:
                st.error("❌ Earnings refresh failed.")
                try:
                    st.exception(e)
                except Exception:
                    st.caption(f"{type(e).__name__}: {e}")
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass


def _render_db_integrity_checks(column: Any, get_db_conn: Callable[[], Any]) -> None:
    with column:
        if not st.button("🧪 DB integrity checks", key="admin_db_integrity", width="stretch"):
            return

        with st.spinner("Running DB checks..."):
            try:
                conn = get_db_conn()
                issues: list[str] = []

                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1;")
                        _ = cur.fetchone()
                except Exception as e:
                    issues.append(f"DB connectivity failed: {type(e).__name__}: {e}")

                earnings_count = None
                bad_symbol_count = None
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT COUNT(*)
                            FROM information_schema.tables
                            WHERE table_name = 'earnings_calendar';
                            """
                        )
                        exists = (cur.fetchone() or [0])[0]

                        if not exists:
                            issues.append("earnings_calendar table is missing.")
                        else:
                            cur.execute("SELECT COUNT(*) FROM earnings_calendar;")
                            earnings_count = (cur.fetchone() or [0])[0]

                            cur.execute(
                                """
                                SELECT COUNT(*)
                                FROM earnings_calendar
                                WHERE symbol IS NULL
                                   OR symbol <> UPPER(TRIM(symbol))
                                   OR symbol = '';
                                """
                            )
                            bad_symbol_count = (cur.fetchone() or [0])[0]

                    if bad_symbol_count and int(bad_symbol_count) > 0:
                        try:
                            from db.earnings import ensure_earnings_table  # type: ignore

                            ensure_earnings_table(conn)
                            issues.append(
                                f"Found {bad_symbol_count} non-normalized symbols; ran normalization UPDATE."
                            )
                        except Exception:
                            issues.append(
                                f"Found {bad_symbol_count} non-normalized symbols; normalization function unavailable."
                            )
                except Exception as e:
                    issues.append(f"Earnings table checks failed: {type(e).__name__}: {e}")

                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass

                st.markdown("#### ✅ DB Check Report")
                if earnings_count is not None:
                    st.write(f"Earnings rows: **{int(earnings_count)}**")
                if bad_symbol_count is not None:
                    st.write(f"Symbol hygiene issues: **{int(bad_symbol_count)}**")

                if issues:
                    st.warning("Issues / notes:")
                    for msg in issues:
                        st.write(f"- {msg}")
                else:
                    st.success("All checks passed.")

            except Exception as e:
                st.error("❌ DB checks failed.")
                try:
                    st.exception(e)
                except Exception:
                    st.caption(f"{type(e).__name__}: {e}")


def _clear_earnings_result_cache() -> None:
    st.session_state.pop("earnings_enriched_df", None)
    st.session_state.pop("earnings_enriched_signature", None)
    st.session_state.pop("earnings_enrichment_rerun_sig", None)
