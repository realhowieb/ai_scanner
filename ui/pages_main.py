from __future__ import annotations

# ui/pages_main.py
import datetime as _dt
from typing import Optional

import pandas as pd
import streamlit as st

from ui.market_heat import fetch_hot_stocks, fetch_most_active_stocks, fetch_trending_stocks
from ui.page_runners import (
    run_nasdaq_button,
    run_postmarket,
    run_premarket,
    run_sp500,
)

# --- Optional imports (graceful fallbacks) ---
try:
    from db.runs import list_runs, load_run_results  # type: ignore
except ImportError:  # pragma: no cover
    list_runs = None  # type: ignore
    load_run_results = None  # type: ignore

# -------------------- UI helpers --------------------

def _pill(label: str, value: str, help_text: Optional[str] = None):
    """Small pill UI element rendered via markdown."""
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:#EEF2FF;color:#1F2937;font-size:12px;margin-right:6px'>"
        f"<b>{label}</b>: {value}</span>",
        unsafe_allow_html=True,
    )

# --- User settings (optional, Neon only) ---
try:
    from db.user_settings import get_user_settings, upsert_user_settings  # type: ignore
except ImportError:  # pragma: no cover
    get_user_settings = None  # type: ignore
    upsert_user_settings = None  # type: ignore

def _render_sidebar_settings():
    # Who is logged in?
    username = st.session_state.get("username") or st.session_state.get("user")

    # One-time load of saved settings for this user (if Neon + helper available)
    if (
        username
        and callable(get_user_settings)  # type: ignore[truthy-function]
        and not st.session_state.get("_loaded_user_settings", False)
    ):
        try:
            saved = get_user_settings(username)  # type: ignore[misc]
        except (RuntimeError, TypeError, ValueError, OSError):
            saved = None
        if isinstance(saved, dict):
            # Only hydrate keys that are not already in session_state
            for key in (
                "universe",
                "min_price",
                "max_price",
                "min_dollar_vol",
                "include_ta",
                "apply_gap_filter",
                "show_diagnostics_ui",
            ):
                if key in saved and saved[key] is not None and key not in st.session_state:
                    st.session_state[key] = saved[key]
        st.session_state["_loaded_user_settings"] = True

    with st.sidebar:
        st.header("Scan Settings")

        # Universe
        universe_options = ["S&P 500", "Nasdaq 100", "S&P 600", "All (US)", "Custom"]
        default_universe = st.session_state.get("universe", "S&P 500")
        try:
            universe_index = universe_options.index(default_universe)
        except ValueError:
            universe_index = 0

        universe = st.selectbox(
            "Universe",
            universe_options,
            index=universe_index,
            help="Which symbol universe to scan.",
        )
        st.session_state["universe"] = universe

        # Price filters
        c1, c2 = st.columns(2)
        with c1:
            min_price = st.number_input(
                "Min Price",
                min_value=0.0,
                value=float(st.session_state.get("min_price", 1.0)),
                step=0.5,
            )
        with c2:
            max_price = st.number_input(
                "Max Price",
                min_value=0.0,
                value=float(st.session_state.get("max_price", 1000.0)),
                step=1.0,
            )
        st.session_state["min_price"] = float(min_price)
        st.session_state["max_price"] = float(max_price)

        # Liquidity filter (Dollar Volume)
        min_dollar_vol = st.number_input(
            "Min $ Volume (1d)",
            min_value=0.0,
            value=float(st.session_state.get("min_dollar_vol", 1_000_000.0)),
            step=100_000.0,
            help="Filter out thinly traded symbols using price*volume.",
        )
        st.session_state["min_dollar_vol"] = float(min_dollar_vol)

        # Technical calculations toggle
        include_ta = st.checkbox(
            "Include technical indicators (EMA/RSI/ATR)",
            value=bool(st.session_state.get("include_ta", True)),
        )
        st.session_state["include_ta"] = bool(include_ta)

        # Gap/Unusual volume scan toggle
        apply_gap_filter = st.checkbox(
            "Also run Gap + Unusual Volume scan",
            value=bool(st.session_state.get("apply_gap_filter", False)),
        )
        st.session_state["apply_gap_filter"] = bool(apply_gap_filter)

        # Diagnostics
        show_diagnostics_ui = st.checkbox(
            "Show Diagnostics",
            value=bool(st.session_state.get("show_diagnostics_ui", False)),
        )
        st.session_state["show_diagnostics_ui"] = bool(show_diagnostics_ui)

        st.caption(
            "Sidebar settings are passed automatically into run functions "
            "(only parameters they accept are used)."
        )

        # --- Save as default for this user ---
        # Diagnostics: show who we think is logged in and whether storage is wired
        st.caption(
            f"User settings status — user: {username or 'not set'}, "
            f"storage: {'available' if callable(upsert_user_settings) else 'unavailable'}"
        )

        if username and callable(upsert_user_settings):  # type: ignore[truthy-function]
            st.caption(f"Signed in as: {username}")
            if st.button("💾 Save as my default settings"):
                try:
                    upsert_user_settings(  # type: ignore[misc]
                        user_id=username,
                        universe=st.session_state.get("universe"),
                        min_price=st.session_state.get("min_price"),
                        max_price=st.session_state.get("max_price"),
                        min_dollar_vol=st.session_state.get("min_dollar_vol"),
                        include_ta=st.session_state.get("include_ta"),
                        apply_gap_filter=st.session_state.get("apply_gap_filter"),
                        show_diagnostics_ui=st.session_state.get("show_diagnostics_ui"),
                    )
                    st.success("Default scan settings saved for your account.")
                except (RuntimeError, TypeError, ValueError, OSError) as e:
                    st.error(f"Failed to save default settings: {e}")
        elif username:
            st.caption(f"Signed in as: {username}")
            st.caption("User settings storage is not available (Neon-only feature).")
        else:
            st.caption(
                "No username set in session_state — defaults cannot be saved between sessions. "
                "Set st.session_state['username'] in your auth/login flow if you want per-user defaults."
            )

def _render_runs_table(max_rows: int = 200):
    if list_runs is None:
        st.info("Database not available yet — list_runs() missing.")
        return
    try:
        runs_df = list_runs(limit=max_rows)  # expected to return a pandas DataFrame
    except (RuntimeError, TypeError, ValueError, OSError) as e:  # pragma: no cover
        st.error(f"Failed to load history: {e}")
        return

    if runs_df is None or runs_df.empty:
        st.write("No runs saved yet.")
        return

    # Light formatting
    view = runs_df.copy()
    for c in ["started_at", "finished_at"]:
        if c in view.columns:
            try:
                view[c] = pd.to_datetime(view[c])
            except (TypeError, ValueError):
                pass
    if "elapsed_s" in view.columns:
        view["elapsed_s"] = pd.to_numeric(view["elapsed_s"], errors="coerce").round(2)

    st.dataframe(view, width="stretch")

    # Details panel
    run_id: Optional[int] = None
    if "id" in runs_df.columns:
        ids = [i for i in runs_df["id"].tolist() if pd.notna(i)]
        if ids:
            run_id = st.selectbox("Inspect run id:", ids, index=0)
    if run_id is not None and load_run_results is not None:
        try:
            details = load_run_results(run_id)
            if isinstance(details, pd.DataFrame) and not details.empty:
                st.markdown("### Results for selected run")
                st.dataframe(details, width="stretch")
            else:
                st.write("This run has no saved rows.")
        except (RuntimeError, TypeError, ValueError, OSError) as e:
            st.error(f"Failed to load run #{run_id} details: {e}")


def _run_button(label: str, fn):
    disabled = fn is None
    target = getattr(fn, "_target_fn", fn)
    target_name = getattr(fn, "_target_name", None)
    if target_name is None and callable(target):
        target_name = f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')}".strip(".")

    if st.button(label, type="primary", disabled=disabled):
        with st.status(f"Running: {label}", expanded=True):
            try:
                res = fn()  # type: ignore[misc]
                st.success("Completed")
                # Extract universe metadata if present
                meta = None
                if isinstance(res, tuple) and len(res) >= 2 and isinstance(res[1], dict):
                    meta = res[1]
                elif isinstance(res, dict) and "meta" in res and isinstance(res["meta"], dict):
                    meta = res["meta"]

                if meta is not None:
                    if "universe_count" in meta:
                        st.caption(f"Universe: {meta['universe_count']:,} tickers")
                    if "universe_head" in meta and meta["universe_head"]:
                        preview_list = ", ".join(meta["universe_head"][:15])
                        st.caption(f"Preview ({len(meta['universe_head'])} shown): {preview_list}")

                # Try to extract a DataFrame from common patterns
                df_to_show = None
                if isinstance(res, pd.DataFrame):
                    df_to_show = res
                elif isinstance(res, dict):
                    # Look for typical keys
                    for k in ("df", "data", "results", "table"):
                        v = res.get(k)
                        if isinstance(v, pd.DataFrame):
                            df_to_show = v
                            break
                    # If looks like a run id
                    if df_to_show is None and load_run_results is not None:
                        for k in ("run_id", "id"):
                            if k in res and res[k] is not None:
                                try:
                                    cand = load_run_results(res[k])
                                    if isinstance(cand, pd.DataFrame) and not cand.empty:
                                        df_to_show = cand
                                except (RuntimeError, TypeError, ValueError, OSError):
                                    pass
                                break
                elif isinstance(res, (list, tuple)):
                    # Find first DataFrame in sequence
                    for item in res:
                        if isinstance(item, pd.DataFrame):
                            df_to_show = item
                            break
                elif isinstance(res, (int, str)) and load_run_results is not None:
                    # Treat as potential run id
                    try:
                        cand = load_run_results(res)
                        if isinstance(cand, pd.DataFrame) and not cand.empty:
                            df_to_show = cand
                    except (RuntimeError, TypeError, ValueError, OSError):
                        pass

                if isinstance(df_to_show, pd.DataFrame) and not df_to_show.empty:
                    st.dataframe(df_to_show.head(50), width="stretch")
                else:
                    st.write("No tabular results returned.")
            except (RuntimeError, TypeError, ValueError, OSError) as e:  # pragma: no cover
                st.error(str(e))

    if disabled:
        st.caption("Function not wired yet.")
    elif target_name:
        # Always compute the inner bound target's real module+name to show it explicitly
        inner = getattr(fn, "_target_fn", None)
        inner_name = None
        if callable(inner):
            inner_mod = getattr(inner, "__module__", "") or ""
            inner_fn = getattr(inner, "__name__", "") or ""
            inner_name = f"{inner_mod}.{inner_fn}".strip(".")
        if inner_name and inner_name not in str(target_name):
            st.caption(f"Using: {target_name} ➜ {inner_name}")
        else:
            st.caption(f"Using: {target_name}")
        # Extra diagnostic if we're still using the local fallback
        if (
            callable(inner)
            and getattr(inner, "__module__", "") == "ui.page_runners"
            and getattr(inner, "__name__", "") == "_fallback_sp500"
        ):
            st.warning("Using local fallback for S&P 500 (proxy list). Real breakout runner not found/importable.")


def _render_market_heat():
    st.subheader("Yahoo Finance — Market Heat")
    st.caption("Quick lists from Yahoo Finance. Data sources may rate-limit occasionally; tables may be empty if unavailable.")
    c1, c2, c3 = st.columns(3)

    # Hot Stocks
    with c1:
        st.markdown("**Hot Stocks**")
        if fetch_hot_stocks is None:
            st.info("fetch_hot_stocks() not available.")
        else:
            try:
                df_hot = fetch_hot_stocks()
                if isinstance(df_hot, pd.DataFrame) and not df_hot.empty:
                    st.dataframe(df_hot, width="stretch")
                else:
                    st.write("No data.")
            except (RuntimeError, TypeError, ValueError, OSError) as e:
                st.error(f"Failed to load: {e}")

    # Most Active
    with c2:
        st.markdown("**Most Active**")
        if fetch_most_active_stocks is None:
            st.info("fetch_most_active_stocks() not available.")
        else:
            try:
                df_act = fetch_most_active_stocks()
                if isinstance(df_act, pd.DataFrame) and not df_act.empty:
                    st.dataframe(df_act, width="stretch")
                else:
                    st.write("No data.")
            except (RuntimeError, TypeError, ValueError, OSError) as e:
                st.error(f"Failed to load: {e}")

    # Trending
    with c3:
        st.markdown("**Trending**")
        if fetch_trending_stocks is None:
            st.info("fetch_trending_stocks() not available.")
        else:
            try:
                df_tr = fetch_trending_stocks()
                if isinstance(df_tr, pd.DataFrame) and not df_tr.empty:
                    st.dataframe(df_tr, width="stretch")
                else:
                    st.write("No data.")
            except (RuntimeError, TypeError, ValueError, OSError) as e:
                st.error(f"Failed to load: {e}")


# -------------------- Main Page --------------------

def render():
    # Tighten top padding so main content sits closer to the header
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI Scanner Dashboard")
    _pill("Env", st.session_state.get("profile", "dev"))
    _pill("Now", _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    _render_sidebar_settings()

    tabs = st.tabs(["Scans", "Market Heat", "History", "Scheduler"])

    # --- Scans ---
    with tabs[0]:
        st.subheader("Manual runs")
        c1, c2 = st.columns(2)
        with c1:
            _run_button("Run S&P 500 Breakout", run_sp500)
            _run_button("Run Nasdaq Breakout", run_nasdaq_button)
        with c2:
            _run_button("Run Pre-market Scan", run_premarket)
            _run_button("Run Post-market Scan", run_postmarket)

        st.subheader("Market Heat (on-demand)")
        c3, c4, c5 = st.columns(3)
        with c3:
            _run_button("Fetch Hot Stocks", fetch_hot_stocks if callable(fetch_hot_stocks) else None)
        with c4:
            _run_button("Fetch Most Active", fetch_most_active_stocks if callable(fetch_most_active_stocks) else None)
        with c5:
            _run_button("Fetch Trending (US)", (lambda: fetch_trending_stocks()) if callable(fetch_trending_stocks) else None)

        st.divider()
        st.subheader("Latest saved runs (compact)")
        _render_runs_table(max_rows=50)

    # --- Market Heat ---
    with tabs[1]:
        _render_market_heat()

    # --- History ---
    with tabs[2]:
        st.subheader("Run history")
        _render_runs_table(max_rows=200)

    # --- Scheduler ---
    with tabs[3]:
        try:
            from scheduler.ui import render_scheduler  # type: ignore
            render_scheduler()
        except ImportError:
            st.info("Scheduler UI module not found. Add `scheduler/ui.py` with a `render_scheduler()` function.")
