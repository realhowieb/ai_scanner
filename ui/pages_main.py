from __future__ import annotations
# ui/pages_main.py
import datetime as _dt
from typing import Optional
import inspect as _inspect
import pandas as pd
import streamlit as st
from ui.market_heat import fetch_hot_stocks, fetch_most_active_stocks, fetch_trending_stocks

# --- Optional imports (graceful fallbacks) ---
try:
    from db.runs import list_runs, load_run_results  # type: ignore
except Exception:  # pragma: no cover
    list_runs = None  # type: ignore
    load_run_results = None  # type: ignore

def _optional_attr(module_name: str, attr_name: str):
    """Return an optional attribute from a module without failing page import."""
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception:
        return None


# Try to import on-demand run helpers (may not exist in all installs)
_run_sp500 = _run_nasdaq = _run_premarket = _run_postmarket = None

# --- Expanded discovery for S&P 500 runner ---
for _module_name, _attr_name in (
    ("scheduler.jobs", "run_sp500_now"),
    ("ai_scanner.scheduler.jobs", "run_sp500_now"),
):
    if _run_sp500 is None:
        _run_sp500 = _optional_attr(_module_name, _attr_name)

# --- Expanded discovery for Nasdaq runner ---
for _module_name, _attr_name in (
    ("scheduler.jobs", "run_nasdaq_now"),
    ("ai_scanner.scheduler.jobs", "run_nasdaq_now"),
):
    if _run_nasdaq is None:
        _run_nasdaq = _optional_attr(_module_name, _attr_name)

# --- Expanded discovery for Pre-market runner ---
for _module_name, _attr_name in (
    ("scheduler.jobs", "run_premarket_now"),
    ("scan.pre_post", "run_premarket_headless"),
    ("ai_scanner.scheduler.jobs", "run_premarket_now"),
    ("ai_scanner.scan.pre_post", "run_premarket_headless"),
):
    if _run_premarket is None:
        _run_premarket = _optional_attr(_module_name, _attr_name)

# --- Expanded discovery for Post-market runner ---
for _module_name, _attr_name in (
    ("scheduler.jobs", "run_postmarket_now"),
    ("scan.pre_post", "run_postmarket_headless"),
    ("ai_scanner.scheduler.jobs", "run_postmarket_now"),
    ("ai_scanner.scan.pre_post", "run_postmarket_headless"),
):
    if _run_postmarket is None:
        _run_postmarket = _optional_attr(_module_name, _attr_name)

# Fallbacks: keep manual run buttons usable even when no scanner runner is importable.

# Final local fallbacks so manual run buttons always work
if _run_sp500 is None:
    def _run_sp500(
        universe: str = "S&P 500",
        min_price: float | None = None,
        max_price: float | None = None,
        min_dollar_vol: float | None = None,
        **kwargs
    ):
        # Use Hot Stocks as a proxy breakout list
        df = fetch_hot_stocks(count=100) if callable(fetch_hot_stocks) else pd.DataFrame()
        if isinstance(df, pd.DataFrame) and not df.empty:
            if min_price is not None and "price" in df:
                df = df[df["price"] >= float(min_price)]
            if max_price is not None and max_price > 0 and "price" in df:
                df = df[df["price"] <= float(max_price)]
            if min_dollar_vol is not None and "price" in df and "volume" in df:
                df = df[(df["price"] * df["volume"]) >= float(min_dollar_vol)]
        return df.reset_index(drop=True)

if _run_nasdaq is None:
    def _run_nasdaq(
        universe: str = "Nasdaq 100",
        min_price: float | None = None,
        max_price: float | None = None,
        min_dollar_vol: float | None = None,
        **kwargs
    ):
        # Use Most Active as a proxy
        df = fetch_most_active_stocks(count=100) if callable(fetch_most_active_stocks) else pd.DataFrame()
        if isinstance(df, pd.DataFrame) and not df.empty:
            if min_price is not None and "price" in df:
                df = df[df["price"] >= float(min_price)]
            if max_price is not None and max_price > 0 and "price" in df:
                df = df[df["price"] <= float(max_price)]
            if min_dollar_vol is not None and "price" in df and "volume" in df:
                df = df[(df["price"] * df["volume"]) >= float(min_dollar_vol)]
        return df.reset_index(drop=True)

if _run_premarket is None:
    def _run_premarket(
        min_price: float | None = None,
        max_price: float | None = None,
        min_dollar_vol: float | None = None,
        **kwargs
    ):
        # Proxy using Most Active (premarket endpoint not available via JSON without bs4)
        df = fetch_most_active_stocks(count=50) if callable(fetch_most_active_stocks) else pd.DataFrame()
        if isinstance(df, pd.DataFrame) and not df.empty:
            if min_price is not None and "price" in df:
                df = df[df["price"] >= float(min_price)]
            if max_price is not None and max_price > 0 and "price" in df:
                df = df[df["price"] <= float(max_price)]
            if min_dollar_vol is not None and "price" in df and "volume" in df:
                df = df[(df["price"] * df["volume"]) >= float(min_dollar_vol)]
        return df.reset_index(drop=True)

if _run_postmarket is None:
    def _run_postmarket(
        min_price: float | None = None,
        max_price: float | None = None,
        min_dollar_vol: float | None = None,
        **kwargs
    ):
        # Proxy using Day Gainers
        df = fetch_hot_stocks(count=50) if callable(fetch_hot_stocks) else pd.DataFrame()
        if isinstance(df, pd.DataFrame) and not df.empty:
            if min_price is not None and "price" in df:
                df = df[df["price"] >= float(min_price)]
            if max_price is not None and max_price > 0 and "price" in df:
                df = df[df["price"] <= float(max_price)]
            if min_dollar_vol is not None and "price" in df and "volume" in df:
                df = df[(df["price"] * df["volume"]) >= float(min_dollar_vol)]
        return df.reset_index(drop=True)


# -------------------- Wiring helpers --------------------
def _collect_scan_context() -> dict:
    """Gather common scan parameters from session_state."""
    return {
        "universe": st.session_state.get("universe"),
        "min_price": st.session_state.get("min_price"),
        "max_price": st.session_state.get("max_price"),
        "min_dollar_vol": st.session_state.get("min_dollar_vol"),
        "include_ta": st.session_state.get("include_ta"),
        "apply_gap_filter": st.session_state.get("apply_gap_filter"),
        "show_diagnostics_ui": st.session_state.get("show_diagnostics_ui"),
    }

def _bind_session_args(fn):
    """Return a callable that invokes `fn` with sensible kwargs mapped from
    Streamlit session_state, adapting to differing function signatures.

    This ensures the correct universe/index/session params are passed even if
    runner functions use different names (e.g., `index` vs `universe`,
    `min_dollar_volume` vs `min_dollar_vol`).
    """
    if fn is None:
        return None

    sig = _inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    fn_name = getattr(fn, "__name__", "") or ""
    fn_mod = getattr(fn, "__module__", "") or ""

    def map_universe(val: str) -> str:
        # Normalize universe to common index tokens if needed
        mapping = {
            "S&P 500": "SP500",
            "Nasdaq 100": "NASDAQ100",
            "S&P 600": "SP600",
            "All (US)": "US",
            "Custom": "CUSTOM",
        }
        return mapping.get(val, val)

    def _runner():
        ctx = _collect_scan_context()
        kwargs = {}

        # Basic passthrough for identically named args
        for k in ("universe", "min_price", "max_price", "include_ta", "apply_gap_filter", "show_diagnostics_ui"):
            if k in accepted and k in ctx:
                kwargs[k] = ctx[k]

        # Dollar volume aliasing
        if "min_dollar_vol" in ctx:
            if "min_dollar_vol" in accepted:
                kwargs["min_dollar_vol"] = ctx["min_dollar_vol"]
            elif "min_dollar_volume" in accepted:
                kwargs["min_dollar_volume"] = ctx["min_dollar_vol"]
            elif "dollar_volume_min" in accepted:
                kwargs["dollar_volume_min"] = ctx["min_dollar_vol"]

        # Universe/index mapping
        universe_val = ctx.get("universe")
        if universe_val:
            # Some runners expect `index`
            if "index" in accepted and "universe" not in accepted:
                kwargs["index"] = map_universe(universe_val)
            # Some use a boolean or token-like flag; fallbacks by function name

        # Pre/Post market sessions
        if "session" in accepted and ("premarket" in fn_name or "premarket" in fn_mod):
            kwargs["session"] = "pre"
        if "session" in accepted and ("postmarket" in fn_name or "postmarket" in fn_mod):
            kwargs["session"] = "post"

        # If runner is clearly SP500/Nasdaq but lacks both universe/index, supply an index
        if ("sp500" in fn_name or "sp500" in fn_mod) and ("universe" not in accepted and "index" in accepted):
            kwargs.setdefault("index", "SP500")
        if ("nasdaq" in fn_name or "nasdaq" in fn_mod) and ("universe" not in accepted and "index" in accepted):
            kwargs.setdefault("index", "NASDAQ100")

        # Finally, call
        try:
            return fn(**kwargs) if kwargs else fn()
        except TypeError:
            # If signature mismatch still occurs, try calling without kwargs
            return fn()

    # Attach debug metadata so buttons can display the actual bound function
    setattr(_runner, "_target_fn", fn)
    setattr(_runner, "_target_name", f"{getattr(fn, '__module__', '')}.{getattr(fn, '__name__', '')}")
    return _runner


# Wrap discovered run functions so they consume sidebar settings automatically
_run_sp500 = _bind_session_args(_run_sp500)
_run_nasdaq = _bind_session_args(_run_nasdaq)
_run_premarket = _bind_session_args(_run_premarket)
_run_postmarket = _bind_session_args(_run_postmarket)

# -------- Per-button universe overrides (ignore sidebar universe) --------
import inspect as _inspect2
def _call_with_overrides(bound_fn, universe_token: str, universe_name: str):
    """
    Call the underlying target function for a runner with a forced universe/index,
    while still passing price/liquidity filters from the sidebar.
    """
    if bound_fn is None:
        raise RuntimeError("Runner function is not available")
    target = getattr(bound_fn, "_target_fn", bound_fn)
    sig = _inspect2.signature(target)
    accepted = set(sig.parameters.keys())

    ctx = _collect_scan_context()
    kwargs = {}

    # Map price/liquidity if accepted by the target
    for k in ("min_price", "max_price"):
        if k in accepted and ctx.get(k) is not None:
            kwargs[k] = ctx[k]
    # Dollar volume aliasing
    if "min_dollar_vol" in accepted and ctx.get("min_dollar_vol") is not None:
        kwargs["min_dollar_vol"] = ctx["min_dollar_vol"]
    elif "min_dollar_volume" in accepted and ctx.get("min_dollar_vol") is not None:
        kwargs["min_dollar_volume"] = ctx["min_dollar_vol"]
    elif "dollar_volume_min" in accepted and ctx.get("min_dollar_vol") is not None:
        kwargs["dollar_volume_min"] = ctx["min_dollar_vol"]

    # Force universe/index, ignoring sidebar universe
    if "index" in accepted:
        kwargs["index"] = universe_token
    # Prefer passing the token to `universe` parameters as well (most scanners expect tokens)
    if "universe" in accepted:
        kwargs["universe"] = universe_token
    # Some APIs expose explicit token params
    if "universe_token" in accepted:
        kwargs["universe_token"] = universe_token
    if "universe_name" in accepted:
        kwargs["universe_name"] = universe_name

    # Pass optional toggles when accepted
    for k in ("include_ta", "apply_gap_filter", "show_diagnostics_ui"):
        if k in accepted and k in ctx:
            kwargs[k] = ctx[k]

    return target(**kwargs) if kwargs else target()

def _wrap_override(bound_fn, universe_token: str, universe_name: str):
    """Return a callable that invokes bound_fn's target with forced universe/index."""
    def _runner():
        return _call_with_overrides(bound_fn, universe_token, universe_name)
    # Attach debug metadata for UI caption
    target = getattr(bound_fn, "_target_fn", bound_fn)
    setattr(_runner, "_target_fn", target)
    setattr(_runner, "_target_name",
            f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')} [override:{universe_token}]")
    return _runner

# Concrete per-button runners
_run_sp500_button = _wrap_override(_run_sp500, universe_token="SP500", universe_name="S&P 500")
_run_nasdaq_button = _wrap_override(_run_nasdaq, universe_token="NASDAQ100", universe_name="Nasdaq 100")

# Public shims so external modules can call ui.pages_main.run_* reliably
def run_sp500():
    """Public entry that invokes the bound SP500 runner with sidebar args."""
    if _run_sp500 is None:
        raise RuntimeError("SP500 runner is not available")
    return _run_sp500()

def run_nasdaq():
    """Public entry that invokes the bound Nasdaq runner with sidebar args."""
    if _run_nasdaq is None:
        raise RuntimeError("Nasdaq runner is not available")
    return _run_nasdaq()

def run_premarket():
    """Public entry that invokes the bound pre-market runner with sidebar args."""
    if _run_premarket is None:
        raise RuntimeError("Premarket runner is not available")
    return _run_premarket()

def run_postmarket():
    """Public entry that invokes the bound post-market runner with sidebar args."""
    if _run_postmarket is None:
        raise RuntimeError("Postmarket runner is not available")
    return _run_postmarket()

# Make these public symbols explicit for importers
__all__ = [
    "run_sp500",
    "run_nasdaq",
    "run_premarket",
    "run_postmarket",
]


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
except Exception:  # pragma: no cover
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
        except Exception:
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
                except Exception as e:
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
    except Exception as e:  # pragma: no cover
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
            except Exception:
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
        except Exception as e:
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
                                except Exception:
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
                    except Exception:
                        pass

                if isinstance(df_to_show, pd.DataFrame) and not df_to_show.empty:
                    st.dataframe(df_to_show.head(50), width="stretch")
                else:
                    st.write("No tabular results returned.")
            except Exception as e:  # pragma: no cover
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
        if callable(inner) and getattr(inner, "__module__", "") == __name__ and getattr(inner, "__name__", "") == "_run_sp500":
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
            except Exception as e:
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
            except Exception as e:
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
            except Exception as e:
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
            _run_button("Run S&P 500 Breakout", _run_sp500)
            _run_button("Run Nasdaq Breakout", _run_nasdaq_button)
        with c2:
            _run_button("Run Pre-market Scan", _run_premarket)
            _run_button("Run Post-market Scan", _run_postmarket)

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
        except Exception:
            st.info("Scheduler UI module not found. Add `scheduler/ui.py` with a `render_scheduler()` function.")
