from __future__ import annotations

import hashlib
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st


def render_earnings_controls(
    *,
    flags: dict[str, bool],
    render_earnings_this_week_panel: Callable[..., Any],
) -> None:
    """Render a single unified Earnings section (toggle + this-week view).

    Previously the enrichment toggle lived in the sidebar while the "this week"
    panel was a separate main-area expander — so the panel told users to "enable
    it in the sidebar". Now it's one '📅 Earnings' section in the main area.
    """
    if not bool(flags.get("can_earnings")):
        for key, value in {
            "enable_earnings_enrichment": False,
            "earnings_enabled": False,
            "enable_earnings": False,
            "enable_earnings_refresh": False,
        }.items():
            try:
                st.session_state[key] = value
            except Exception:
                pass
        _clear_earnings_refresh_state()
        return

    with st.expander("📅 Earnings", expanded=False):
        # Seed a default without passing value= (which conflicts with the
        # session-state key set during profile restore and warns in Streamlit).
        st.session_state.setdefault("enable_earnings_enrichment", False)
        earn_enabled = st.checkbox(
            "Enable earnings enrichment (adds 📅 Earnings in X days)",
            key="enable_earnings_enrichment",
            help=(
                "Adds earnings timing from the DB to your results. "
                "Turn this OFF for the fastest scans."
            ),
        )

        try:
            st.session_state["earnings_enabled"] = bool(earn_enabled)
            st.session_state["enable_earnings"] = bool(earn_enabled)
            st.session_state["enable_earnings_refresh"] = False
        except Exception:
            pass

        if not bool(earn_enabled):
            clear_earnings_result_cache()
            _clear_earnings_refresh_state()

        # This-week view, in the same section.
        st.markdown("**📆 Reporting this week**")
        is_admin = bool(st.session_state.get("is_admin", False))
        if not earn_enabled and not is_admin:
            st.caption("Enable enrichment above to load this week's earnings.")
        else:
            render_earnings_this_week_panel(can_earnings=True)


def prepare_results_with_earnings(
    df: pd.DataFrame | None,
    *,
    flags: dict[str, bool],
    earn_col_days: str,
    add_earnings_days_column: Callable[[pd.DataFrame], pd.DataFrame],
    quiet_external_calls: Callable[[], Any] | None = None,
) -> tuple[pd.DataFrame | None, Any]:
    """Track result freshness, optionally enrich earnings columns, and apply filters."""
    _update_results_signature(df)
    scan_ran_at = st.session_state.get("scan_ran_at_utc")

    show_earnings = bool(flags.get("can_earnings")) and bool(
        st.session_state.get("enable_earnings_enrichment", False)
    )

    if show_earnings and isinstance(df, pd.DataFrame) and not df.empty:
        _ensure_earnings_columns(df, earn_col_days)

    if not flags.get("can_earnings"):
        st.sidebar.caption("🔒 Earnings timing is a Pro feature.")

    signature = str(st.session_state.get("results_signature") or "")
    cached_signature = str(st.session_state.get("earnings_enriched_signature") or "")
    cached_df = st.session_state.get("earnings_enriched_df")

    if (
        show_earnings
        and signature
        and cached_signature == signature
        and isinstance(cached_df, pd.DataFrame)
    ):
        df = cached_df

    if (
        show_earnings
        and isinstance(df, pd.DataFrame)
        and not df.empty
        and signature
        and cached_signature != signature
    ):
        df = _apply_earnings_enrichment(
            df,
            earn_col_days=earn_col_days,
            add_earnings_days_column=add_earnings_days_column,
            quiet_external_calls=quiet_external_calls,
        )
        st.session_state["earnings_enriched_df"] = df
        st.session_state["earnings_enriched_signature"] = signature

    if (
        show_earnings
        and isinstance(df, pd.DataFrame)
        and not df.empty
        and earn_col_days in df.columns
        and pd.to_numeric(df[earn_col_days], errors="coerce").isna().all()
    ):
        enriched_df = _apply_earnings_enrichment(
            df,
            earn_col_days=earn_col_days,
            add_earnings_days_column=add_earnings_days_column,
            quiet_external_calls=quiet_external_calls,
        )
        try:
            if not pd.to_numeric(enriched_df[earn_col_days], errors="coerce").isna().all():
                st.session_state["earnings_enriched_df"] = enriched_df
                st.session_state["earnings_enriched_signature"] = signature
                df = enriched_df
        except Exception:
            pass

    df = _apply_earnings_filters(df, show_earnings=show_earnings, earn_col_days=earn_col_days)
    return df, scan_ran_at


def _update_results_signature(df: pd.DataFrame | None) -> None:
    try:
        if isinstance(df, pd.DataFrame) and not df.empty:
            signature = _results_signature(df)
            previous = str(st.session_state.get("results_signature") or "")
            if signature and signature != previous:
                st.session_state["results_signature"] = signature
                st.session_state["scan_ran_at_utc"] = datetime.now(timezone.utc)
                clear_earnings_result_cache()
            return

        st.session_state.pop("results_signature", None)
        st.session_state.pop("scan_ran_at_utc", None)
        clear_earnings_result_cache()
    except Exception:
        pass


def _results_signature(df: pd.DataFrame) -> str:
    rows = int(len(df))
    ticker_col = None
    for col in ("Ticker", "ticker", "Symbol", "symbol"):
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        return f"{rows}|no_ticker_col"

    try:
        tickers = (
            df[ticker_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": None})
            .dropna()
            .unique()
            .tolist()
        )
        tickers.sort()
        payload = "|".join(tickers).encode("utf-8", errors="ignore")
        return f"{rows}|{hashlib.md5(payload).hexdigest()}"
    except Exception:
        return f"{rows}|fallback"


def _canonical_symbol_series(df: pd.DataFrame) -> pd.Series | None:
    base_col = None
    for col in ("symbol", "Symbol", "Ticker", "ticker"):
        if col in df.columns:
            base_col = col
            break
    if base_col is None:
        return None
    return (
        df[base_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": None, "NONE": None, "NAN": None})
    )


def _apply_earnings_enrichment(
    df: pd.DataFrame,
    *,
    earn_col_days: str,
    add_earnings_days_column: Callable[[pd.DataFrame], pd.DataFrame],
    quiet_external_calls: Callable[[], Any] | None,
) -> pd.DataFrame:
    try:
        work = df.copy()
        symbols = _canonical_symbol_series(work)
        if symbols is not None:
            work["symbol"] = symbols

        st.session_state["enable_earnings_refresh"] = False

        context = quiet_external_calls() if callable(quiet_external_calls) else nullcontext()
        with context:
            enriched = add_earnings_days_column(work)

        if not isinstance(enriched, pd.DataFrame):
            enriched = work

        earn_series = None
        if earn_col_days in enriched.columns:
            earn_series = enriched[earn_col_days]
        elif "earnings_in_days" in enriched.columns:
            earn_series = enriched["earnings_in_days"]

        out = df.copy()
        if earn_series is not None:
            out[earn_col_days] = pd.to_numeric(earn_series, errors="coerce")
            out["Earnings"] = out[earn_col_days]
        else:
            _ensure_earnings_columns(out, earn_col_days)
        return out
    except Exception:
        out = df.copy()
        _ensure_earnings_columns(out, earn_col_days)
        return out


def _ensure_earnings_columns(df: pd.DataFrame, earn_col_days: str) -> None:
    try:
        if earn_col_days not in df.columns:
            df[earn_col_days] = np.nan
        if "Earnings" not in df.columns:
            df["Earnings"] = np.nan
    except Exception:
        pass


def _apply_earnings_filters(
    df: pd.DataFrame | None,
    *,
    show_earnings: bool,
    earn_col_days: str,
) -> pd.DataFrame | None:
    if not (show_earnings and isinstance(df, pd.DataFrame) and not df.empty and earn_col_days in df.columns):
        return df

    with st.sidebar.expander("📅 Earnings Filters", expanded=False):
        exclude_three = st.checkbox("Exclude earnings in next 3 days", value=False, key="earn_excl_3")
        within_seven = st.checkbox("Only earnings within 7 days", value=False, key="earn_within_7")

    series = pd.to_numeric(df[earn_col_days], errors="coerce")
    if exclude_three:
        df = df[series.isna() | (series > 3)]
        series = pd.to_numeric(df[earn_col_days], errors="coerce")
    if within_seven:
        df = df[(series >= 0) & (series <= 7)]
    return df


def clear_earnings_result_cache() -> None:
    st.session_state.pop("earnings_enriched_df", None)
    st.session_state.pop("earnings_enriched_signature", None)
    st.session_state.pop("earnings_enrichment_rerun_sig", None)


def _clear_earnings_refresh_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("earnings_refresh") or str(key).startswith("earn_refresh"):
            st.session_state.pop(key, None)
