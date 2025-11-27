from __future__ import annotations

import importlib
import inspect
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date

import json

import numpy as np
import pandas as pd
import streamlit as st


# Optional for Yahoo Finance universe fallback
try:
    import requests
except Exception:  # requests may not be installed in some runtimes
    requests = None

try:
    from charts import render_chart_for_ticker
except ModuleNotFoundError:
    from ai_scanner.charts import render_chart_for_ticker
# ============================================
# Breakout Stock Scanner — Subscription Ready
# Single-file entrypoint (replaces bootstrapper)
# ============================================

# ---------- Safe import helpers ----------

def _try_import(path: str, attr: str | None = None):
    """Import a module by dotted path; optionally return a named attribute."""
    try:
        mod = importlib.import_module(path)
        return getattr(mod, attr) if attr else mod
    except Exception:
        return None


def banner(msg: str, level: str = "info"):
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)




# ---------- Page config ----------

st.set_page_config(
    page_title="Breakout Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Tiers / Plans ----------

from config import TIERS_CONFIG, STRIPE_MONTHLY_LINKS, STRIPE_YEARLY_LINKS

# ---------- Local demo user store ----------
# Replace with your real user DB / Firestore later.
# Password hashes should be bcrypt from streamlit-authenticator.

USERS_DB = {
    # -----------------------
    # BASIC TIER USERS
    # -----------------------
    "basic1": {
        "name": "Basic User 1",
        "password": "test123",
        "tier": "basic",
    },
    "basic2": {
        "name": "Basic User 2",
        "password": "test123",
        "tier": "basic",
    },
    "basic3": {
        "name": "Basic User 3",
        "password": "test123",
        "tier": "basic",
    },

    # -----------------------
    # PRO TIER USERS
    # -----------------------
    "pro1": {
        "name": "Pro User 1",
        "password": "pro123",
        "tier": "pro",
    },
    "pro2": {
        "name": "Pro User 2",
        "password": "pro123",
        "tier": "pro",
    },
    "pro3": {
        "name": "Pro User 3",
        "password": "pro123",
        "tier": "pro",
    },

    # -----------------------
    # PREMIUM TIER USERS
    # -----------------------
    "premium1": {
        "name": "Premium User 1",
        "password": "premium123",
        "tier": "premium",
    },
    "premium2": {
        "name": "Premium User 2",
        "password": "premium123",
        "tier": "premium",
    },
    "premium3": {
        "name": "Premium User 3",
        "password": "premium123",
        "tier": "premium",
    },
}
ADMIN_USERS = {"premium1","howard"}  # usernames allowed to access the Admin Users page


@dataclass
class Tier:
    name: str
    can_scan_sp500: bool = True
    can_scan_nasdaq: bool = False
    can_premarket: bool = False
    can_afterhours: bool = False
    can_unusual_volume: bool = False
    can_export_csv: bool = False
    can_ai_notes: bool = False
    max_results: int = 25

def get_user_tier(username: str, users: Dict[str, Dict[str, str]]) -> Tier:
    tier_key = users.get(username, {}).get("tier", "basic")
    cfg = TIERS_CONFIG.get(tier_key, TIERS_CONFIG["basic"])
    return Tier(
        name=cfg["name"],
        can_scan_sp500=("SP500 Scan" in cfg["features"]),
        can_scan_nasdaq=("NASDAQ" in cfg["features"]),
        can_premarket=("Premarket" in cfg["features"]),
        can_afterhours=("AfterHours" in cfg["features"]),
        can_unusual_volume=("UnusualVolume" in cfg["features"]),
        can_export_csv=("ExportCSV" in cfg["features"]),
        can_ai_notes=("AI Notes" in cfg["features"]),
        max_results=cfg.get("max_results", 25),
    )




# ---------- Auth ----------
try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None
import streamlit as st
from db.users import seed_neon_users_from_local, load_users, fetch_all_users
from db.runs import save_run, save_daily_snapshot, list_runs, load_run_results

from db.engine import get_db_status, get_neon_conn

try:
    from ui.universe import (
        load_sp500_universe,
        load_nasdaq_universe,
        filter_universe,
        apply_liquidity_filter_batch,
    )
except ModuleNotFoundError:
    # Fallback for environments where `ai_scanner` is the package root
    from ai_scanner.ui.universe import (
        load_sp500_universe,
        load_nasdaq_universe,
        filter_universe,
        apply_liquidity_filter_batch,
    )
from scan.engine import safe_call, cached_real_scan, _override_last_prices, safe_yf_download

# Optional live price override for the 'Last' column
try:
    import yfinance as yf
except Exception:
    yf = None


def auth_ui() -> Tuple[bool, Optional[str], Optional[str]]:
    """Returns (authenticated, username, display_name)."""
    if stauth is None:
        banner("streamlit-authenticator not installed. Running in DEMO mode.", "warning")
        return True, "howard", "Howard"

    users_map = load_users()
    usernames = list(users_map.keys())
    authenticator = stauth.Authenticate(
        {"usernames": {u: {"name": users_map[u]["name"], "password": users_map[u]["password"]} for u in usernames}},
        "breakout_scanner_cookie",
        "breakout_scanner_signature",
        cookie_expiry_days=7,
    )

    # New API (v0.3+): login() returns None for rendered locations; values are in st.session_state
    try:
        authenticator.login(
            "main",
            fields={
                "Form name": "Login",
                "Username": "Username",
                "Password": "Password",
                "Login": "Login",
            },
        )
    except Exception as e:
        banner(f"Auth error: {e}", "error")
        return False, None, None

    auth_status = st.session_state.get("authentication_status", None)
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if auth_status is False:
        banner("Username/password incorrect", "error")
        return False, None, None
    if auth_status is None:
        banner("Please enter your credentials.", "info")
        return False, None, None

    with st.sidebar:
        authenticator.logout("Logout", "sidebar")

    return True, username, name


def pricing_sidebar(current_username: Optional[str], users: Dict[str, Dict[str, str]]):
    """Show upgrade options only for tiers above the user's current plan.

    - Basic users see Pro + Premium
    - Pro users see Premium
    - Premium users see a small thank-you message (no upgrade cards)
    - Includes a Monthly / Yearly toggle that switches Stripe links.
    """
    tiers_order = ["basic", "pro", "premium"]
    current_key = users.get(current_username or "", {}).get("tier", "basic")
    try:
        start_idx = tiers_order.index(current_key) + 1
    except ValueError:
        start_idx = 1  # default: treat as basic if unknown

    upsell_keys = tiers_order[start_idx:]

    st.sidebar.markdown("## 💳 Upgrade")

    if not upsell_keys:
        st.sidebar.caption("You're on the top Premium plan. Thank you for subscribing!")
        return

    # Billing period toggle
    billing_period = st.sidebar.radio(
        "Billing period",
        ["Monthly", "Yearly"],
        index=0,
        horizontal=True,
    )

    cols = st.sidebar.columns(len(upsell_keys))
    for i, key in enumerate(upsell_keys):
        cfg = TIERS_CONFIG.get(key, {})
        name = cfg.get("name", key.title())
        monthly_price = cfg.get("price_monthly", 0)
        yearly_price = cfg.get("price_yearly", 0)
        features = cfg.get("features", [])
        with cols[i]:
            # Center-align everything in this card
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

            st.markdown(f"**{name}**")

            # Best value badge for Pro on Yearly
            if key == "pro" and billing_period == "Yearly":
                st.markdown(
                    "<div style='font-size: 0.8rem; color: #22c55e; font-weight: 600; margin-bottom: 0.2rem;'>"
                    "⭐ Best value"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # Show only the selected billing period price
            if billing_period == "Monthly":
                st.markdown(f"<div style='font-size: 1.1rem; font-weight: 700;'>${monthly_price}/mo</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='font-size: 1.1rem; font-weight: 700; color: #22c55e;'>${yearly_price}/yr</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-size: 0.8rem; color: #9ca3af;'>≈ 2 months free vs monthly.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"- SP500: {'✅' if ('SP500 Scan' in features or 'SP500' in features) else '❌'}")
            st.markdown(f"- NASDAQ: {'✅' if 'NASDAQ' in features else '❌'}")
            st.markdown(f"- Export: {'✅' if 'ExportCSV' in features else '❌'}")
            if 'AI Notes' in features:
                st.markdown("- AI Notes: ✅")
            else:
                st.markdown("- AI Notes: ❌")

            # Choose the correct Stripe link key based on billing period
            checkout_url = (STRIPE_MONTHLY_LINKS if billing_period == "Monthly" else STRIPE_YEARLY_LINKS).get(key)

            if checkout_url:
                st.link_button(
                    f"Subscribe {name} ({billing_period})",
                    checkout_url,
                )
            else:
                st.caption("Stripe link not configured yet for this plan/period.")




def generate_ai_note(row: pd.Series) -> str:
    """Heuristic, on-device "AI" note using the breakout metrics for one ticker.

    This keeps the app self-contained (no external API calls) and explains why a
    name is scoring well or poorly based on its columns.
    """
    def _fmt(val, nd=2, suffix=""):
        try:
            if pd.isna(val):
                return "N/A"
            return f"{float(val):.{nd}f}{suffix}"
        except Exception:
            return "N/A"

    ticker = row.get("Ticker", "?")
    score = row.get("BreakoutScore", np.nan)
    pattern = row.get("PatternTag", "") or "Neutral"
    gap = row.get("Gap%", np.nan)
    trend20 = row.get("Trend20D%", np.nan)
    trend10 = row.get("Trend10D%", np.nan)
    vol_rel = row.get("VolRel20", np.nan)
    rs = row.get("RS_Rank", np.nan)
    vol20 = row.get("Volatility20D%", np.nan)
    dollar_vol = row.get("DollarVol20", np.nan)

    parts = []

    # High-level summary
    parts.append(
        f"**{ticker}** currently has a BreakoutScore of **{_fmt(score, 1)}** with pattern tag **{pattern}**."
    )

    # Trend + relative strength
    trend_bits = []
    if not pd.isna(trend20):
        trend_bits.append(f"~{_fmt(trend20, 1, '%')} over the last 20 days")
    if not pd.isna(trend10):
        trend_bits.append(f"~{_fmt(trend10, 1, '%')} over the last 10 days")
    if trend_bits:
        line = "Price trend: " + ", ".join(trend_bits) + "."
        parts.append(line)

    if not pd.isna(rs):
        try:
            rs_val = float(rs)
            if rs_val >= 80:
                rs_comment = "strong relative strength vs the universe (top 20%)."
            elif rs_val >= 60:
                rs_comment = "above-average relative strength (top 40%)."
            elif rs_val >= 40:
                rs_comment = "roughly middle-of-the-pack relative strength."
            else:
                rs_comment = "weak relative strength vs peers right now."
            parts.append(f"RS Rank is **{_fmt(rs, 1)}**, indicating {rs_comment}")
        except Exception:
            pass

    # Gap + volume behaviour
    gap_bits = []
    if not pd.isna(gap):
        gap_bits.append(f"gap of {_fmt(gap, 1, '%')} vs the prior close")
    if not pd.isna(vol_rel):
        gap_bits.append(f"volume running at roughly {_fmt(vol_rel, 2)}x the 20D average")
    if gap_bits:
        parts.append("Today it is showing a " + " and ".join(gap_bits) + ".")

    if not pd.isna(dollar_vol):
        parts.append(
            f"Liquidity check: 20D avg dollar volume is around **${_fmt(dollar_vol/1_000_000, 1)}M**, "
            "which helps with entries and exits."
        )

    if not pd.isna(vol20):
        try:
            vol_val = float(vol20)
            if vol_val <= 8:
                vol_comment = "Price action has been relatively quiet (low volatility)."
            elif vol_val <= 18:
                vol_comment = "Volatility is moderate and tradable for most setups."
            else:
                vol_comment = "This is a high-volatility name; position sizing and risk management are critical."
            parts.append(f"Volatility (20D) sits near **{_fmt(vol20, 1, '%')}**, {vol_comment}")
        except Exception:
            pass

    parts.append(
        "This is not a trade recommendation. Consider support/resistance on the chart, overall market context, "
        "and your own risk management rules before acting."
    )

    return "\n\n".join(parts)

# ---------- Main UI ----------

def main():
    st.title("📈 Breakout Stock Scanner")
    st.caption("Money Moves • AI Breakout Score • Subscription Ready")

    authed, username, display_name = auth_ui()
    if not authed:
        st.stop()

    # Seed Neon users table once (no-op if already populated or Neon unavailable)
    try:
        if not st.session_state.get("seeded_neon_users"):
            seed_neon_users_from_local()
            st.session_state["seeded_neon_users"] = True
    except Exception:
        pass

    # Load users once per rerun to avoid redundant Neon hits
    users_map = load_users()

    tier = get_user_tier(username, users_map)

    st.sidebar.markdown(f"### 👤 {display_name}")
    if username in ADMIN_USERS:
        st.sidebar.markdown("**Plan:** `Admin`")
    else:
        st.sidebar.markdown(f"**Plan:** `{tier.name}`")

    # DB status badge
    try:
        db_status = get_db_status()
    except Exception:
        db_status = "none"

    if db_status == "neon":
        st.sidebar.markdown("🟢 **DB:** Neon (cloud)")
    elif db_status == "sqlite":
        st.sidebar.markdown("🟡 **DB:** Local SQLite")
    else:
        st.sidebar.markdown("🔴 **DB:** Unavailable")

    pricing_sidebar(username, users_map)

    # Sidebar filters
    st.sidebar.markdown("## Filters")
    min_gap = st.sidebar.slider("Min Gap %", -10.0, 20.0, 1.0, 0.5)
    min_price = st.sidebar.number_input("Min Price", 0.5, 500.0, 1.0, 0.5)
    max_price = st.sidebar.number_input("Max Price", 1.0, 5000.0, 1000.0, 1.0)
    top_n = st.sidebar.slider("Top N Results", 5, tier.max_results, min(25, tier.max_results), 5)

    max_nasdaq_scan = st.sidebar.number_input(
        "Max NASDAQ tickers to scan",
        min_value=100,
        max_value=6000,
        value=1200,
        step=100,
        help="Caps NASDAQ universe to speed up scans. Applied to NASDAQ + Combo scans.",
    )

    max_combo_scan = st.sidebar.number_input(
        "Max Combo tickers to scan",
        min_value=100,
        max_value=6000,
        value=1000,
        step=100,
        help="Caps SP500+NASDAQ universe for Combo scans.",
    )

    premarket = st.sidebar.checkbox("Include Premarket Scan", value=False, disabled=not tier.can_premarket)
    afterhours = st.sidebar.checkbox("Include After-hours Scan", value=False, disabled=not tier.can_afterhours)
    unusual_vol = st.sidebar.checkbox("Unusual Volume Filter", value=False, disabled=not tier.can_unusual_volume)

    st.sidebar.divider()
    diagnostics = st.sidebar.checkbox("Show diagnostics", value=True)

    # Universe state (lazy-loaded on first scan to keep startup fast)
    if "sp500_universe" not in st.session_state:
        st.session_state["sp500_universe"] = []
    if "nasdaq_universe" not in st.session_state:
        st.session_state["nasdaq_universe"] = []
    if "nasdaq_capped" not in st.session_state:
        st.session_state["nasdaq_capped"] = []
    if "combo_capped" not in st.session_state:
        st.session_state["combo_capped"] = []

    # Universe diagnostics (lazy; based on last scan)
    with st.expander("Universe Info", expanded=True):
        sp500 = st.session_state.get("sp500_universe", [])
        nasdaq_full = st.session_state.get("nasdaq_universe", [])
        nasdaq_capped = st.session_state.get("nasdaq_capped", [])

        if sp500 or nasdaq_full:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**SP500 size:** {len(sp500)}" if sp500 else "**SP500 size:** (not loaded yet)")
                if sp500:
                    st.caption(f"Sample: {', '.join(sp500[:20])}")
            with c2:
                if nasdaq_full:
                    st.markdown(f"**NASDAQ size:** {len(nasdaq_capped) or len(nasdaq_full)}"
                                f"{' (capped)' if nasdaq_capped else ''}")
                    st.caption(f"Sample: {', '.join((nasdaq_capped or nasdaq_full)[:20])}")
                else:
                    st.markdown("**NASDAQ size:** (not loaded yet)")
                    st.caption("Run a NASDAQ or Combo scan to populate NASDAQ universe.")
        else:
            st.caption("Universes will appear here after you run your first scan (SP500, NASDAQ, or Combo).")

    # Buttons (hard-wired universes)
    b1, b2, b3 = st.columns([1, 1, 2])

    with b1:
        run_sp500_btn = st.button("Run SP500 Scan", use_container_width=True, disabled=not tier.can_scan_sp500)
        st.caption("Runs SP500 regardless of sidebar universe.")

    with b2:
        run_nasdaq_btn = st.button("Run NASDAQ Scan", use_container_width=True, disabled=not tier.can_scan_nasdaq)
        st.caption("Runs NASDAQ regardless of sidebar universe.")

    with b3:
        run_combo_btn = st.button(
            "Run Combo Scan (SP500+NASDAQ)",
            use_container_width=True,
            disabled=not (tier.can_scan_sp500 and tier.can_scan_nasdaq),
        )
        st.caption("Pro/Premium only.")

    # Session state for results
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    def do_scan(tickers: List[str], label: str):
        def _run_scan_body():
            n_input = len(tickers)
            t0 = time.time()
            try:
                st.caption(f"🔎 Scanning {len(tickers)} tickers for {label}...")
                if len(tickers) < 50:
                    st.warning(
                        f"{label} universe is very small ({len(tickers)} tickers). "
                        "This usually means a fallback/stub universe is still being used."
                    )

                df = safe_call(
                    cached_real_scan,
                    tuple(tickers),
                    premarket=premarket,
                    afterhours=afterhours,
                    unusual_volume=unusual_vol,
                    min_gap=min_gap,
                    min_price=min_price,
                    max_price=max_price,
                    top_n=top_n,
                    diagnostics=diagnostics,
                    label="cached_real_scan",
                )

                # Apply Top N cap here to avoid doing last-price overrides on hundreds of rows.
                if df is not None and not df.empty:
                    df = df.head(top_n).reset_index(drop=True)

                    if premarket or afterhours:
                        df = _override_last_prices(df)

                filtered_count = len(df) if df is not None else 0
                if diagnostics:
                    st.caption(f"📊 Filtered down from {n_input} tickers to {filtered_count} results after filters.")

                st.caption(f"✅ {label}: {len(df)} results returned from scan.")
                dt = time.time() - t0
                st.session_state.results_df = df
                banner(f"✅ {label} scan complete in {dt:.1f}s. Returned {len(df)} rows.", "success")
                # Persist this scan to the runs DB (history + optional daily snapshot)
                try:
                    results_json = df.to_json(orient="records") if df is not None else "[]"
                    row_count = len(df) if df is not None else 0
                    run_name = f"{label} | {row_count} results | {dt:.1f}s"
                    save_run(
                        run_name,
                        results_json,
                        label=label,
                        username=username,
                        row_count=row_count,
                        duration_sec=dt,
                        is_snapshot=False,
                    )

                    # Morning snapshot: one per day per label (approx. before noon server time)
                    try:
                        current_hour = datetime.now().hour
                        if current_hour < 12:
                            save_daily_snapshot(label, results_json, username=username)
                    except Exception:
                        # Snapshot is best-effort only
                        pass
                except Exception:
                    # Never fail the UI just because DB logging failed
                    pass

                # Clear cached run list so new scan appears immediately in history
                try:
                    list_runs.clear()  # type: ignore
                except Exception:
                    pass
            except Exception as e:
                banner(f"❌ Scan failed: {e}", "error")
                if diagnostics:
                    st.code(traceback.format_exc())

        # Some environments (e.g., restricted sandboxes, Python 3.13 runtimes) may not
        # allow starting new threads, which Streamlit's spinner uses internally.
        # Wrap the spinner in a try/except and fall back to running without it.
        try:
            with st.spinner(f"Scanning {label}..."):
                _run_scan_body()
        except Exception:
            _run_scan_body()

    if run_sp500_btn:
        sp500 = safe_call(load_sp500_universe, label="SP500 universe")
        sp500 = filter_universe(sp500)
        st.session_state["sp500_universe"] = sp500
        do_scan(sp500, "SP500")

    if run_nasdaq_btn:
        nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")
        nasdaq = filter_universe(nasdaq)
        nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        do_scan(nasdaq_capped, "NASDAQ")

    if run_combo_btn:
        sp500 = safe_call(load_sp500_universe, label="SP500 universe")
        nasdaq = safe_call(load_nasdaq_universe, label="NASDAQ universe")
        sp500 = filter_universe(sp500)
        nasdaq = filter_universe(nasdaq)
        nasdaq_capped = nasdaq[: int(max_nasdaq_scan)]
        combo_universe = sp500 + nasdaq_capped
        combo_liquid = apply_liquidity_filter_batch(combo_universe)
        combo_capped = combo_liquid[: int(max_combo_scan)]

        st.session_state["sp500_universe"] = sp500
        st.session_state["nasdaq_universe"] = nasdaq
        st.session_state["nasdaq_capped"] = nasdaq_capped
        st.session_state["combo_capped"] = combo_capped

        do_scan(combo_capped, "Combo")

    df = st.session_state.results_df

    if df is not None and not df.empty:
        st.subheader("Results")
        st.caption(
            f"Showing {len(df)} results. Increase 'Top N Results' in the sidebar to see more, "
            "or relax filters (Min Gap %, price range, Unusual Volume Filter). "
            "If you see 0 results, try lowering Min Gap or turning off the Unusual Volume Filter."
        )

        # --- Pro styling for results table ---
        styled = df.style

        # Heatmap for BreakoutScore
        if "BreakoutScore" in df.columns:
            styled = styled.background_gradient(axis=None, cmap="RdYlGn", subset=["BreakoutScore"])

        # Conditional formatting for RS_Rank (0-100)
        if "RS_Rank" in df.columns:
            styled = styled.background_gradient(axis=None, cmap="Greens", subset=["RS_Rank"])

        # Bold / color trend markers
        def _trend_style(series: pd.Series):
            styles = []
            for v in series:
                try:
                    val = float(v)
                except Exception:
                    styles.append("")
                    continue
                if val >= 20:
                    styles.append("font-weight: bold; color: #006400;")  # strong uptrend
                elif val <= -10:
                    styles.append("font-weight: bold; color: #8B0000;")  # strong downtrend
                else:
                    styles.append("")
            return styles

        if "Trend20D%" in df.columns:
            styled = styled.apply(_trend_style, subset=["Trend20D%"])
        if "Trend10D%" in df.columns:
            styled = styled.apply(_trend_style, subset=["Trend10D%"])

        st.dataframe(styled, use_container_width=True, height=420)

        # Export (tier-gated)
        if tier.can_export_csv:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name="breakout_results.csv",
                mime="text/csv",
                use_container_width=False,
            )
        else:
            st.info("CSV export is available on Pro/Premium.")

        # Chart picker
        st.subheader("Charts")
        pick = st.selectbox("Select ticker to chart", df["Ticker"].tolist())
        render_chart_for_ticker(pick)

        # AI notes (tier-gated)
        if tier.can_ai_notes:
            st.subheader("AI Notes (Premium)")
            try:
                # Use the same ticker the user selected for the chart
                row = df[df["Ticker"] == pick].iloc[0]
                auto_note = generate_ai_note(row)
                st.markdown(auto_note)
                st.text_area(
                    "Edit or copy these notes (Premium only):",
                    value=auto_note,
                    height=220,
                )
            except Exception:
                st.caption("AI notes are unavailable for the selected row.")
        else:
            st.caption("AI Notes are Premium-only.")
    else:
        st.caption("Run a scan to see results.")

    # --- Scan History (DB-backed via local SQLite) ---
    with st.expander("📜 Scan History", expanded=False):
        runs_list = []
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

    # --- Admin Users Page ---
    if username in ADMIN_USERS:
        with st.expander("👑 Admin: Manage Users", expanded=False):
            # Avoid Neon hits on every rerun; only load when admin opts in.
            enable_admin = st.checkbox(
                "Enable admin user management",
                value=False,
                key="enable_admin_users",
            )

            if not enable_admin:
                st.caption("Toggle the switch above to load and manage Neon users.")
            else:
                # --- Create New User ---
                st.subheader("➕ Create New User")

                new_username = st.text_input("New Username")
                new_full_name = st.text_input("Full Name")
                new_password = st.text_input("Password", type="password")
                new_tier_create = st.selectbox("Tier", ["basic", "pro", "premium"], key="create_user_tier")
                new_active_create = st.checkbox("Active", value=True, key="create_user_active")

                if st.button("Create User"):
                    if not new_username or not new_full_name or not new_password:
                        st.error("All fields are required.")
                    else:
                        try:
                            conn = get_neon_conn()
                            if conn is None:
                                st.error("Neon connection unavailable; cannot create user.")
                            else:
                                ensure_neon_users_schema(conn)
                                cur = conn.cursor()

                                # Hash password before storing in Neon when auth library is available
                                pwd_to_store = new_password
                                try:
                                    if stauth is not None:
                                        pwd_to_store = stauth.Hasher([new_password]).generate()[0]
                                except Exception:
                                    # If hashing fails, fall back to raw (not ideal but avoids blocking admin)
                                    pwd_to_store = new_password

                                cur.execute(
                                    """
                                    INSERT INTO users (username, full_name, password, tier, is_active)
                                    VALUES (%s, %s, %s, %s, %s)
                                    ON CONFLICT (username) DO NOTHING
                                    """,
                                    (new_username, new_full_name, pwd_to_store, new_tier_create, new_active_create),
                                )
                                conn.commit()
                                cur.close()
                                conn.close()

                                # Clear cache so new user is available immediately
                                try:
                                    load_users.clear()  # type: ignore
                                except Exception:
                                    pass

                                st.success(f"User '{new_username}' created successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create user: {e}")

                # --- Existing Users Table + Edit UI ---
                users_df = fetch_all_users()
                if users_df is None or users_df.empty:
                    st.caption("No users found in Neon users table.")
                else:
                    st.caption("View and edit user tiers. Changes apply to Neon-backed accounts.")
                    desired_cols = ["id", "username", "full_name", "tier", "is_active", "created_at"]
                    display_cols = [c for c in desired_cols if c in users_df.columns]
                    st.dataframe(
                        users_df[display_cols],
                        use_container_width=True,
                        height=260,
                    )

                    usernames_list = users_df["username"].tolist()
                    selected_user = st.selectbox("Select user to edit", usernames_list)
                    row = users_df[users_df["username"] == selected_user].iloc[0]

                    new_tier = st.selectbox(
                        "Tier",
                        ["basic", "pro", "premium"],
                        index=["basic", "pro", "premium"].index(
                            row["tier"] if row["tier"] in ["basic", "pro", "premium"] else "basic"
                        ),
                    )
                    new_active = st.checkbox("Active", value=bool(row["is_active"]))

                    if st.button("Update User"):
                        try:
                            conn = get_neon_conn()
                            if conn is None:
                                st.error("Neon connection unavailable; cannot update user.")
                            else:
                                ensure_neon_users_schema(conn)
                                cur = conn.cursor()
                                cur.execute(
                                    """
                                    UPDATE users
                                    SET tier = %s,
                                        is_active = %s
                                    WHERE username = %s
                                    """,
                                    (new_tier, new_active, selected_user),
                                )
                                conn.commit()
                                cur.close()
                                conn.close()

                                try:
                                    load_users.clear()  # type: ignore
                                except Exception:
                                    pass

                                st.success(f"User '{selected_user}' updated successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update user: {e}")

    st.divider()
    st.caption("⚠️ Not financial advice. Educational tool only.")


if __name__ == "__main__":
    main()