"""Personalized watchlist intelligence surface for Run 27.

Renders existing HSF intelligence around the user's watchlist. Reads persisted
watchlists, opportunity snapshots, and intelligence alerts; does not scan,
score, freeze, mature outcomes, deliver alerts, or call Claude.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

_STATUS_ICON = {"STRONG": "🟢", "WATCH": "🟡", "CAUTION": "🟠"}


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "—"


def _freshness(row: Dict[str, Any]) -> str:
    minutes = row.get("last_updated_minutes")
    if minutes is None:
        return "Freshness unavailable"
    if minutes < 90:
        return f"Updated {minutes}m ago"
    return f"Updated {int(minutes) // 60}h ago"


def _set_stock(ticker: str) -> None:
    st.session_state["hsf_stock_ticker"] = str(ticker).strip().upper()
    try:
        st.switch_page("pages/stock.py")
    except Exception:
        st.caption("Open Stock Intel from the sidebar.")


def _watchlist_badge(row: Dict[str, Any]) -> str:
    status = row.get("hsf_status")
    score = _fmt_score(row.get("hsf_score"))
    if status:
        return f"HSF Score {score} · {status}"
    return "Not currently ranked"


def _render_row(row: Dict[str, Any], *, key_prefix: str) -> None:
    ticker = row["ticker"]
    status = row.get("hsf_status")
    icon = _STATUS_ICON.get(str(status or "").upper(), "•")
    with st.container(border=True):
        top = st.columns([1.1, 1.6, 2.8, 1.2])
        top[0].markdown(f"### {ticker}")
        top[1].markdown(f"{icon} **{_watchlist_badge(row)}**")
        top[2].caption(row.get("attention_reason") or "No recent HSF change detected.")
        if top[3].button("Full Intelligence", key=f"{key_prefix}_intel_{ticker}", width="stretch"):
            _set_stock(ticker)

        bits: List[str] = []
        if row.get("score_delta") is not None:
            try:
                bits.append(f"HSF Score {int(row['score_delta']):+d}")
            except (TypeError, ValueError):
                pass
        signals = row.get("confirming_signals") or []
        if signals:
            bits.append(f"{len(signals)} confirming signals")
        if row.get("fading"):
            bits.append("FADING")
        if row.get("recent_alert"):
            bits.append(f"Recent intelligence alert: {row['recent_alert'].get('event_type')}")
        bits.append(_freshness(row))
        st.caption(" · ".join(bits))


def _render_empty_watchlist() -> None:
    st.info("Build your watchlist to see the HSF intelligence that matters to you.")
    c1, c2, c3 = st.columns(3)
    c1.page_link("pages/brief.py", label="Browse Market Brief", icon="📬")
    c2.page_link("app.py", label="Open Scanner", icon="🔎")
    c3.page_link("pages/stock.py", label="Search a ticker", icon="🔬")


def render_personal_watchlist(user_id: str) -> None:
    if st is None:
        return
    user = str(user_id or "").strip().lower()
    if not user:
        st.info("Please log in to view your watchlist intelligence.")
        return
    try:
        from analytics.watchlist_intelligence import build_watchlist_intelligence

        intel = build_watchlist_intelligence(user)
    except Exception as exc:
        st.warning("Watchlist intelligence is temporarily unavailable.")
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    summary = intel.get("summary") or {}
    rows = intel.get("rows") or []
    st.markdown("## My Watchlist")
    if not rows:
        _render_empty_watchlist()
        return

    st.caption(
        f"{summary.get('tracked', 0)} stocks tracked · "
        f"{summary.get('needs_attention', 0)} need attention · "
        f"{summary.get('strengthening', 0)} strengthened or entered HSF opportunities · "
        f"{summary.get('fading', 0)} fading · "
        f"{summary.get('stable', 0)} stable or quiet"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked", summary.get("tracked", 0))
    c2.metric("Needs Attention", summary.get("needs_attention", 0))
    c3.metric("Strengthening", summary.get("strengthening", 0))
    c4.metric("Fading", summary.get("fading", 0))

    search = st.text_input("Search watched stocks", placeholder="Ticker", key="my_watchlist_search")
    try:
        view = st.segmented_control(
            "View",
            ["Attention", "All", "Stable"],
            default="Attention",
            key="my_watchlist_view",
        )
    except AttributeError:  # pragma: no cover - older Streamlit fallback
        view = st.radio(
            "View",
            ["Attention", "All", "Stable"],
            horizontal=True,
            key="my_watchlist_view",
        )
    filtered = rows
    if search.strip():
        q = search.strip().upper()
        filtered = [row for row in filtered if q in row["ticker"]]
    if view == "Attention":
        visible = intel.get("groups", {}).get("needs_attention") or []
        visible += intel.get("groups", {}).get("improving") or []
        visible = [row for row in visible if row in filtered]
        if not visible:
            st.caption("No meaningful HSF changes detected recently.")
    elif view == "Stable":
        visible = [
            row
            for group in ("stable", "quiet")
            for row in (intel.get("groups", {}).get(group) or [])
            if row in filtered
        ]
    else:
        visible = filtered

    for row in visible:
        _render_row(row, key_prefix="my_watchlist")

    if intel.get("recent_alerts"):
        with st.expander("Recent HSF Intelligence Alerts", expanded=False):
            for alert in intel["recent_alerts"][:10]:
                st.caption(
                    f"{str(alert.get('ticker') or '').upper()} · "
                    f"{alert.get('event_type')} · {alert.get('copy') or alert.get('severity') or ''}"
                )
