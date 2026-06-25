"""Premium AI insights: scan diff, watchlist digest, and alert rationale.

All thin wrappers over ui.ai.ask_claude, so they inherit the global kill
switch, per-user daily cap, and request timeout automatically.
"""
from __future__ import annotations

import pandas as pd

_DIFF_COLS = ["Ticker", "Symbol", "BreakoutScore", "Gap%", "Trend20D%", "VolRel20", "DollarVol20"]
_SCORE_COL = "BreakoutScore"

_DIFF_SYSTEM = (
    "You are a concise equity-scan analyst. You are given two scans of the same "
    "screener: a PREVIOUS run and the CURRENT run. Summarize what changed in "
    "plain language: notable NEW entries, tickers that DROPPED OUT, and the "
    "biggest breakout-score moves. Group by theme/sector when obvious. Keep it "
    "under 200 words, markdown bullets, end with one risk caveat. Educational "
    "technical commentary only — no financial advice or price targets."
)

_DIGEST_SYSTEM = (
    "You are a concise equity-scan analyst. You are given the current scan "
    "metrics for the stocks on a user's watchlist. Give a short morning-style "
    "digest: which watchlist names look strongest right now and why, and which "
    "look weak or risky. Reference the numbers. Under 180 words, markdown "
    "bullets, end with one risk caveat. Educational only — no advice/targets."
)

_ALERT_SYSTEM = (
    "You write a single-sentence alert rationale (max 20 words) explaining why a "
    "stock's technical setup is notable right now, referencing one or two "
    "specific metrics. No advice, no targets, no preamble — just the sentence."
)


def _current_user() -> str | None:
    try:
        import streamlit as st
        return st.session_state.get("username")
    except Exception:
        return None


def _table(df: pd.DataFrame, max_rows: int = 20) -> str:
    cols = [c for c in _DIFF_COLS if c in df.columns]
    sub = df[cols] if cols else df
    return sub.head(max_rows).to_csv(index=False)


# ---------------------------------------------------------------------------
# 1) What changed since last scan
# ---------------------------------------------------------------------------

def generate_scan_diff(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> tuple[str | None, str | None]:
    if curr_df is None or len(curr_df) == 0:
        return None, "No current scan to compare. Run a scan first."
    if prev_df is None or len(prev_df) == 0:
        return None, "No previous scan/snapshot found to compare against."

    from ui.ai import ask_claude
    return ask_claude(
        system=_DIFF_SYSTEM,
        user=(
            f"PREVIOUS scan (CSV):\n{_table(prev_df)}\n\n"
            f"CURRENT scan (CSV):\n{_table(curr_df)}\n\n"
            "Summarize what changed."
        ),
        max_tokens=900,
        username=_current_user(),
    )


def render_scan_diff(curr_df, load_run_results, list_runs, normalize_results_to_df, username: str) -> None:
    """Compare the current scan to the most recent prior snapshot and narrate."""
    import streamlit as st

    st.markdown("#### 🔀 What changed since last scan")
    st.caption("Claude compares this scan to your most recent saved snapshot.")

    if st.button("📊 Show what changed", key="ai_diff_btn"):
        prev_df = None
        try:
            runs = list_runs(limit=10, username=username) if list_runs else []
            snap = next((r for r in runs if r.get("is_snapshot")), None) or (runs[0] if runs else None)
            if snap and load_run_results:
                raw = load_run_results(snap["id"])
                prev_df = normalize_results_to_df(raw) if raw else None
        except Exception:
            prev_df = None

        with st.spinner("Comparing scans…"):
            text, err = generate_scan_diff(prev_df, curr_df)
        if text:
            st.markdown(text)
        else:
            st.warning(err or "Could not generate a comparison.")


# ---------------------------------------------------------------------------
# 2) Watchlist digest
# ---------------------------------------------------------------------------

def _filter_to_watchlist(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if df is None or len(df) == 0 or not tickers:
        return pd.DataFrame()
    want = {str(t).strip().upper() for t in tickers}
    col = "Ticker" if "Ticker" in df.columns else ("Symbol" if "Symbol" in df.columns else None)
    if col is None:
        return pd.DataFrame()
    return df[df[col].astype(str).str.upper().isin(want)]


def generate_watchlist_digest(tickers: list[str], df: pd.DataFrame) -> tuple[str | None, str | None]:
    sub = _filter_to_watchlist(df, tickers)
    if len(sub) == 0:
        return None, "None of your watchlist tickers are in the latest scan. Run a scan that covers them."
    from ui.ai import ask_claude
    return ask_claude(
        system=_DIGEST_SYSTEM,
        user=f"Watchlist scan metrics (CSV):\n{_table(sub)}\n\nGive the watchlist digest.",
        max_tokens=700,
        username=_current_user(),
    )


def render_watchlist_digest(tickers: list[str], df) -> None:
    import streamlit as st

    st.markdown("#### 📋 AI watchlist digest")
    st.caption("Claude reviews your watchlist names against the latest scan.")
    if st.button("✨ Generate digest", key="ai_wl_digest_btn"):
        with st.spinner("Reviewing your watchlist…"):
            text, err = generate_watchlist_digest(tickers, df)
        if text:
            st.markdown(text)
        else:
            st.warning(err or "Could not generate a digest.")


# ---------------------------------------------------------------------------
# 3) AI alert rationale (one-liners, optionally pushed to Slack)
# ---------------------------------------------------------------------------

def generate_alert_line(ticker: str, row) -> tuple[str | None, str | None]:
    try:
        fields = {c: row.get(c) for c in _DIFF_COLS if hasattr(row, "get") and row.get(c) is not None}
    except Exception:
        fields = {}
    metrics = ", ".join(f"{k}={v}" for k, v in fields.items()) or "no metrics"
    from ui.ai import ask_claude
    return ask_claude(
        system=_ALERT_SYSTEM,
        user=f"{ticker}: {metrics}",
        max_tokens=80,
        username=_current_user(),
    )


def render_watchlist_alert_preview(tickers: list[str], df, *, max_alerts: int = 5) -> None:
    """Show AI alert-style one-liners for top watchlist movers; optional Slack push."""
    import streamlit as st

    st.markdown("#### 🔔 Watchlist alert preview")
    st.caption("One-line AI rationale per top watchlist mover. Optionally send to Slack.")

    if not st.button("📝 Build alert preview", key="ai_alert_preview_btn"):
        return

    sub = _filter_to_watchlist(df, tickers)
    if len(sub) == 0:
        st.warning("No watchlist tickers in the latest scan.")
        return
    if _SCORE_COL in sub.columns:
        sub = sub.sort_values(_SCORE_COL, ascending=False)

    col = "Ticker" if "Ticker" in sub.columns else "Symbol"
    lines: list[str] = []
    with st.spinner("Writing alert lines…"):
        for _, row in sub.head(max_alerts).iterrows():
            tkr = str(row.get(col, "?"))
            line, err = generate_alert_line(tkr, row)
            if line:
                lines.append(f"*{tkr}* — {line}")
            elif err and ("limit" in err or "disabled" in err):
                st.warning(err)
                break

    if not lines:
        st.info("No alert lines generated.")
        return

    for ln in lines:
        st.markdown(f"- {ln}")

    if st.button("📤 Send to Slack", key="ai_alert_send_slack"):
        try:
            from telemetry import send_slack_alert
            msg = "🔔 *Watchlist alerts*\n" + "\n".join(f"• {ln}" for ln in lines)
            ok = send_slack_alert(msg)
            st.success("Sent to Slack.") if ok else st.warning(
                "Slack not configured (set SLACK_WEBHOOK_URL)."
            )
        except Exception as e:
            st.warning(f"Could not send to Slack: {e}")
