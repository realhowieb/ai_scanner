"""Alpaca paper-trading UI: connect account + confirmed market order.

Thin vertical slice (Phase 1a). Premium-gated. Flow:
  1. Connect: user enters their OWN Alpaca paper key/secret; we validate against
     the paper endpoint, then store them encrypted (db.paper_trading).
  2. Order: a "Paper Trade This Setup" button beside the trade plan opens an
     inline confirmation. Nothing is sent until the user explicitly confirms —
     every order requires that confirmation. On fill the trade imports into the
     journal with the scanner BreakoutScore and AI confidence attached.

Never raises into the results view; degrades to a caption when unavailable.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _num(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        try:
            if key not in row:
                continue
            f = float(row[key])
        except (TypeError, ValueError):
            continue
        if f == f:  # not NaN
            return f
    return None


def _user() -> str:
    try:
        return (st.session_state.get("username") or "").strip().lower()
    except Exception:
        return ""


def _can_paper_trade() -> bool:
    try:
        ent = st.session_state.get("entitlements") or {}
        return bool(ent.get("can_paper_trade"))
    except Exception:
        return False


def render_connect_panel(user_id: str) -> None:
    """Account-connection panel (settings/account area). Premium-gated."""
    if st is None or not user_id:
        return
    if not _can_paper_trade():
        return
    try:
        from db.paper_trading import (
            account_meta,
            delete_paper_account,
            has_paper_account,
            save_paper_account,
        )
        from db.secret_box import encryption_available
    except Exception:
        return

    with st.expander("🔗 Alpaca paper trading", expanded=False):
        if not encryption_available():
            st.warning(
                "Secure credential storage isn't configured on the server "
                "(APP_ENCRYPTION_KEY). Paper trading is disabled until it is."
            )
            return

        connected = False
        try:
            connected = has_paper_account(user_id)
        except Exception:
            connected = False

        if connected:
            meta = None
            try:
                meta = account_meta(user_id)
            except Exception:
                meta = None
            when = ""
            if meta and meta.get("connected_at"):
                try:
                    when = f" · connected {meta['connected_at']:%b %d, %Y}"
                except Exception:
                    when = ""
            st.success(f"Paper account connected{when}.")
            st.caption(
                "Orders route to your Alpaca **paper** account (no real money). "
                "You confirm every order before it's sent."
            )
            if st.button("Disconnect paper account", key="pt_disconnect"):
                try:
                    delete_paper_account(user_id)
                    st.toast("Paper account disconnected")
                    st.rerun()
                except Exception:
                    st.warning("Could not disconnect.")
            return

        st.caption(
            "Connect your Alpaca **paper** account to place practice trades from "
            "a setup. Generate paper API keys at alpaca.markets → Paper Trading → "
            "API Keys. Keys are encrypted at rest; we never place an order without "
            "your explicit confirmation."
        )
        # Plain widgets (no st.form): a form here was rendering the caption but
        # not the fields on some deploys, and any error inside was swallowed by
        # the caller. Surface failures visibly instead of vanishing.
        try:
            api_key = st.text_input("Paper API key ID", key="pt_key")
            api_secret = st.text_input(
                "Paper API secret", type="password", key="pt_secret"
            )
            submitted = st.button("Validate & connect", key="pt_connect_btn")
        except Exception as e:
            st.error("Couldn't render the connect form.")
            st.caption(f"{type(e).__name__}: {e}")
            return
        if not submitted:
            return
        key = (api_key or "").strip()
        secret = (api_secret or "").strip()
        if not key or not secret:
            st.warning("Enter both the API key ID and secret.")
            return
        with st.spinner("Validating with Alpaca…"):
            try:
                from data.alpaca_trading import get_account

                acct = get_account(key, secret)
            except Exception:
                acct = None
        if not acct:
            st.error(
                "Could not validate those keys against the Alpaca paper endpoint. "
                "Double-check they're **paper** keys (not live) and try again."
            )
            return
        try:
            ok = save_paper_account(user_id, key, secret)
        except Exception:
            ok = False
        if ok:
            bp = acct.get("buying_power")
            bp_s = f" · buying power ${float(bp):,.0f}" if bp else ""
            st.success(f"Connected to Alpaca paper account{bp_s}.")
            st.rerun()
        else:
            st.error("Validated, but could not securely store the keys. Try again.")


def render_paper_trade_button(row: Mapping[str, Any], *, shares: int, plan: Mapping[str, Any],
                              key: str) -> None:
    """'Paper Trade This Setup' button + required confirmation. Never raises.

    Sends a whole-share market order only after the user explicitly confirms in
    the inline form. On success the fill imports into the journal with the
    scanner score and AI confidence attached.
    """
    if st is None:
        return
    try:
        user = _user()
        if not user:
            return
        ticker = str(row.get("Ticker") or "").upper() if hasattr(row, "get") else ""
        if not ticker:
            return

        if not _can_paper_trade():
            st.caption("💹 Paper trading is a **Premium** feature — upgrade to trade setups.")
            return

        try:
            from db.paper_trading import has_paper_account
        except Exception:
            return
        try:
            if not has_paper_account(user):
                st.caption(
                    "💹 Connect your Alpaca paper account (Account settings) to "
                    "paper trade this setup."
                )
                return
        except Exception:
            return

        qty = int(max(int(shares or 0), 0))
        entry = _num(row, "Last", "Close")
        confirm_key = f"pt_confirm_{key}_{ticker}"

        if st.button(f"💹 Paper Trade This Setup ({ticker})", key=f"pt_btn_{key}_{ticker}"):
            st.session_state[confirm_key] = True

        if not st.session_state.get(confirm_key):
            return

        # --- Explicit confirmation is REQUIRED before every order ---
        with st.form(f"pt_order_form_{key}_{ticker}", clear_on_submit=False):
            st.markdown(f"**Confirm paper order — {ticker}**")
            q = st.number_input(
                "Quantity (shares)", min_value=1, value=max(qty, 1), step=1,
                key=f"pt_qty_{key}_{ticker}",
            )
            entry_s = f"~${entry:,.2f}" if entry else "market"
            stop = plan.get("stop") if hasattr(plan, "get") else None
            targets = plan.get("targets") if hasattr(plan, "get") else None
            tgt = targets[0] if isinstance(targets, (list, tuple)) and targets else None
            bits = [f"Buy **{int(q)}** {ticker} at market (entry {entry_s})"]
            if stop:
                bits.append(f"plan stop ${float(stop):,.2f}")
            if tgt:
                bits.append(f"target ${float(tgt):,.2f}")
            st.caption(" · ".join(bits))
            st.caption(
                "This sends a **market order to your Alpaca paper account**. "
                "No real money. Educational only — not financial advice."
            )
            c1, c2 = st.columns(2)
            do_submit = c1.form_submit_button("✅ Submit Paper Trade", type="primary")
            do_cancel = c2.form_submit_button("Cancel")

        if do_cancel:
            st.session_state[confirm_key] = False
            st.rerun()
            return
        if not do_submit:
            return

        st.session_state[confirm_key] = False
        _submit_and_import(user, row, ticker, int(q), plan)
    except Exception:
        pass


def _submit_and_import(user: str, row: Mapping[str, Any], ticker: str, qty: int,
                       plan: Mapping[str, Any]) -> None:
    """Load keys, submit the market order, import the fill into the journal."""
    try:
        from db.paper_trading import get_paper_account

        acct = get_paper_account(user)
    except Exception:
        acct = None
    if not acct:
        st.error("Paper account not available. Reconnect it in Account settings.")
        return

    with st.spinner(f"Submitting paper order for {ticker}…"):
        try:
            from data.alpaca_trading import submit_market_order

            res = submit_market_order(
                acct["api_key"], acct["api_secret"], ticker, qty, side="buy"
            )
        except Exception:
            res = {"ok": False, "error": "Unexpected error submitting order."}

    if not res.get("ok"):
        st.error(f"Order rejected: {res.get('error') or 'unknown error'}")
        return

    # Import the fill into the journal with score + AI confidence attached.
    filled = _num(res, "filled_avg_price")
    entry = filled if filled else _num(row, "Last", "Close")
    breakout_score = _num(row, "BreakoutScore")
    ai_conf = _num(row, "AI Confidence", "AIConfidence", "ai_confidence")
    stop = plan.get("stop") if hasattr(plan, "get") else None
    targets = plan.get("targets") if hasattr(plan, "get") else None
    tgt = targets[0] if isinstance(targets, (list, tuple)) and targets else None
    try:
        from db.trades import log_trade

        log_trade(
            user, ticker, float(entry or 0.0), int(qty), source="paper",
            alpaca_order_id=res.get("order_id"),
            stop_price=stop, target_price=tgt,
            breakout_score=breakout_score, ai_confidence=ai_conf,
        )
    except Exception:
        pass

    status = res.get("status") or "submitted"
    st.success(f"Paper order {status} for {qty} {ticker} — imported to your journal.")
    st.rerun()
