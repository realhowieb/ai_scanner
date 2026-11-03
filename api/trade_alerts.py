"""Automatic stop and target alerts for logged trades (Premium automation, 2026-11-03).

When a Premium member logs a trade (POST /v1/journal) or a paper order fills into the
journal, HSF adds two price alerts for that stock: one below at the Trade plan's stop and
one above at its first target (1.5R), worked out from the price the trade was taken at.
The stop distance is the Trade plan's (half the 20-day volatility, 2-8%) when the stock
is in the latest scan, otherwise DEFAULT_STOP_PCT.

The alerts are ordinary price alerts (they show on the Alerts page, count toward the plan's
alert limit and can be deleted there). Which ones were added is kept in db.premium_auto
(kind "trade_alerts", ref the ticker), so when the last open trade in that stock is closed
or deleted, those alerts are removed again. Nothing here raises into the journal call.

Env: HSF_AUTO_TRADE_ALERTS=0 stops it.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

log = logging.getLogger("hsf.trade_alerts")

KIND = "trade_alerts"
FLAG = "can_auto_research"
DEFAULT_STOP_PCT = 5.0
TARGET_R = 1.5


def enabled() -> bool:
    return os.environ.get("HSF_AUTO_TRADE_ALERTS", "1").strip() != "0"


def levels(entry: float, stop_pct: Optional[float]) -> Dict[str, float]:
    """Stop and first-target prices for a trade taken at `entry`. Pure."""
    pct = float(stop_pct) if stop_pct else DEFAULT_STOP_PCT
    pct = min(max(pct, 2.0), 8.0)
    stop = entry * (1 - pct / 100.0)
    target = entry + TARGET_R * (entry - stop)
    return {"stop": round(stop, 2), "target": round(target, 2), "stop_pct": round(pct, 1)}


def _stop_pct(ticker: str) -> Optional[float]:
    try:
        from api import trading

        p = trading.plan(ticker, 10_000.0, 1.0)
        return float(p["stop_pct"]) if p and p.get("stop_pct") else None
    except Exception:
        return None


def _entitlements(username: str) -> Optional[Dict[str, Any]]:
    from services.scheduled_scans import _account_entitlements

    return _account_entitlements(username)


def add_for_trade(username: str, ticker: str, entry_price: float) -> Optional[Dict[str, Any]]:
    """Add the stop and target alerts for a new trade. Returns the saved levels, or None when
    not Premium, switched off, already set for this stock, or the alert limit is reached."""
    if not enabled():
        return None
    try:
        from api import ai, user_data

        t = str(ticker or "").strip().upper()
        entry = float(entry_price)
        if not t or entry <= 0:
            return None
        ent = _entitlements(username)
        if not ent or not ent["entitlements"].get(FLAG):
            return None
        if ai.stored(username, KIND, t):
            return None   # one pair per stock; a second lot keeps the first trade's levels
        lv = levels(entry, _stop_pct(t))
        ids: List[int] = []
        for direction, price in (("below", lv["stop"]), ("above", lv["target"])):
            try:
                a = user_data.create_alert(username, int(ent["alert_limit"]), "price",
                                           ticker=t, threshold=price, direction=direction)
                ids.append(int(a["id"]))
            except (user_data.LimitReached, user_data.Conflict):
                continue
        if not ids:
            return None
        text = f"Price alerts at ${lv['stop']:.2f} (stop) and ${lv['target']:.2f} (target)"
        ai.store(username, KIND, t, text, alert_ids=ids, entry=round(entry, 2), **lv)
        return {**lv, "alert_ids": ids}
    except Exception as e:
        log.info(json.dumps({"event": "trade_alerts_add_failed", "error": type(e).__name__}))
        return None


def remove_if_flat(username: str, ticker: str) -> int:
    """After a trade closes or is deleted: when no open trade in the stock is left, delete
    the alerts this module added for it. Returns how many were removed."""
    try:
        from api import ai, user_data
        from db import premium_auto
        from db.trades import list_trades

        t = str(ticker or "").strip().upper()
        saved = ai.stored(username, KIND, t)
        if not saved:
            return 0
        still_open = any(not r.get("closed_at") and str(r.get("ticker") or "").upper() == t
                         for r in list_trades(username, limit=500))
        if still_open:
            return 0
        removed = 0
        for alert_id in saved.get("alert_ids") or []:
            try:
                user_data.delete_alert(username, int(alert_id))
                removed += 1
            except Exception:
                continue   # the member already deleted it
        premium_auto.delete(username, KIND, t)
        return removed
    except Exception as e:
        log.info(json.dumps({"event": "trade_alerts_remove_failed", "error": type(e).__name__}))
        return 0


def for_journal(username: str) -> Dict[str, Dict[str, Any]]:
    """{TICKER: {"stop", "target"}} for the journal page. Never raises."""
    try:
        from db import premium_auto

        out = {}
        for row in premium_auto.list_kind(username, KIND):
            try:
                body = json.loads(row["body"])
                out[str(row["ref"]).upper()] = {"stop": float(body["stop"]), "target": float(body["target"])}
            except (TypeError, ValueError, KeyError):
                continue
        return out
    except Exception:
        return {}
