"""Scans and stocks, watchlists, price alerts and alert rules."""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from api import models, user_data
from api.deps import (
    _AUTH,
    _OWNED,
    TICKER,
    _aware,
    _capabilities,
    _rule_errors,
    _user,
    current_account,
    entitlements_for,
)
from api.scans import json_safe

log = logging.getLogger("hsf_api")



class WatchlistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    make_default: bool = False


class WatchlistUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    make_default: bool = Field(default=False, description="true makes this the default watchlist")


class TickersBody(BaseModel):
    tickers: List[str] = Field(min_length=1, max_length=user_data.MAX_TICKERS_PER_REQUEST)


class NoteBody(BaseModel):
    note: Optional[str] = Field(default=None, max_length=user_data.MAX_NOTE_LEN)


class AlertCreate(BaseModel):
    type: str = Field(description="breakout, watchlist, price, move, rvol, ema_cross or ewo_cross")
    ticker: Optional[str] = Field(default=None, max_length=12)
    threshold: Optional[float] = None
    direction: Optional[str] = Field(default=None, max_length=12)
    watchlist_only: bool = False


class AlertRuleCreate(BaseModel):
    rule_type: str = Field(description="See GET /v1/alerts/rules/types, e.g. HSF_SCORE_CROSS_ABOVE")
    ticker: Optional[str] = Field(default=None, max_length=12, description="One ticker, or set watchlist_id")
    watchlist_id: Optional[int] = Field(default=None, ge=1, description="Every symbol on this watchlist")
    threshold: Optional[float] = None
    value: Optional[str] = Field(default=None, max_length=40, description="SETUP_APPEARED: only this setup (optional)")
    delivery_channels: Optional[List[Literal["in_app", "email"]]] = Field(
        default=None, max_length=2, description="Default in_app; email is Pro+")
    cooldown_seconds: Optional[int] = Field(default=None, ge=0, le=7 * 86400,
                                            description="Default 1 day for level rules, 1 hour for transitions")
    enabled: bool = True


class AlertRuleUpdate(BaseModel):
    threshold: Optional[float] = None
    value: Optional[str] = Field(default=None, max_length=40)
    enabled: Optional[bool] = None
    delivery_channels: Optional[List[Literal["in_app", "email"]]] = Field(default=None, min_length=1, max_length=2)
    cooldown_seconds: Optional[int] = Field(default=None, ge=0, le=7 * 86400)


class AlertUpdate(BaseModel):
    enabled: bool


def register(app: FastAPI) -> None:
    # ---- step 5: scans and stock detail ----
    @app.get("/v1/today/me", response_model=models.TodayPersonal, responses=_AUTH, summary="Today: your sections")
    def today_personal(account: Dict[str, Any] = Depends(current_account),
                       seen: Optional[int] = Query(None, ge=1, description="Market run this browser last saw (from `marker`)"),
                       baseline: Optional[int] = Query(None, ge=1, description="Its baseline run (from `marker`)")) -> Dict[str, Any]:
        """New since your last visit and your watchlist against the latest market scan."""
        from api.today import build_personal

        return build_personal(_user(account), seen, baseline)

    @app.get("/v1/market/tape", response_model=models.Tape, responses=_AUTH, summary="Price strip")
    def market_tape(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA and TSLA: last price and change vs the previous close."""
        from api.today import tape_quotes

        return {"quotes": tape_quotes()}

    @app.get("/v1/scans/latest", response_model=models.LatestScan, responses=_AUTH)
    def scans_latest(account: Dict[str, Any] = Depends(current_account),
                     limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=10_000),
                     min_score: int = Query(0, ge=0, le=100),
                     signal: Optional[str] = Query(None, pattern="^(golden_cross|breakout|prebreakout|gapper|gainer)$",
                                                   description="Only setups with this signal"),
                     sort: Literal["score", "chg_pct", "gap_pct", "rvol", "prob"] = Query(
                         "score", description="Order of the rows the plan sees (descending); the plan's rows are "
                                              "always its top HSF-ranked setups")) -> Dict[str, Any]:
        """The latest market scan's HSF setups, ranked as in the Scanner, up to the plan's row cap."""
        from api.scans import latest_scan

        ent = entitlements_for(account)
        if signal == "prebreakout" and not ent["entitlements"].get("can_early_breakout"):
            raise HTTPException(403, "PreBreakout is a Premium feature.")
        return latest_scan(ent["entitlements"], ent["tier"], limit=limit, offset=offset,
                           min_score=min_score, signal=signal, sort=sort)

    @app.get("/v1/stocks/{ticker}", response_model=models.StockDetail, responses=_AUTH)
    def stock(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stock Intelligence for one ticker: score, signals, reasons and risks, lifecycle,
        daily bars, and your watchlists and alerts on it."""
        from api.scans import daily_bars, stock_detail

        t = ticker.strip().upper()
        user = _user(account)
        out = stock_detail(t, entitlements_for(account)["entitlements"])
        try:
            bars = daily_bars(t)
        except Exception:  # chart data is optional; the page still renders
            bars = {"bars": [], "as_of": None}
        out.update({"bars": bars["bars"], "bars_as_of": bars["as_of"],
                    "watchlists": user_data.watchlists_with(user, t),
                    "alerts": json_safe(user_data.alerts_for(user, t))})
        return out

    # ---- step 6: watchlists ----
    @app.get("/v1/watchlists", response_model=List[models.Watchlist], responses=_AUTH)
    def watchlists(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        return user_data.list_watchlists(_user(account))

    @app.post("/v1/watchlists", response_model=models.WatchlistDetail, status_code=201,
              responses={**_AUTH, 403: {"description": "Watchlist limit reached"},
                         409: {"description": "A watchlist with that name exists"}})
    def watchlist_create(body: WatchlistCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.create_watchlist(_user(account), body.name, body.make_default))

    @app.get("/v1/watchlists/{watchlist_id}", response_model=models.WatchlistDetail, responses=_OWNED)
    def watchlist_get(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """The list with each ticker's row in the latest market scan (`items[].latest`)."""
        from api.scans import watchlist_scan_state

        out = user_data.get_watchlist(_user(account), watchlist_id)
        state = watchlist_scan_state([i["ticker"] for i in out["items"]], entitlements_for(account)["entitlements"])
        out = {**out, "scan_at": state["scan_at"], "scan_total": state.get("total"), "stale": state["stale"],
               "items": [{**i, "latest": state["rows"].get(str(i["ticker"]).upper())} for i in out["items"]]}
        return json_safe(out)

    @app.patch("/v1/watchlists/{watchlist_id}", response_model=models.WatchlistDetail,
               responses={**_OWNED, 409: {"description": "A watchlist with that name exists"}})
    def watchlist_update(watchlist_id: int, body: WatchlistUpdate,
                         account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.update_watchlist(_user(account), watchlist_id, name=body.name,
                                                    make_default=body.make_default))

    @app.delete("/v1/watchlists/{watchlist_id}", status_code=204, responses=_OWNED)
    def watchlist_delete(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> None:
        """Alert rules on this watchlist are switched off (kept, so you can see why)."""
        from api import alert_rules
        from db import alert_rules as rule_store

        user_data.delete_watchlist(_user(account), watchlist_id)
        try:  # best effort: the evaluator skips rules whose watchlist is gone either way
            alert_rules._db(rule_store.disable_rules_for_watchlist, _user(account), watchlist_id)
        except Exception as e:
            log.warning(json.dumps({"event": "watchlist_rules_disable_failed", "error": type(e).__name__}))

    @app.get("/v1/watchlists/{watchlist_id}/intelligence", response_model=models.WatchlistIntelligence,
             responses=_OWNED, summary="HSF intelligence for every symbol on a watchlist")
    def watchlist_intelligence(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Score, rank and their change since the previous scan, setup, signals, PreBreakout
        (Premium), price, RVOL, EMA cross, freshness and active alert count for each symbol,
        from the latest saved market scan (no live quote calls). Fields with no canonical
        source are null and listed in `unavailable_fields`."""
        from api import watchlist_intel

        user = _user(account)
        wl = user_data.get_watchlist(user, watchlist_id)
        return json_safe(watchlist_intel.intelligence(user, wl, entitlements_for(account)["entitlements"]))

    @app.get("/v1/watchlists/{watchlist_id}/changes", response_model=models.WatchlistChanges,
             responses=_OWNED, summary="What changed on a watchlist since the previous scan")
    def watchlist_changes(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """The canonical HSF changes (new, dropped, rising/falling, status, fading, signals
        incl. PreBreakout) between the two latest market scans for this list's symbols, with
        rank moves, plus your rule alerts on them since the previous scan."""
        from api import watchlist_intel

        user = _user(account)
        wl = user_data.get_watchlist(user, watchlist_id)
        return json_safe(watchlist_intel.changes(user, wl, entitlements_for(account)["entitlements"]))

    @app.post("/v1/watchlists/{watchlist_id}/tickers", response_model=models.TickersResult, responses=_OWNED)
    def watchlist_add(watchlist_id: int, body: TickersBody,
                      account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return user_data.add_tickers(_user(account), watchlist_id, body.tickers)

    @app.post("/v1/watchlists/{watchlist_id}/symbols", response_model=models.TickersResult, responses=_OWNED,
              summary="Add symbols (same as /tickers)")
    def watchlist_add_symbols(watchlist_id: int, body: TickersBody,
                              account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return user_data.add_tickers(_user(account), watchlist_id, body.tickers)

    @app.delete("/v1/watchlists/{watchlist_id}/tickers/{ticker}", status_code=204, responses=_OWNED)
    def watchlist_remove(watchlist_id: int, ticker: str = TICKER,
                         account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.remove_ticker(_user(account), watchlist_id, ticker.upper())

    @app.delete("/v1/watchlists/{watchlist_id}/symbols/{ticker}", status_code=204, responses=_OWNED,
                summary="Remove a symbol (same as /tickers/{ticker})")
    def watchlist_remove_symbol(watchlist_id: int, ticker: str = TICKER,
                                account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.remove_ticker(_user(account), watchlist_id, ticker.upper())

    @app.patch("/v1/watchlists/{watchlist_id}/tickers/{ticker}", status_code=204, responses=_OWNED)
    def watchlist_note(watchlist_id: int, body: NoteBody, ticker: str = TICKER,
                       account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.set_note(_user(account), watchlist_id, ticker.upper(), body.note)

    # ---- step 6: alerts ----
    @app.get("/v1/alerts", response_model=models.Alerts, responses=_AUTH)
    def alerts(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        ent = entitlements_for(account)
        items = user_data.list_alerts(_user(account))
        return json_safe({"limit": ent["alert_limit"], "used": len(items),
                          "email_enabled": bool(ent["entitlements"].get("can_email_alerts")), "alerts": items})

    @app.post("/v1/alerts", response_model=models.Alert, status_code=201,
              responses={**_AUTH, 403: {"description": "Plan alert limit reached"},
                         409: {"description": "You already have this alert"},
                         422: {"description": "Invalid type, ticker, threshold or direction"}})
    def alert_create(body: AlertCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Same types and rules as the web app. Free 1 alert, Pro 5, Premium 25."""
        ent = entitlements_for(account)
        try:
            created = user_data.create_alert(_user(account), ent["alert_limit"], body.type, ticker=body.ticker,
                                             threshold=body.threshold, direction=body.direction,
                                             watchlist_only=body.watchlist_only)
        except user_data.Conflict:
            raise
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return json_safe(created)

    @app.patch("/v1/alerts/{alert_id}", response_model=models.Alert, responses=_OWNED)
    def alert_update(alert_id: int, body: AlertUpdate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.set_alert_enabled(_user(account), alert_id, body.enabled))

    @app.delete("/v1/alerts/{alert_id}", status_code=204, responses=_OWNED)
    def alert_delete(alert_id: int, account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.delete_alert(_user(account), alert_id)

    @app.get("/v1/alerts/types", response_model=List[models.AlertType], responses=_AUTH)
    def alert_types(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        """The alert types and their input rules (what POST /v1/alerts validates), for building forms."""
        return user_data.alert_types()

    @app.get("/v1/alerts/events", response_model=List[models.AlertEvent],
             responses={**_AUTH, 422: {"description": "Invalid cursor or filter"}})
    def alert_events(account: Dict[str, Any] = Depends(current_account),
                     limit: int = Query(20, ge=1, le=100),
                     ticker: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$"),
                     rule_id: Optional[int] = Query(None, ge=1, description="Only this rule's events"),
                     watchlist_id: Optional[int] = Query(None, ge=1, description="Only rule events from this watchlist"),
                     triggered_after: Optional[dt.datetime] = Query(None),
                     triggered_before: Optional[dt.datetime] = Query(None),
                     source: Optional[Literal["alert", "rule"]] = Query(None, description="alert: ticker alerts; rule: alert rules"),
                     cursor: Optional[str] = Query(None, max_length=200, description="The last event's cursor, for the next page")
                     ) -> List[Dict[str, Any]]:
        """Your fired alerts (ticker alerts and alert rules), newest first. Page with `cursor`."""
        from api import alert_rules

        try:
            return json_safe(alert_rules.list_events(
                _user(account), limit=limit, ticker=ticker.upper() if ticker else None, rule_id=rule_id,
                watchlist_id=watchlist_id, triggered_after=_aware(triggered_after),
                triggered_before=_aware(triggered_before), cursor=cursor, source=source))
        except alert_rules.RuleError as e:
            raise HTTPException(422, str(e)) from e

    # ---- alert rules (server-evaluated conditions on HSF intelligence) ----
    @app.get("/v1/alerts/rules", response_model=models.AlertRules, responses=_AUTH, summary="Your alert rules")
    def alert_rules_list(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Your alert rules, the plan's active-alert limit (shared with ticker alerts) and
        what the plan allows."""
        from api import alert_rules

        ent = entitlements_for(account)
        user = _user(account)
        return json_safe({"limit": ent["alert_limit"], "used": alert_rules.count_active(user),
                          "capabilities": _capabilities(ent), "rules": alert_rules.list_rules(user)})

    @app.get("/v1/alerts/rules/types", response_model=List[models.AlertRuleType], responses=_AUTH,
             summary="Alert rule types")
    def alert_rule_types(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        """The rule types, their thresholds and whether your plan includes each."""
        from api import alert_rules

        return alert_rules.rule_types(entitlements_for(account)["entitlements"])

    @app.post("/v1/alerts/rules", response_model=models.AlertRule, status_code=201, summary="Create an alert rule",
              responses={**_AUTH, 403: {"description": "Plan limit, rule type or channel not on your plan"},
                         404: {"description": "Watchlist not found (or not yours)"},
                         409: {"description": "You already have this rule"},
                         422: {"description": "Invalid rule"}})
    def alert_rule_create(body: AlertRuleCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Set exactly one of `ticker` or `watchlist_id`. Rules count toward the plan's alert
        limit together with ticker alerts."""
        from api import alert_rules

        with _rule_errors():
            return json_safe(alert_rules.create_rule(_user(account), entitlements_for(account), body.model_dump()))

    @app.get("/v1/alerts/rules/{rule_id}", response_model=models.AlertRule, responses=_OWNED, summary="One alert rule")
    def alert_rule_get(rule_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api import alert_rules

        return json_safe(alert_rules.get_rule(_user(account), rule_id))

    @app.patch("/v1/alerts/rules/{rule_id}", response_model=models.AlertRule, summary="Change an alert rule",
               responses={**_OWNED, 403: {"description": "Plan limit or channel not on your plan"},
                          422: {"description": "Invalid change"}})
    def alert_rule_update(body: AlertRuleUpdate, rule_id: int = Path(ge=1),
                          account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Threshold, value, enabled, channels, cooldown. A new threshold or value starts the
        rule from a fresh baseline."""
        from api import alert_rules

        with _rule_errors():
            return json_safe(alert_rules.update_rule(_user(account), entitlements_for(account), rule_id,
                                                     body.model_dump(exclude_unset=True)))

    @app.delete("/v1/alerts/rules/{rule_id}", status_code=204, responses=_OWNED, summary="Delete an alert rule")
    def alert_rule_delete(rule_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        """Its past events stay in /v1/alerts/events."""
        from api import alert_rules

        alert_rules.delete_rule(_user(account), rule_id)

    @app.get("/v1/me/capabilities", response_model=models.Capabilities, responses=_AUTH,
             summary="Your plan's watchlist and alert limits")
    def my_capabilities(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return _capabilities(entitlements_for(account))
