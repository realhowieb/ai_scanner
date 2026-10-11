"""Market Brief, Day Trader, earnings, tape and trade plans."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from api import models, user_data
from api.deps import _AUTH, _user, current_account, entitlements_for, require_feature
from api.scans import json_safe


def register(app: FastAPI) -> None:
    from api import market

    @app.get("/v1/earnings", response_model=List[models.EarningsItem], responses={**_AUTH, 403: {"description": "Pro feature"}},
             summary="Upcoming earnings (Pro)")
    def earnings(account: Dict[str, Any] = Depends(current_account),
                 days: int = Query(7, ge=0, le=market.MAX_EARNINGS_DAYS),
                 tickers: Optional[str] = Query(None, max_length=4000,
                                                description="Comma-separated tickers to keep (e.g. a scan's rows)")) -> List[Dict[str, Any]]:
        """The web's earnings calendar: earnings in the next `days` days, soonest first."""
        require_feature(account, "can_earnings")
        wanted = [t for t in (tickers or "").split(",") if t.strip()][:500]
        return market.earnings(days, wanted)

    _DT = {**_AUTH, 403: {"description": "Pro feature"}}

    @app.get("/v1/day-trader", response_model=models.DayTrader, responses=_DT, summary="Live Day Trader monitor (Pro)")
    def day_trader(account: Dict[str, Any] = Depends(current_account),
                   source: Literal["watchlist", "movers", "movers_sp500", "movers_nasdaq", "premarket", "postmarket",
                                   "scan_picks", "megacaps", "custom"] = "watchlist",
                   symbols: Optional[str] = Query(None, max_length=2000, description="source=custom: comma-separated"),
                   watchlist_id: Optional[int] = Query(None, ge=1, description="source=watchlist; default: your default list")
                   ) -> Dict[str, Any]:
        """The web's Day Trader table: live quotes, gap, VWAP, relative volume and the day-trade
        score for a symbol source. Quotes are shared for 30 s and movers screens for 2 min;
        poll every 30-60 s while the market is open."""
        require_feature(account, "can_day_trader")
        watch: List[str] = []
        if source == "watchlist":
            user = _user(account)
            wid = watchlist_id or next((w["id"] for w in user_data.list_watchlists(user) if w["is_default"]), None)
            if wid is not None:
                watch = [i["ticker"] for i in user_data.get_watchlist(user, int(wid))["items"]]   # 404 if not yours
        return json_safe(market.day_trader(source, (symbols or "").split(","), watch))

    @app.get("/v1/day-trader/sparklines", response_model=models.DayTraderSparklines, responses=_DT,
             summary="Intraday sparklines (Pro)")
    def day_trader_sparklines(account: Dict[str, Any] = Depends(current_account),
                              symbols: str = Query(..., max_length=2000,
                                                   description="Comma-separated; the first 40 are returned")
                              ) -> Dict[str, Any]:
        """The web's Day Trader row sparklines: the latest session's 1-minute closes."""
        require_feature(account, "can_day_trader")
        return json_safe(market.day_trader_sparklines(symbols.split(",")))

    @app.get("/v1/day-trader/stair-steppers", response_model=models.StairSteppers, responses=_DT,
             summary="Smooth 1-minute trends (Pro)")
    def stair_steppers(account: Dict[str, Any] = Depends(current_account),
                       symbols: str = Query(..., max_length=2000, description="Comma-separated; the first 40 are checked"),
                       window: int = Query(45, description="Bars fitted: 10, 15, 20, 30, 45 or 60"),
                       direction: Literal["up", "down", "either"] = "up",
                       r2_min: float = Query(0.8, ge=0.5, le=0.99),
                       max_pullback: float = Query(1.0, ge=0.1, le=5.0),
                       min_trend: float = Query(0.5, ge=0.0, le=20.0)) -> Dict[str, Any]:
        """The web's Stair-steppers check: symbols moving in a tight, straight line on the
        1-minute chart. Descriptive only; not a prediction."""
        require_feature(account, "can_day_trader")
        from analytics.stair_step import WINDOW_OPTIONS

        if window not in WINDOW_OPTIONS:
            raise HTTPException(422, f"window must be one of {list(WINDOW_OPTIONS)}.")
        return json_safe(market.stair_steppers(symbols.split(","), window=window, direction=direction, r2_min=r2_min,
                                               max_pullback_pct=max_pullback, min_trend_pct_per_hour=min_trend))

    @app.get("/v1/brief", response_model=models.Brief, responses=_AUTH, summary="Market Brief")
    def brief(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """The web's Market Brief (same builder as the morning email): market backdrop, top
        opportunities with movement, gappers, movers, setups and catalysts. Cached 5 minutes.
        PreBreakout picks and model fields are Premium. AI narrative: /v1/ai (Premium);
        historical scorecard: /v1/track-record (Pro); your alerts: /v1/alerts/events."""
        return json_safe(market.brief(entitlements_for(account)["entitlements"]))
