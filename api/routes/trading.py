"""Trade journal and Alpaca paper trading."""
from __future__ import annotations

from typing import Any, Dict, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from api import models, ratelimit, user_data
from api.deps import _AUTH, _OWNED, TICKER, _user, current_account, require_feature, require_min_plan
from api.scans import json_safe


class JournalCreate(BaseModel):
    ticker: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")
    entry_price: float = Field(gt=0, le=1_000_000)
    shares: int = Field(ge=0, le=10_000_000)


class JournalClose(BaseModel):
    exit_price: float = Field(gt=0, le=1_000_000)


class PaperConnect(BaseModel):
    api_key: str = Field(min_length=8, max_length=128, description="Alpaca PAPER API key ID")
    api_secret: str = Field(min_length=8, max_length=256, description="Alpaca PAPER API secret (stored encrypted, never returned)")


class PaperOrderBody(BaseModel):
    ticker: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")
    qty: int = Field(ge=1, le=100_000, description="Whole shares")
    confirm: Literal[True] = Field(description="Must be true: the user confirmed this order (the web's confirmation step)")


def register(app: FastAPI) -> None:
    from api import trading

    _PRO = {**_AUTH, 403: {"description": "Pro feature"}}
    _PREM = {**_AUTH, 403: {"description": "Premium feature"}, 503: {"description": "Paper trading unavailable on the server"}}
    _PRO_MSG = "Trade plans and logging trades are part of Pro."

    @app.get("/v1/journal", response_model=models.Journal, responses=_AUTH, summary="Your trade journal")
    def journal(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Your logged trades (open first), open ones marked to live quotes, with closed-trade stats."""
        return json_safe(trading.journal(_user(account)))

    @app.post("/v1/journal", status_code=201, responses=_PRO, summary="Log a trade (Pro)")
    def journal_log(body: JournalCreate, account: Dict[str, Any] = Depends(current_account)) -> None:
        require_min_plan(account, "pro", _PRO_MSG)
        trading.log(_user(account), body.ticker.upper(), body.entry_price, body.shares)

    @app.post("/v1/journal/{trade_id}/close", status_code=204, responses={**_PRO, **_OWNED, 409: {"description": "Already closed"}},
              summary="Close a logged trade (Pro)")
    def journal_close(body: JournalClose, trade_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        require_min_plan(account, "pro", _PRO_MSG)
        if not trading.close(_user(account), trade_id, body.exit_price):
            raise user_data.NotFound("trade")

    @app.delete("/v1/journal/{trade_id}", status_code=204, responses={**_PRO, **_OWNED}, summary="Delete a logged trade (Pro)")
    def journal_delete(trade_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        require_min_plan(account, "pro", _PRO_MSG)
        if not trading.delete(_user(account), trade_id):
            raise user_data.NotFound("trade")

    @app.get("/v1/stocks/{ticker}/plan", response_model=models.TradePlan, responses={**_PRO, 404: {"description": "Not in the latest scan"}},
             summary="Trade plan for a scan result (Pro)")
    def stock_plan(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account),
                   account_size: float = Query(10_000.0, ge=100, le=1e9), risk_pct: float = Query(1.0, gt=0, le=10)
                   ) -> Dict[str, Any]:
        """The web's trade plan: stop at half the 20-day volatility (2-8%), targets at 1.5R and 3R,
        size from your risk budget. Educational only, not advice."""
        require_min_plan(account, "pro", _PRO_MSG)
        out = trading.plan(ticker.strip().upper(), account_size, risk_pct)
        if out is None:
            raise HTTPException(404, "That ticker isn't in the latest market scan.")
        return json_safe(out)

    @app.get("/v1/paper/account", response_model=models.PaperStatus, responses=_PREM, summary="Paper account status (Premium)")
    def paper_account(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_feature(account, "can_paper_trade")
        return json_safe(trading.paper_status(_user(account)))

    @app.post("/v1/paper/account", response_model=models.PaperStatus, responses={**_PREM, 400: {"description": "Keys rejected"}},
              summary="Connect your Alpaca paper account (Premium)")
    def paper_connect(body: PaperConnect, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Validates the keys against Alpaca's paper endpoint, then stores them encrypted. Paper keys only."""
        require_feature(account, "can_paper_trade")
        user = _user(account)
        ratelimit.check("paper_connect", user)
        return json_safe(trading.connect(user, body.api_key, body.api_secret))

    @app.delete("/v1/paper/account", status_code=204, responses=_PREM, summary="Disconnect your paper account (Premium)")
    def paper_disconnect(account: Dict[str, Any] = Depends(current_account)) -> None:
        require_feature(account, "can_paper_trade")
        trading.disconnect(_user(account))

    @app.get("/v1/paper/activity", response_model=models.PaperActivity, responses=_PREM, summary="Paper positions and orders (Premium)")
    def paper_activity(account: Dict[str, Any] = Depends(current_account),
                       limit: int = Query(25, ge=1, le=100)) -> Dict[str, Any]:
        require_feature(account, "can_paper_trade")
        return json_safe(trading.activity(_user(account), limit))

    @app.post("/v1/paper/orders", response_model=models.PaperOrder, status_code=201,
              responses={**_PREM, 400: {"description": "Order rejected"}, 422: {"description": "confirm must be true"},
                         429: {"description": "Too many orders this hour"}},
              summary="Paper trade a setup (Premium)")
    def paper_order(body: PaperOrderBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """A whole-share market BUY sent to your Alpaca PAPER account (no real money), imported into
        your journal. Show the user what will be sent and send confirm=true only after they confirm."""
        require_feature(account, "can_paper_trade")
        user = _user(account)
        ratelimit.check("paper_order", user)
        return json_safe(trading.order(user, body.ticker.upper(), body.qty))
