"""Getting started checklist, unread alert badge, in-app feedback and shared stock links."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api import models, ratelimit
from api.deps import _AUTH, TICKER, _user, current_account

log = logging.getLogger("hsf_api")


class FeedbackBody(BaseModel):
    kind: Literal["problem", "idea", "praise", "other"] = "other"
    message: str = Field(min_length=3, max_length=2000)
    path: Optional[str] = Field(default=None, max_length=200, description="The page it was sent from, no query string")
    build: Optional[str] = Field(default=None, max_length=64, pattern=r"^[\w.\-]*$", description="The web build's commit")


def _db_error(e: Exception) -> HTTPException:
    log.warning(json.dumps({"event": "web_state_write_failed", "error": type(e).__name__}))
    return HTTPException(503, "Couldn't save that right now. Try again in a moment.")


def register(app: FastAPI) -> None:
    from api import web_state

    @app.get("/v1/me/getting-started", response_model=models.GettingStarted, responses=_AUTH,
             summary="Getting started checklist")
    def getting_started(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Four first steps for a new account (save stocks, set an alert, choose where alerts go,
        read how the score works), each ticked from what the account has already done."""
        return web_state.getting_started(account)

    @app.post("/v1/me/getting-started/dismiss", response_model=models.GettingStarted, responses=_AUTH,
              summary="Close the Getting started card")
    def getting_started_dismiss(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        try:
            web_state.mark(_user(account), "onboarding_dismissed_at")
        except Exception as e:
            raise _db_error(e) from None
        return web_state.getting_started(account)

    @app.post("/v1/me/getting-started/guide", response_model=models.GettingStarted, responses=_AUTH,
              summary="Tick the 'how the HSF Score works' step")
    def getting_started_guide(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        try:
            web_state.mark(_user(account), "guide_seen_at")
        except Exception as e:
            raise _db_error(e) from None
        return web_state.getting_started(account)

    @app.get("/v1/me/alerts/unread", response_model=models.UnreadAlerts, responses=_AUTH,
             summary="Alerts fired since you last opened Alerts")
    def alerts_unread(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """The count behind the badge on the Alerts tab. With no visit on record, the last 7 days."""
        return web_state.unread_alerts(_user(account))

    @app.post("/v1/me/alerts/seen", response_model=models.UnreadAlerts, responses=_AUTH,
              summary="Mark fired alerts as seen")
    def alerts_seen(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        try:
            stamp = web_state.mark(_user(account), "alerts_seen_at")
        except Exception as e:
            raise _db_error(e) from None
        return {"count": 0, "seen_at": stamp}

    @app.post("/v1/me/feedback", response_model=models.Message, status_code=201,
              responses={**_AUTH, 429: {"description": "Too many messages"}}, summary="Send feedback to the HSF team")
    def feedback(body: FeedbackBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stored, then sent to the admins' Telegram/Discord, or emailed when none is linked."""
        ratelimit.check("feedback", _user(account))
        try:
            web_state.send_feedback(account, body.kind, body.message, body.path, body.build)
        except Exception as e:
            raise _db_error(e) from None
        msg = "Thanks, your message was sent."
        if account.get("founder_tier"):
            from api.founding import reply_target

            target = reply_target()
            msg = (f"Thanks, founder. Your message goes to the top of our list. We aim to reply within {target}."
                   if target else "Thanks, founder. Your message goes to the top of our list.")
        return {"ok": True, "message": msg}

    @app.get("/v1/admin/feedback", response_model=List[models.FeedbackItem], tags=["admin (admins only)"],
             responses={**_AUTH, 403: {"description": "Admins only"}}, summary="Recent feedback from the app",
             openapi_extra={"x-internal": True})
    def admin_feedback(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        from db import web_state as store

        if not account.get("is_admin"):
            raise HTTPException(403, "Admins only.")
        return store.recent_feedback(50)

    @app.post("/v1/admin/feedback/{feedback_id}/answered", response_model=models.Message, tags=["admin (admins only)"],
              responses={**_AUTH, 403: {"description": "Admins only"}, 404: {"description": "No such message"}},
              summary="Mark a feedback message answered or not", openapi_extra={"x-internal": True})
    def answer_feedback(feedback_id: int, body: models.FeedbackAnswered,
                        account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from db import web_state as store

        if not account.get("is_admin"):
            raise HTTPException(403, "Admins only.")
        if not store.mark_answered(feedback_id, _user(account), body.answered):
            raise HTTPException(404, "No such message.")
        return {"ok": True, "message": "Marked answered." if body.answered else "Marked not answered."}

    @app.get("/v1/admin/feedback/founders", response_model=models.FounderFeedbackStatus, tags=["admin (admins only)"],
             responses={**_AUTH, 403: {"description": "Admins only"}},
             summary="Founder feedback still waiting for an answer", openapi_extra={"x-internal": True})
    def founder_feedback(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api.founding import reply_target
        from db import web_state as store

        if not account.get("is_admin"):
            raise HTTPException(403, "Admins only.")
        return {**store.unanswered_founder_feedback(), "reply_target": reply_target()}

    @app.get("/v1/public/top-setups", response_model=models.PublicTopSetups,
             responses={429: {"description": "Too many requests"}},
             summary="The latest scan's top setups, for the landing page (signed out)")
    def public_top_setups(_l: None = Depends(ratelimit.limit("share"))) -> Dict[str, Any]:
        """The top three ranked setups of the latest market scan (ticker, HSF Score, setup type,
        day change) and how many more a free account sees. Everything else needs an account."""
        from api import share

        try:
            return share.public_top_setups()
        except Exception as e:
            log.warning(json.dumps({"event": "public_top_setups_failed", "error": type(e).__name__}))
            raise HTTPException(503, "The live preview isn't available right now.") from None

    @app.get("/v1/public/stocks/{ticker}", response_model=models.PublicStock,
             responses={404: {"description": "Not a listed US stock"}, 429: {"description": "Too many requests"}},
             summary="A stock's public snapshot for shared links (signed out)")
    def public_stock(ticker: str = TICKER, _l: None = Depends(ratelimit.limit("share"))) -> Dict[str, Any]:
        """Name, price and day change; HSF Score, setup and the first reason only when the stock
        is in the latest scan's ranked setups. The rest of the stock page needs an account."""
        from api import share

        try:
            return share.public_stock(ticker)
        except share.UnknownTicker:
            raise HTTPException(404, "We don't have that stock.") from None
