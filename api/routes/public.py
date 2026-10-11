"""Signed-out endpoints: plans, funnel events, web crash reports, email unsubscribe."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from api import models, ratelimit
from api.deps import _track

log = logging.getLogger("hsf_api")



FUNNEL_EVENTS = ("landing_visit", "primary_cta_click", "signup_started")


class FunnelEvent(BaseModel):
    event: Literal["landing_visit", "primary_cta_click", "signup_started"]
    attribution: Dict[str, str] = Field(default_factory=dict, max_length=12,
                                        description="utm_* tags and referrer from the visitor's first page")
    surface: Optional[str] = Field(default=None, max_length=40, description="Which button or page")


class ClientErrorReport(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    kind: Literal["boundary", "global", "window", "promise"] = "boundary"
    path: Optional[str] = Field(default=None, max_length=200, description="Page path, no query string")
    digest: Optional[str] = Field(default=None, max_length=64, description="Next.js server error digest")
    stack: Optional[str] = Field(default=None, max_length=2000)
    request_id: Optional[str] = Field(default=None, max_length=64)


class UnsubscribeBody(BaseModel):
    token: str = Field(min_length=10, max_length=64)
    kind: Literal["digest", "evening", "alerts", "all"]


def register(app: FastAPI) -> None:
    """Signed-out endpoints for the web app: plans and pricing, funnel events and the
    emailed unsubscribe link."""

    @app.get("/v1/plans", response_model=models.Plans, summary="Plans and pricing (public)")
    def plans() -> Dict[str, Any]:
        """The plan comparison the landing and pricing pages show, from the same source as
        the Billing page (ui.pricing), so copy can't drift from what each plan gets."""
        from ui import pricing as p

        highlights = p.plan_highlights()
        tiers = [{"id": t, "name": p.TIER_NAMES[t], "price": p.PRICES[t], "yearly_price": p.YEARLY_PRICES.get(t),
                  "tagline": p.TAGLINES[t], "alert_limit": int(p.ALERT_LIMIT_BY_TIER.get(t, 1)),
                  "highlights": highlights.get(t, [])} for t in p.TIERS]
        rows = [{"label": label, **{t: (int(p.ALERT_LIMIT_BY_TIER.get(t, 1)) if flag == p.ALERTS else p.included(flag, t))
                                     for t in p.TIERS}} for label, flag in p.ROWS]
        return {"tiers": tiers, "rows": rows}

    @app.post("/v1/events", status_code=202, responses={429: {"description": "Too many events from this address"}},
              summary="Record a signed-out funnel event")
    def funnel_event(body: FunnelEvent, _l: None = Depends(ratelimit.limit("events"))) -> None:
        """Landing visit, call-to-action click or sign-up started, with the visitor's utm tags.
        Stores no email, name or IP. Best effort: always accepted."""
        _track(body.attribution, body.event, metadata={"surface": body.surface or "web", "app": "web"})

    @app.post("/v1/client-errors", status_code=202, responses={429: {"description": "Too many reports from this address"}},
              summary="Report a crash in the web app")
    def client_error(body: ClientErrorReport, _l: None = Depends(ratelimit.limit("client_errors"))) -> None:
        """A page the web app couldn't render. Logged as one JSON line and sent to Sentry
        when SENTRY_DSN is set. Holds no account, token or query string; always accepted."""
        report = {"event": "web_client_error", "kind": body.kind, "message": body.message,
                  "path": (body.path or "").split("?")[0][:200] or None, "digest": body.digest,
                  "request_id": body.request_id}
        log.warning(json.dumps(report))
        from api.monitoring import capture_client_error

        capture_client_error(report, body.stack)

    def _unsub_user(token: str) -> str:
        from db.email_prefs import user_for_token

        user = user_for_token(token)
        if not user:
            raise HTTPException(400, "This unsubscribe link isn't valid. Sign in and open Account to change your emails.")
        return user

    def _unsub_state(user: str) -> Dict[str, Any]:
        from db.email_prefs import get_prefs
        from ui.log_privacy import mask_email

        return {"email": mask_email(user), "prefs": get_prefs(user)}

    _UNSUB = {400: {"description": "Invalid link"}, 429: {"description": "Too many attempts"}}

    @app.get("/v1/email-preferences/unsubscribe", response_model=models.UnsubscribeState, responses=_UNSUB,
             summary="Email settings behind an unsubscribe link")
    def unsubscribe_state(t: str = Query(min_length=10, max_length=64), _l: None = Depends(ratelimit.limit("unsubscribe"))
                          ) -> Dict[str, Any]:
        """Which emails the link's account gets. Changes nothing (email scanners open links)."""
        return _unsub_state(_unsub_user(t))

    @app.post("/v1/email-preferences/unsubscribe", response_model=models.UnsubscribeState,
              responses={**_UNSUB, 503: {"description": "Couldn't save"}}, summary="Unsubscribe with an emailed link")
    def unsubscribe(body: UnsubscribeBody, _l: None = Depends(ratelimit.limit("unsubscribe"))) -> Dict[str, Any]:
        """Turns off one kind of email, or all of them, for the link's account. Account emails
        (verification, password reset) still go out."""
        from db.email_prefs import KINDS, set_prefs

        user = _unsub_user(body.token)
        kinds = KINDS if body.kind == "all" else (body.kind,)
        if not set_prefs(user, **{k: False for k in kinds}):
            raise HTTPException(503, "Couldn't save that right now. Please try again in a minute.")
        return _unsub_state(user)
