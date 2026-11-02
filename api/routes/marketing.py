"""Marketing endpoints: the weekly-setups email list, the public weekly recap and the
admin Marketing report (with the confirmed-list CSV export)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path, Query
from fastapi.responses import Response

from api import models, ratelimit
from api.deps import _ADMIN
from api.routes.admin import _admin, _analytics

_TAG = "admin (admins only)"
_EXTRA = {"x-internal": True}
CHECK_INBOX = "Check your inbox: we sent a link to confirm your email."


def register(app: FastAPI) -> None:
    @app.get("/v1/public/list", response_model=models.ListConfig, summary="Whether the email list form is open")
    def list_config() -> Dict[str, Any]:
        from services.marketing_list import enabled

        return {"enabled": enabled()}

    @app.post("/v1/public/subscribe", response_model=models.ListSubscribeResult,
              responses={422: {"description": "Not an email address"}, 429: {"description": "Too many sign-ups"},
                         503: {"description": "List closed or unavailable"}},
              summary="Join the weekly setups email list (double opt-in)")
    def subscribe(body: models.ListSubscribe, tasks: BackgroundTasks,
                  _l: None = Depends(ratelimit.limit("list_signup"))) -> Dict[str, Any]:
        """Sends one confirm email; the address joins the list only after the link is opened.
        Answers the same way whether or not the address was already on the list."""
        from db.marketing_list import normalize_email, subscribe as add
        from services.marketing_list import enabled, send_confirm
        from ui.acquisition import attribution_from_params

        if not enabled():
            raise HTTPException(503, "The email list isn't open right now.")
        email = normalize_email(body.email)
        if not email:
            raise HTTPException(422, "That doesn't look like an email address.")
        if body.website:  # bots fill every field; people never see this one
            return {"ok": True, "message": CHECK_INBOX}
        params = {str(k)[:40]: str(v)[:200] for k, v in (body.attribution or {}).items()}
        result = add(email, attribution_from_params(params, params.get("referrer")), body.surface or "web")
        if result is None:
            raise HTTPException(503, "Couldn't save that right now. Please try again in a minute.")
        if result["send_confirm"]:
            tasks.add_task(send_confirm, email, result["token"])
        return {"ok": True, "message": CHECK_INBOX}

    _LINK = {400: {"description": "Invalid link"}, 429: {"description": "Too many attempts"}}

    def _state(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        from ui.log_privacy import mask_email

        if not row:
            raise HTTPException(400, "This link isn't valid. Sign up again from the HSF home page.")
        return {"email": mask_email(row["email"]), "status": row["status"]}

    @app.get("/v1/public/subscribe/status", response_model=models.ListStatus, responses=_LINK,
             summary="Email list status behind a confirm or unsubscribe link")
    def list_status(t: str = Query(min_length=10, max_length=64),
                    _l: None = Depends(ratelimit.limit("unsubscribe"))) -> Dict[str, Any]:
        """Changes nothing (mail scanners open links); the page asks for a click."""
        from db.marketing_list import lookup

        return _state(lookup(t))

    @app.post("/v1/public/subscribe/action", response_model=models.ListStatus, responses=_LINK,
              summary="Confirm or unsubscribe with an emailed link")
    def list_action(body: models.ListAction, tasks: BackgroundTasks,
                    _l: None = Depends(ratelimit.limit("unsubscribe"))) -> Dict[str, Any]:
        from db.marketing_list import confirm, unsubscribe
        from services.marketing_list import sync_confirmed, sync_unsubscribed

        row = confirm(body.token) if body.action == "confirm" else unsubscribe(body.token)
        state = _state(row)
        if row and row.get("changed"):
            tasks.add_task(sync_confirmed if row["status"] == "confirmed" else sync_unsubscribed, row["email"])
        return state

    @app.get("/v1/public/weekly", response_model=models.PublicRecapIndex, summary="The latest weekly recap (public)")
    def weekly_index() -> Dict[str, Any]:
        from api.public_recap import public_index

        return public_index()

    @app.get("/v1/public/weekly/{week}", response_model=models.PublicRecap,
             responses={404: {"description": "No recap for that week"}}, summary="One week's recap (public)")
    def weekly(week: str = Path(pattern=r"^\d{4}-W\d{2}$")) -> Dict[str, Any]:
        from api.public_recap import BadWeek, recap

        try:
            found = recap(week)
        except BadWeek as e:
            raise HTTPException(404, str(e)) from None
        if not found:
            raise HTTPException(404, "There's no recap for that week yet.")
        return found

    @app.get("/v1/admin/marketing", response_model=models.AdminMarketing, responses=_ADMIN, tags=[_TAG],
             openapi_extra=_EXTRA, summary="Where visitors, list sign-ups, accounts and sales come from")
    def admin_marketing(days: int = 30, _a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        from api.marketing_report import marketing

        return _analytics(lambda: marketing(days))

    @app.get("/v1/admin/marketing/subscribers.csv", responses={**_ADMIN, 200: {"content": {"text/csv": {}}}},
             tags=[_TAG], openapi_extra=_EXTRA, response_class=Response,
             summary="Confirmed email-list addresses as CSV (for a newsletter tool import)")
    def admin_subscribers_csv(status: str = Query("confirmed", pattern="^(confirmed|pending|unsubscribed|all)$"),
                              _a: Dict[str, Any] = Depends(_admin)) -> Response:
        from api.marketing_report import subscribers_csv
        from db.marketing_list import all_rows

        return Response(subscribers_csv(all_rows(), status=status), media_type="text/csv",
                        headers={"content-disposition": f'attachment; filename="hsf-email-list-{status}.csv"',
                                 "cache-control": "no-store"})
