"""Daily research helpers; preferences and notifications are scoped to the session."""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from api.deps import _AUTH, _user, current_account
from api.workspace import WorkspacePatch, WorkspacePreferences


class SymbolSuggestion(BaseModel):
    ticker: str
    name: str


class SymbolSuggestions(BaseModel):
    results: list[SymbolSuggestion]
    available: bool


def register(app: FastAPI):
    @app.get("/v1/me/research-preferences", response_model=WorkspacePreferences, responses=_AUTH)
    def get_preferences(account: dict = Depends(current_account)):
        from api.workspace import preferences
        return preferences(_user(account))

    @app.patch("/v1/me/research-preferences", response_model=WorkspacePreferences, responses=_AUTH)
    def update_preferences(body: WorkspacePatch, account: dict = Depends(current_account)):
        from api import ratelimit
        from api.workspace import preferences
        ratelimit.check("research_preferences", _user(account))
        return preferences(_user(account), body.model_dump(exclude_none=True))

    @app.get("/v1/symbols/search", response_model=SymbolSuggestions, responses=_AUTH)
    def symbols(q: str = Query(..., min_length=1, max_length=80), account: dict = Depends(current_account)):
        from api import ratelimit
        from api.symbol_search import search
        ratelimit.check("symbol_search", _user(account))
        return search(q)

    @app.get("/v1/alerts/deliveries", response_model=list[Delivery], responses=_AUTH)
    def deliveries(account: dict = Depends(current_account)):
        from api.alert_delivery import recent
        return recent(_user(account))

    @app.post("/v1/alerts/deliveries/{delivery_id}/retry", response_model=Delivery, responses=_AUTH)
    def retry_delivery(delivery_id: str, account: dict = Depends(current_account)):
        from api import ratelimit
        from api.alert_delivery import retry
        ratelimit.check("notification_retry", _user(account))
        result = retry(_user(account), delivery_id)
        if not result:
            raise HTTPException(409, "This notification cannot be retried. Refresh the delivery list.")
        return result


class Delivery(BaseModel):
    id: str
    status: str
    attempts: int
    at: str
    title: str
    retryable: bool
