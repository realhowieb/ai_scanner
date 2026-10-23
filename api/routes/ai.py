"""AI summaries and chat (Premium)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from api import models, ratelimit, user_data
from api.deps import _AUTH, TICKER, _user, current_account, entitlements_for, require_feature
from api.scans import json_safe


class AISummaryBody(BaseModel):
    run_id: Optional[int] = Field(None, ge=1, description="One of your saved scans (GET /v1/runs); default: the latest market scan")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AIChatBody(AISummaryBody):
    messages: List[ChatTurn] = Field(min_length=1, max_length=16,
                                     description="The conversation so far, oldest first, ending with the new question")


class AIAskBody(BaseModel):
    messages: List[ChatTurn] = Field(min_length=1, max_length=16,
                                     description="The conversation so far, oldest first, ending with the new question")


class AIScanBuilderBody(BaseModel):
    text: str = Field(min_length=3, max_length=500, description="The scan you want, in plain English")


def register(app: FastAPI) -> None:
    from api import ai

    _AI = {**_AUTH, 403: {"description": "Premium feature"}, 404: {"description": "Scan not found (or not yours)"},
           429: {"description": "Daily AI limit or hourly request limit reached"},
           502: {"description": "AI call failed"}, 503: {"description": "AI unavailable"}}

    def _premium(account: Dict[str, Any]) -> str:
        require_feature(account, "can_ai_notes")
        user = _user(account)
        ratelimit.check("ai", user)
        return user

    @app.post("/v1/ai/summary", response_model=models.AIText, responses=_AI, summary="AI scan summary (Premium)")
    def ai_summary(body: AISummaryBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Claude summarizes the top results in HSF Score order (the web's AI Scan Summary).
        Research commentary, not investment advice."""
        user = _premium(account)
        out = ai.summary(user, body.run_id, shared=body.run_id is None)
        if out["run_id"] is None and body.run_id is not None:
            raise user_data.NotFound("scan")
        return out

    @app.post("/v1/ai/chat", response_model=models.AIChatAnswer, responses=_AI, summary="Ask about a scan (Premium)")
    def ai_chat(body: AIChatBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Questions about one scan's results (the web's results chat). Send the conversation so
        far; the last message must be the user's question. Up to 8 prior turns are used."""
        user = _premium(account)
        if body.messages[-1].role != "user":
            raise HTTPException(422, "The last message must be the user's question.")
        out = ai.chat(user, body.run_id, [m.model_dump() for m in body.messages])
        if out["run_id"] is None and body.run_id is not None:
            raise user_data.NotFound("scan")
        return out

    @app.post("/v1/ai/notes/{ticker}", response_model=models.AIText, responses=_AI,
              summary="AI setup note for a ticker (Premium)")
    def ai_note(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Claude's note on one result of the latest market scan; text is null when the ticker
        isn't in it."""
        user = _premium(account)
        return ai.ticker_note(user, ticker.strip().upper())

    @app.post("/v1/ai/watchlists/{watchlist_id}/note", response_model=models.AIText, responses=_AI,
              summary="AI rundown of a watchlist (Premium)")
    def ai_watchlist_note(watchlist_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """A short plain-English rundown of what changed on one of your watchlists since the
        previous market scan (score and rank moves, new or lost setups, volume, earnings within
        a week), from the latest saved scan only. text is null when the list is empty or no
        scan exists. Cached per watchlist and scan."""
        user = _premium(account)
        wl = user_data.get_watchlist(user, watchlist_id)
        return json_safe(ai.watchlist_note(user, wl, entitlements_for(account)["entitlements"]))

    @app.get("/v1/ai/brief-narrative", response_model=models.AIText, responses=_AI,
             summary="AI Market Brief narrative (Premium)")
    def ai_brief_narrative(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """A 2-3 sentence brief written from the Market Brief's facts only."""
        user = _premium(account)
        return json_safe(ai.brief_narrative(user))

    @app.post("/v1/ai/ask", response_model=models.AIChatAnswer, responses=_AI, summary="Ask HSF (Premium)")
    def ai_ask(body: AIAskBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Questions about your watchlists, the latest market scan, earnings, news headlines and
        HSF's track record, answered from HSF's data only. Send the conversation so far; the last
        message must be the user's question. Up to 8 prior turns are used."""
        from api import ai_premium

        user = _premium(account)
        if body.messages[-1].role != "user":
            raise HTTPException(422, "The last message must be the user's question.")
        return ai_premium.ask(user, entitlements_for(account)["entitlements"], [m.model_dump() for m in body.messages])

    @app.post("/v1/ai/journal-coach", response_model=models.AIJournalCoach, responses=_AI,
              summary="AI journal coach (Premium)")
    def ai_journal_coach(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """A short note on patterns in your closed trades (by HSF Score when taken, holding time,
        source; win and loss sizes). text is null below min_trades. Cached per journal state."""
        from api import ai_premium

        user = _premium(account)
        return json_safe(ai_premium.journal_coach(user))

    @app.post("/v1/ai/scan-builder", response_model=models.AIScanBuilder, responses=_AI,
              summary="Describe a scan in plain English (Premium)")
    def ai_scan_builder(body: AIScanBuilderBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Turns a plain-English request into Custom scan settings (the saved-view shape). Nothing
        runs; plan rules still apply when you start the scan."""
        from api import ai_premium

        user = _premium(account)
        return ai_premium.scan_builder(user, body.text)
