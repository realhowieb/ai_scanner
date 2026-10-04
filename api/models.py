"""Response models for the HSF API (P1-59).

Every route declares one, so /docs documents each payload and iOS/Android/web
clients can be generated from the OpenAPI schema.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Health(BaseModel):
    ok: bool


class TokenPair(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds")
    refresh_token: str = Field(description="Single use; POST /v1/auth/refresh returns a new one")


class Me(BaseModel):
    email: str
    name: Optional[str] = None
    plan: Literal["basic", "pro", "premium", "admin"]
    plan_label: str = Field(description="Customer-facing name, e.g. Free, Pro")
    is_admin: bool
    alert_limit: int
    entitlements: Dict[str, bool] = Field(description="Feature flags, e.g. can_day_trader, can_ai_notes")


class Market(BaseModel):
    phase: Literal["premarket", "open", "afterhours", "closed"]


class Mover(BaseModel):
    ticker: str
    pct: Optional[float] = Field(default=None, description="Move in percent vs the previous close")
    last: Optional[float] = None
    score: Optional[int] = Field(default=None, description="HSF Score when the name qualifies")


class SessionCard(BaseModel):
    """Before the open / After the close. Pro+; below Pro `locked` is true and `movers` empty."""
    scan_at: Optional[str] = Field(default=None, description="ISO time of the session scan")
    locked: bool
    movers: List[Mover] = []


class Setup(BaseModel):
    ticker: str
    score: int
    primary_setup: Optional[str] = None
    status: Optional[str] = None
    n_signals: Optional[int] = None
    last: Optional[float] = None
    chg_pct: Optional[float] = None
    gap_pct: Optional[float] = None
    rvol: Optional[float] = None
    prob: Optional[float] = Field(default=None, description="PreBreakout model output; null below Premium")


class TopSetups(BaseModel):
    state: Literal["qualifying", "no_qualifying", "empty_scan"]
    threshold: Optional[int] = None
    scan_at: Optional[str] = None
    setups: List[Setup] = []


class Standout(BaseModel):
    ticker: str
    score: int
    setup: Optional[str] = None


class Recap(BaseModel):
    day: str = Field(description="ET date of the recapped session (YYYY-MM-DD)")
    title: str
    scans: int = Field(description="Full-market scans that day")
    premarket_scans: int = 0
    postmarket_scans: int = 0
    entered: List[str] = Field(default=[], description="'TICKER (score)', HSF 40+, strongest first")
    left: List[str] = []
    standouts: List[Standout] = []


class SectionError(BaseModel):
    section: str
    error: str = Field(description="Error type only, never details")


class Today(BaseModel):
    as_of: str
    market: Market
    before_open: Optional[SessionCard] = None
    top_setups: Optional[TopSetups] = None
    after_close: Optional[SessionCard] = None
    recap: Optional[Recap] = None
    errors: List[SectionError] = []
