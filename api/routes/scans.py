"""Custom scans (POST /v1/scans) and their jobs."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from api import models, ratelimit, user_data
from api.deps import _AUTH, _OWNED, _user, current_account, entitlements_for
from api.scans import json_safe


class ScanFilters(BaseModel):
    """Same filters and ranges as the web's Custom Scan page (defaults match it)."""
    min_price: float = Field(1.0, ge=0.5, le=500)
    max_price: float = Field(1000.0, ge=1, le=5000)
    min_dollar_vol: float = Field(5_000_000.0, ge=0, le=1e12, description="20-day average dollar volume floor")
    min_gap: float = Field(1.0, ge=0, le=20, description="Used with apply_gap_filter (Pro+)")
    apply_gap_filter: bool = Field(False, description="Pro+")
    unusual_volume: bool = Field(False, description="Pro+")
    session: Literal["regular", "premarket", "afterhours"] = Field("regular", description="premarket / afterhours: Pro+")
    profile: Literal["regular", "aggressive", "conservative"] = "regular"
    top_n: Optional[int] = Field(None, ge=5, le=10_000,
                                 description="Rows to return; at most the plan's cap (Free 25, Pro 100, Premium 200). "
                                             "Default min(25, cap)")
    max_nasdaq: Optional[int] = Field(None, ge=100, le=100_000,
                                      description="NASDAQ ticker cap, Pro up to 4000 (default 1200); "
                                                  "ignored for Premium (full list)")
    max_combo: Optional[int] = Field(None, ge=100, le=100_000,
                                     description="Combo ticker cap, Pro up to 6000 (default 1000); "
                                                 "ignored for Premium (full list)")


class ScanCreate(BaseModel):
    universe: Literal["sp500", "nasdaq", "combo", "us_market", "watchlist", "ticker"] = Field(
        description="sp500 (every plan) · nasdaq, combo (Pro+) · us_market (Premium+) · "
                    "watchlist (one of yours) · ticker (one symbol)")
    ticker: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$", description="For universe=ticker")
    watchlist_id: Optional[int] = Field(None, ge=1, description="For universe=watchlist")
    score_all: bool = Field(False, description="Watchlist only: score every symbol, ignoring the screens (as on the web)")
    filters: ScanFilters = Field(default_factory=ScanFilters)


def _job_out(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {**row, "scan_id": row["id"]}
    return json_safe(out)


def register(app: FastAPI) -> None:
    from api import custom_scans, scan_jobs

    _SCAN_ERRORS = {**_AUTH, 403: {"description": "Not in your plan (universe, rows, session or filter)"},
                    404: {"description": "Watchlist not found (or not yours)"},
                    409: {"description": "You already have a scan queued or running (body has scan_id)"},
                    422: {"description": "Invalid request"}, 429: {"description": "Too many scans this hour"},
                    503: {"description": "Scans busy or database unavailable (Retry-After)"}}

    @app.post("/v1/scans", response_model=models.ScanJob, status_code=202, responses=_SCAN_ERRORS,
              summary="Start a custom scan")
    def scan_create(body: ScanCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Queue a custom scan (the web's Custom Scan page). Plan rules are checked here; the
        scan runs in the background — poll GET /v1/scans/{scan_id} (every 2-5 s) until
        status is complete or failed. One scan per account at a time; 30 per hour."""
        user = _user(account)
        ratelimit.check("scan", user)
        try:
            params = custom_scans.plan_or_raise(body.model_dump(), entitlements_for(account))
        except custom_scans.PlanError:
            raise
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if params["universe"] == "watchlist":
            user_data.get_watchlist(user, int(params["watchlist_id"]))   # 404 when not yours
        job = scan_jobs.create_job(user, params["universe"], params)
        scan_jobs.submit(job["id"], lambda report: custom_scans.run_scan(params, user, report))
        return _job_out(job)

    @app.get("/v1/scans", response_model=List[models.ScanJob], responses=_AUTH, summary="Your recent custom scans")
    def scan_list(account: Dict[str, Any] = Depends(current_account),
                  limit: int = Query(10, ge=1, le=50)) -> List[Dict[str, Any]]:
        """Newest first, without results (fetch one by id for its rows). Kept 7 days."""
        return [_job_out(j) for j in scan_jobs.list_jobs(_user(account), limit)]

    @app.get("/v1/scans/{scan_id}", response_model=models.ScanJob, responses=_OWNED, summary="A custom scan's status and results")
    def scan_get(scan_id: str = Path(pattern="^[0-9a-f]{32}$"),
                 account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        job = scan_jobs.get_job(_user(account), scan_id)
        if job is None:
            raise user_data.NotFound("scan")
        return _job_out(job)

    @app.delete("/v1/scans/{scan_id}", response_model=models.ScanJob, responses=_OWNED, summary="Cancel a custom scan")
    def scan_cancel(scan_id: str = Path(pattern="^[0-9a-f]{32}$"),
                    account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Cancel your queued or running scan: it reads `failed` with error "Cancelled." and you
        can start another at once. A running scan stops at its next progress step. A scan that
        already finished is returned unchanged."""
        job = scan_jobs.cancel_job(_user(account), scan_id)
        if job is None:
            raise user_data.NotFound("scan")
        return _job_out(job)
