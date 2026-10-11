"""Outcome Intelligence (/v1/outcomes/*)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from api import models
from api.deps import _AUTH, TICKER, current_account, require_feature
from api.scans import json_safe


class OutcomeParams:
    """Explicit, validated Outcome Intelligence filters. Nothing is filtered unless
    asked for, and the response echoes every filter back."""

    def __init__(self,
                 setup: Optional[str] = Query(None, max_length=40, description="Canonical setup (primary_setup), e.g. breakout"),
                 signal: Optional[str] = Query(None, max_length=40, description="Signal label, e.g. golden_cross"),
                 min_score: Optional[float] = Query(None, ge=0, le=100),
                 max_score: Optional[float] = Query(None, ge=0, le=100),
                 score_bucket: Optional[str] = Query(None, pattern=r"^\d{1,3}-\d{1,3}$", description="e.g. 80-89"),
                 score_version: Optional[str] = Query(None, max_length=20, description="HSF score version as frozen"),
                 start_date: Optional[dt.date] = Query(None, description="Observed on/after (YYYY-MM-DD, UTC)"),
                 end_date: Optional[dt.date] = Query(None, description="Observed on/before (YYYY-MM-DD, UTC)"),
                 certified_only: bool = Query(False, description="Only matured rows passing the canonical eligibility rule"),
                 matured_only: bool = Query(False, description="Only records matured at the horizon (counts drop pending)"),
                 unit: Literal["signal_day", "observation"] = Query(
                     "signal_day", description="signal_day: one record per ticker per entry day (default); observation: every frozen row")):
        from analytics import outcome_intelligence as oi

        self.unit = unit
        try:
            self.filters = oi.normalize_filters(setup=setup, signal=signal, min_score=min_score, max_score=max_score,
                                                score_bucket=score_bucket, score_version=score_version,
                                                start_date=start_date, end_date=end_date,
                                                certified_only=certified_only, matured_only=matured_only)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e


def register(app: FastAPI) -> None:
    from api import outcomes

    _PRO = {**_AUTH, 403: {"description": "Pro feature"}, 422: {"description": "Invalid filter"},
            503: {"description": "Database unavailable"}}
    HORIZON = Query(5, description="Trading days: 1, 3 or 5. 5 is the pre-declared primary horizon, not a data-chosen one")

    def _run(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return json_safe(fn())
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    def _horizon(h: Optional[int]) -> Optional[int]:
        if h is not None and h not in (1, 3, 5):
            raise HTTPException(422, "horizon must be 1, 3 or 5.")
        return h

    @app.get("/v1/outcomes/summary", response_model=models.OutcomeSummary, responses=_PRO,
             summary="Outcome Intelligence: overall evidence (Pro)")
    def outcomes_summary(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                         horizon: int = HORIZON) -> Dict[str, Any]:
        """Every eligible HSF signal unless filters are given: counts, date range, raw and
        SPY-relative returns, win and beat rates, MFE/MAE, with sample sizes."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.summary(p.filters, h, p.unit))

    @app.get("/v1/outcomes/scores", response_model=models.OutcomeScores, responses=_PRO,
             summary="Outcome Intelligence: by HSF score bucket (Pro)")
    def outcomes_scores(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                        horizon: int = HORIZON,
                        buckets: Optional[str] = Query(None, max_length=120, pattern=r"^[0-9,\- ]+$",
                                                       description="e.g. 40-49,50-59,60-69 (default: canonical HSF buckets)")
                        ) -> Dict[str, Any]:
        """Every bucket in score order, weak ones included, plus a monotonicity check
        that lists each inversion (a lower bucket beating a higher one)."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.scores(p.filters, h, p.unit, buckets))

    @app.get("/v1/outcomes/horizons", response_model=models.OutcomeHorizons, responses=_PRO,
             summary="Outcome Intelligence: by horizon (Pro)")
    def outcomes_horizons(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends()) -> Dict[str, Any]:
        """The same metrics for 1, 3 and 5 trading days. No horizon is singled out."""
        require_feature(account, "can_track_record")
        return _run(lambda: outcomes.horizons(p.filters, p.unit))

    @app.get("/v1/outcomes/setups", response_model=models.OutcomeGroups, responses=_PRO,
             summary="Outcome Intelligence: by setup or signal (Pro)")
    def outcomes_setups(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                        horizon: int = HORIZON,
                        group_by: Literal["setup", "signal"] = Query("setup")) -> Dict[str, Any]:
        """Groups ordered by sample size (never by performance), each with its counts."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.setups(p.filters, h, p.unit, group_by))

    @app.get("/v1/outcomes/timeseries", response_model=models.OutcomeTimeseries, responses=_PRO,
             summary="Outcome Intelligence: through time (Pro)")
    def outcomes_timeseries(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                            horizon: int = HORIZON,
                            period: Literal["day", "week", "month"] = Query("week")) -> Dict[str, Any]:
        """Per observation period (grouped by when the signal was observed)."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.timeseries(p.filters, h, p.unit, period))

    @app.get("/v1/outcomes/symbols/{ticker}", response_model=models.OutcomeSymbol, responses=_PRO,
             summary="Outcome Intelligence: one ticker's HSF history (Pro)")
    def outcomes_symbol(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account),
                        p: OutcomeParams = Depends(),
                        horizon: Optional[int] = Query(None, description="1, 3 or 5; omit for all three"),
                        page: int = Query(1, ge=1, le=10_000), page_size: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
        """Aggregate evidence per horizon plus the observation-level records, newest first."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.symbol(ticker.strip().upper(), p.filters, h, p.unit, page, page_size))

    @app.get("/v1/outcomes/query", response_model=models.OutcomeQuery, responses=_PRO,
             summary="Outcome Intelligence: filtered evidence with records (Pro)")
    def outcomes_query(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                       ticker: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$"),
                       horizon: int = HORIZON,
                       page: int = Query(1, ge=1, le=10_000), page_size: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
        """Any combination of the explicit filters, one horizon: metrics plus paged records."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        f = {**p.filters, "ticker": ticker.strip().upper() if ticker else None}
        return _run(lambda: outcomes.query(f, h, p.unit, page, page_size))
