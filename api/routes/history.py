"""Scan history and the track record (Pro)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query

from api import models, user_data
from api.deps import _AUTH, _OWNED, _user, current_account, require_feature
from api.scans import json_safe


def register(app: FastAPI) -> None:
    from api import history

    _PRO = {**_AUTH, 403: {"description": "Pro feature"}}

    @app.get("/v1/runs", response_model=List[models.RunSummary], responses=_PRO, summary="Your scan history (Pro)")
    def runs(account: Dict[str, Any] = Depends(current_account), limit: int = Query(50, ge=1, le=200),
             include_snapshots: bool = Query(False, description="Include daily snapshot copies")) -> List[Dict[str, Any]]:
        """Your saved scans, newest first (the web's Scan History tab)."""
        require_feature(account, "can_scan_history")
        return json_safe(history.saved_runs(_user(account), limit, include_snapshots))

    @app.get("/v1/runs/{run_id}", response_model=models.RunDetail, responses={**_PRO, **_OWNED},
             summary="One of your saved scans with its rows (Pro)")
    def run_detail(run_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        ent = require_feature(account, "can_scan_history")
        from api.scans import max_results_for

        out = history.get_run(_user(account), run_id, early_breakout=bool(ent["entitlements"].get("can_early_breakout")),
                              max_results=max_results_for(ent["tier"]))
        if out is None:
            raise user_data.NotFound("scan")
        return json_safe(out)

    @app.get("/v1/track-record", response_model=models.TrackRecord, responses=_PRO,
             summary="Historical research: saved scan picks vs SPY (Pro)")
    def track_record(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Descriptive backtest summaries by ranking and horizon (computed daily by the scheduler)."""
        require_feature(account, "can_track_record")
        return json_safe(history.track_record())

    @app.get("/v1/track-record/daily", response_model=List[models.TrackRecordDay], responses=_PRO,
             summary="Daily excess return vs SPY (Pro)")
    def track_record_daily(account: Dict[str, Any] = Depends(current_account),
                           ranking: Literal["breakout", "prebreakout"] = "breakout",
                           horizon: int = Query(5, description="1, 3, 5, 10 or 20 trading days"),
                           days: int = Query(120, ge=1, le=365)) -> List[Dict[str, Any]]:
        require_feature(account, "can_track_record")
        if horizon not in history.HORIZONS:
            raise HTTPException(422, "horizon must be 1, 3, 5, 10 or 20.")
        return json_safe(history.track_record_daily(ranking, horizon, days))
