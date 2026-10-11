"""Internal, admin-only research dataset and ML readiness."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query

from api.deps import _ADMIN, current_account, require_admin
from api.scans import json_safe

_RESEARCH_TAG = "research (internal, admin only)"
_RESEARCH_EXTRA = {"x-internal": True, "x-audience": "admin research-only"}


def register(app: FastAPI) -> None:
    """Internal, admin-only, read-only research dataset (no Web v2 navigation)."""
    from api import research

    def bad(e: Exception) -> HTTPException:
        return HTTPException(422, str(e))

    def route(path: str, summary: str, responses: Optional[Dict[int, Any]] = None):
        return app.get(path, tags=[_RESEARCH_TAG], openapi_extra=_RESEARCH_EXTRA, responses=responses or _ADMIN,
                       summary=f"[Internal · admin · research-only] {summary}")

    @route("/v1/research/datasets", "Finalized research dataset versions and the schemas")
    def research_datasets(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Finalized (immutable) dataset versions, plus the current feature and label schemas.
        Versions are created only by `scripts/research_dataset.py --finalize`, never by the API."""
        require_admin(account)
        return json_safe(research.datasets())

    @route("/v1/research/datasets/{dataset_version}", "One finalized research dataset version",
           {**_ADMIN, 404: {"description": "No such version"}})
    def research_dataset(dataset_version: str = Path(pattern=r"^[a-z0-9][a-z0-9.\-]{2,80}$"),
                         verify: bool = Query(False, description="Rebuild from current data and compare fingerprints"),
                         account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        out = research.dataset(dataset_version, verify=verify)
        if out is None:
            raise HTTPException(404, "No such dataset version.")
        return json_safe(out)

    @route("/v1/research/coverage", "Real coverage of the point-in-time research dataset",
           {**_ADMIN, 422: {"description": "Bad date window"}})
    def research_coverage(account: Dict[str, Any] = Depends(current_account),
                          start_date: Optional[dt.date] = Query(None), end_date: Optional[dt.date] = Query(None)
                          ) -> Dict[str, Any]:
        """Counts and per-feature/label coverage computed from persisted rows in the window
        (default: last 90 days), plus a data-quality report. Nothing is estimated."""
        require_admin(account)
        try:
            return json_safe(research.coverage(start_date, end_date))
        except research.BadRequest as e:
            raise bad(e) from None

    @route("/v1/research/observations", "Paginated point-in-time research observations",
           {**_ADMIN, 404: {"description": "No such dataset version"}, 422: {"description": "Bad filter"}})
    def research_observations(
            account: Dict[str, Any] = Depends(current_account),
            start_date: Optional[dt.date] = Query(None), end_date: Optional[dt.date] = Query(None),
            ticker: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$"),
            setup: Optional[str] = Query(None, max_length=40),
            min_score: Optional[float] = Query(None, ge=0, le=100), max_score: Optional[float] = Query(None, ge=0, le=100),
            horizon: Optional[int] = Query(None, description="1, 3 or 5 trading days (used by matured_only)"),
            matured_only: bool = False, certified_only: bool = False,
            dataset_version: Optional[str] = Query(None, pattern=r"^[a-z0-9][a-z0-9.\-]{2,80}$"),
            model_version: Optional[str] = Query(None, max_length=60, description="UNKNOWN matches rows with no recorded version"),
            scoring_version: Optional[str] = Query(None, max_length=20, description="UNKNOWN matches legacy rows"),
            include_outcomes: bool = Query(False, description="Add a separate `outcome` object per item"),
            limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
        """Each item keeps `observation` (identity, provenance, maturity, overlap group) and `features`
        (point-in-time only) apart; outcomes appear only under `outcome` with include_outcomes=true."""
        require_admin(account)
        try:
            return json_safe(research.observations(
                limit=limit, offset=offset, include_outcomes=include_outcomes, dataset_version=dataset_version,
                start_date=start_date, end_date=end_date, ticker=ticker, setup=setup, min_score=min_score,
                max_score=max_score, horizon=horizon, matured_only=matured_only, certified_only=certified_only,
                model_version=model_version, scoring_version=scoring_version))
        except research.BadRequest as e:
            raise bad(e) from None
        except LookupError:
            raise HTTPException(404, "No such dataset version.") from None

    @route("/v1/research/observations/{observation_id}", "One research observation",
           {**_ADMIN, 404: {"description": "No such observation"}})
    def research_observation(observation_id: int = Path(ge=1), include_outcome: bool = Query(False),
                             account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        out = research.observation(observation_id, include_outcome=include_outcome)
        if out is None:
            raise HTTPException(404, "No such observation.")
        return json_safe(out)

    @route("/v1/research/features/{observation_id}", "Point-in-time feature snapshot (no outcomes)",
           {**_ADMIN, 404: {"description": "No such observation"}})
    def research_features(observation_id: int = Path(ge=1),
                          account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Only what HSF knew at observed_at. Never returns returns, MFE/MAE, benchmark returns or
        maturity; outcomes come from Outcome Intelligence or /observations?include_outcomes=true."""
        require_admin(account)
        out = research.features(observation_id)
        if out is None:
            raise HTTPException(404, "No such observation.")
        return json_safe(out)


def register_ml(app: FastAPI) -> None:
    """Internal, admin-only, read-only ML v4 data readiness and the operations summary."""
    from api import ml_readiness

    @app.get("/v1/ml/readiness", tags=["ml (internal, admin only)"], openapi_extra=_RESEARCH_EXTRA, responses=_ADMIN,
             summary="[Internal · admin · research-only] ML v4 data readiness: gates, coverage, maturation")
    def ml_readiness_status(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Whether HSF has collected enough trustworthy, matured point-in-time observations to start ML v4
        development: status (NOT_READY / COLLECTING / NEAR_READY / READY), every readiness gate with its
        threshold and reason, coverage, maturation diagnostics and a growth projection. Aggregates only;
        computed from persisted rows (cached 30 minutes). Never trains, scores or changes anything."""
        require_admin(account)
        return json_safe(ml_readiness.readiness())

    @app.get('/v1/admin/operations', tags=['operations (admin only)'],
             responses={401: {'description': 'Not signed in'}, 403: {'description': 'Admins only'}},
             openapi_extra={'x-internal': True})
    def operations_health(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_admin(account)
        from api.operations import get_summary
        return get_summary()
