"""ML v4 model registry and shadow mode (admins only, internal).

Read the registry, move a model through its lifecycle, roll back a champion and
see forward shadow results. Model files, artifact locations and credentials are
never returned. Nothing here changes what customers see: a registry champion is
not wired into AI Confidence.

Evidence is built on the server: "validated" needs the evaluation stored on the
model row, "champion" needs the pre-registered promotion gates to pass on
forward shadow predictions plus an explicit approval. The API never imports the
training libraries.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from api.deps import _ADMIN, _user, current_account
from api.scans import json_safe

_TAG = "ml models (internal, admin only)"
_EXTRA = {"x-internal": True, "x-audience": "admin ml-only"}
_HIDDEN = ("artifact_uri",)  # where the file lives is internal; the checksum is enough to identify it
_LABEL = r"^[A-Za-z0-9_]{1,40}$"


class MlTransition(BaseModel):
    to_status: Literal["validated", "shadow", "champion", "retired"]
    reason: str = Field(min_length=3, max_length=500)
    approve: bool = Field(False, description="Required (true) to make a model champion")


class MlRollback(BaseModel):
    target_label: str = Field(pattern=_LABEL)
    horizon_days: int = Field(ge=1, le=5)
    reason: str = Field(min_length=3, max_length=500)
    approve: bool = Field(False, description="Must be true: rollback changes the registry champion")


def _admin(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
    if not account.get("is_admin"):
        raise HTTPException(403, "Admins only.")
    return account


def _public(model: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in model.items() if k not in _HIDDEN}
    out["has_artifact"] = bool(model.get("artifact_uri") and model.get("artifact_sha256"))
    return out


def _call(fn):
    from analytics.ml_v4_lifecycle import LifecycleError
    from db.ml_model_registry import RegistryUnavailable
    from db.ml_shadow_predictions import ShadowUnavailable

    try:
        return fn()
    except LifecycleError as e:
        raise HTTPException(422, str(e)) from None
    except (RegistryUnavailable, ShadowUnavailable):
        raise HTTPException(503, "Model registry unavailable.", headers={"Retry-After": "30"}) from None


def _shadow(models: List[Dict[str, Any]], days: int) -> Dict[str, Any]:
    from analytics import ml_v4_shadow as S
    from db import ml_shadow_predictions as D

    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    preds = D.predictions([int(m["id"]) for m in models], since=since)
    by_model: Dict[int, List[Dict[str, Any]]] = {int(m["id"]): [] for m in models}
    for p in preds:
        by_model.setdefault(int(p["model_id"]), []).append(p)
    return {"by_model": by_model, "summaries": {mid: S.summarize_model(v) for mid, v in by_model.items()}}


def register(app: FastAPI) -> None:
    from analytics import ml_v4_shadow as S
    from db import ml_model_registry as R

    responses = {**_ADMIN, 422: {"description": "Refused by the lifecycle rules (the detail says why)"},
                 503: {"description": "Registry unavailable"}}

    def get(path: str, summary: str, extra: Optional[Dict[int, Any]] = None):
        return app.get(path, tags=[_TAG], openapi_extra=_EXTRA, responses={**responses, **(extra or {})},
                       summary=f"[Internal · admin] {summary}")

    def post(path: str, summary: str, extra: Optional[Dict[int, Any]] = None):
        return app.post(path, tags=[_TAG], openapi_extra=_EXTRA, responses={**responses, **(extra or {})},
                        summary=f"[Internal · admin] {summary}")

    @get("/v1/admin/ml/models", "ML v4 registry models")
    def ml_models(status: Optional[Literal["candidate", "validated", "shadow", "champion", "retired"]] = None,
                  target_label: Optional[str] = Query(None, pattern=_LABEL),
                  horizon_days: Optional[int] = Query(None, ge=1, le=5),
                  _a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        rows = _call(lambda: R.list_models(status=status, target_label=target_label, horizon_days=horizon_days))
        return json_safe({"models": [_public(m) for m in rows], "count": len(rows),
                          "note": "A registry champion does not change what customers see."})

    @get("/v1/admin/ml/models/{model_id}", "One ML v4 model: history, shadow results and promotion gates",
         {404: {"description": "No such model"}})
    def ml_model(model_id: int = Path(ge=1), days: int = Query(90, ge=1, le=180),
                 _a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        m = _call(lambda: R.get_model(model_id))
        if m is None:
            raise HTTPException(404, "No such model.")
        events = _call(lambda: R.model_events(model_id))
        summary = _call(lambda: _shadow([m], days))["summaries"][int(m["id"])]
        return json_safe({"model": _public(m), "events": events, "shadow": summary,
                          "promotion_gates": S.evaluate_gates(m, summary) if m["status"] == "shadow" else None})

    @post("/v1/admin/ml/models/{model_id}/transition", "Move an ML v4 model to its next lifecycle stage",
          {404: {"description": "No such model"}})
    def ml_transition(body: MlTransition, model_id: int = Path(ge=1),
                      a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        m = _call(lambda: R.get_model(model_id))
        if m is None:
            raise HTTPException(404, "No such model.")
        evidence: Optional[Dict[str, Any]] = None
        if body.to_status == "validated":
            evidence = S.evaluation_evidence(m)
            if evidence is None:
                raise HTTPException(422, "No evaluation is stored on this model yet; register it with "
                                         "an evaluation report first.")
        elif body.to_status == "champion":
            summary = _call(lambda: _shadow([m], 180))["summaries"][int(m["id"])]
            gates = S.evaluate_gates(m, summary)
            if not gates["passed"]:
                raise HTTPException(422, {"message": "Promotion gates have not passed.", "promotion_gates": gates})
            evidence = {"promotion_gates_passed": True, "promotion_gates": gates}
        res = _call(lambda: R.transition(model_id, body.to_status, actor=_user(a), reason=body.reason,
                                         evidence=evidence, approved=body.approve))
        return json_safe({**res, "note": "Registry only: AI Confidence and alerts are unchanged."})

    @post("/v1/admin/ml/rollback", "Restore the previous ML v4 registry champion")
    def ml_rollback(body: MlRollback, a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        if not body.approve:
            raise HTTPException(422, "Rollback needs approve=true.")
        res = _call(lambda: R.rollback(body.target_label, body.horizon_days, actor=_user(a), reason=body.reason))
        return json_safe({**res, "note": "Registry only: AI Confidence and alerts are unchanged."})

    @get("/v1/admin/ml/shadow", "Forward shadow results for shadow and champion models")
    def ml_shadow(target_label: Optional[str] = Query(None, pattern=_LABEL),
                  horizon_days: Optional[int] = Query(None, ge=1, le=5), days: int = Query(30, ge=1, le=180),
                  _a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        models = [m for st in ("shadow", "champion") for m in _call(
            lambda st=st: R.list_models(status=st, target_label=target_label, horizon_days=horizon_days))]
        if not models:
            return {"models": [], "days": days, "disagreement": [],
                    "note": "No model is in shadow yet."}
        sh = _call(lambda: _shadow(models, days))
        out = [{"model": {k: m.get(k) for k in ("id", "model_version", "provider", "feature_set", "target_label",
                                                 "horizon_days", "status", "status_changed_at")},
                "shadow": sh["summaries"][int(m["id"])],
                "promotion_gates": S.evaluate_gates(m, sh["summaries"][int(m["id"])]) if m["status"] == "shadow"
                else None} for m in models]
        groups: Dict[tuple, Dict[int, Any]] = {}
        for m in models:
            groups.setdefault((m["target_label"], int(m["horizon_days"])), {})[int(m["id"])] = sh["by_model"][int(m["id"])]
        return json_safe({"models": out, "days": days,
                          "disagreement": [{"target_label": k[0], "horizon_days": k[1], **S.disagreement(v)}
                                           for k, v in sorted(groups.items())],
                          "note": "Shadow predictions are never shown to customers."})
