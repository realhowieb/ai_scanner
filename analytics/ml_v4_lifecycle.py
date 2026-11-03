"""ML v4 model lifecycle rules (pure; the registry in ``db/ml_model_registry.py`` enforces them).

    candidate -> validated -> shadow -> champion -> retired
        \\____________\\__________\\___________________^      (any stage can be retired)
    retired -> champion only as a ROLLBACK to a model that was champion before

* candidate: registered after training; nothing is known about it yet.
* validated: walk-forward + holdout evidence attached and reviewed.
* shadow:    scored live on new observations without touching production.
* champion:  the ML v4 registry's approved model for one target + horizon.
             Exactly one per target + horizon. Becoming champion needs an
             admin's explicit approval and passing promotion evidence.
             NOTE: this does not by itself change what customers see; wiring
             a champion into AI Confidence is a separate, later decision.
* retired:   kept for history and rollback; never deleted.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

STATUSES = ("candidate", "validated", "shadow", "champion", "retired")
TRANSITIONS: Dict[str, frozenset] = {
    "candidate": frozenset({"validated", "retired"}),
    "validated": frozenset({"shadow", "retired"}),
    "shadow": frozenset({"champion", "retired"}),
    "champion": frozenset({"retired"}),
    "retired": frozenset({"champion"}),  # rollback only (checked below)
}


class LifecycleError(ValueError):
    """A transition the lifecycle does not allow (nothing was changed)."""


def check_transition(from_status: str, to_status: str, *, actor: Optional[str], reason: Optional[str],
                     evidence: Optional[Mapping[str, Any]] = None, approved: bool = False,
                     was_champion_before: bool = False) -> None:
    """Raise ``LifecycleError`` unless ``from_status -> to_status`` is allowed with this context."""
    if from_status not in STATUSES or to_status not in STATUSES:
        raise LifecycleError(f"unknown status {from_status!r} -> {to_status!r}")
    if to_status not in TRANSITIONS[from_status]:
        raise LifecycleError(f"{from_status} -> {to_status} is not an allowed transition")
    if not (actor and str(actor).strip()):
        raise LifecycleError("every transition needs the acting admin")
    if not (reason and str(reason).strip()):
        raise LifecycleError("every transition needs a reason")
    if to_status == "validated" and not (evidence or {}).get("evaluation"):
        raise LifecycleError("validated needs evaluation evidence (walk-forward + holdout report)")
    if to_status == "champion":
        if not approved:
            raise LifecycleError("champion needs explicit admin approval")
        if from_status == "retired" and not was_champion_before:
            raise LifecycleError("a retired model can only return to champion as a rollback to a former champion")
        if from_status == "shadow" and not (evidence or {}).get("promotion_gates_passed"):
            raise LifecycleError("champion needs a passing promotion-gate report")


def rollback_target(history: Iterable[Mapping[str, Any]], current_champion_id: Optional[int]) -> Optional[int]:
    """The most recent former champion (by event time) other than the current one.

    ``history`` = registry events for one target + horizon with keys model_id,
    to_status, created_at, status (the model's status now)."""
    best = None
    for e in history:
        if e.get("to_status") != "champion" or e.get("model_id") == current_champion_id:
            continue
        if e.get("status") != "retired":
            continue
        if best is None or (str(e["created_at"]), e["model_id"]) > (str(best["created_at"]), best["model_id"]):
            best = e
    return int(best["model_id"]) if best else None
