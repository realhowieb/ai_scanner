"""Terms of Service and Privacy Policy for every account: whether they are public, what the
sign-up form and the upgrade button say, and recording that an account accepted them.

Off until HSF_LEGAL_PAGES_ENABLED=1 (after the texts in legal/ are reviewed). Recording is
best effort: a database problem is logged, never turned into a failed sign-up or checkout.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from domain import legal as policies

log = logging.getLogger("hsf_api.legal")

_SESSION = re.compile(r"\b(cs_(?:live|test)_[A-Za-z0-9]+)")


def pages_open() -> bool:
    return policies.general_pages_open()


def page_open(policy_type: str) -> bool:
    """Whether GET /v1/founding/policies/{type} answers signed-out visitors."""
    from api import founding

    if policy_type in policies.GENERAL_POLICIES and pages_open():
        return True
    return policy_type != policies.SITE_TERMS and founding.pages_open()


def status() -> Dict[str, Any]:
    """What the sign-up form and the upgrade button show (public)."""
    if not pages_open():
        return {"open": False, "policies": [], "signup_statement": None, "subscription_statement": None}
    current = policies.current(policies.load())
    return {
        "open": True,
        "policies": [{k: current[t][k] for k in ("policy_type", "version", "title", "path", "content_hash", "status")}
                     for t in policies.GENERAL_POLICIES if t in current],
        "signup_statement": policies.SIGNUP_STATEMENT,
        "subscription_statement": policies.SUBSCRIPTION_STATEMENT,
    }


def _version_ids() -> List[int]:
    from api import founding

    recorded = founding.recorded()
    return [int(recorded[t]["id"]) for t in policies.GENERAL_POLICIES
            if t in recorded and recorded[t].get("available")]


def record(user: str, *, kind: str, ref: str, tier: Optional[str]) -> int:
    """Record that `user` accepted the current Terms and Privacy Policy. `ref` ties the rows
    to the sign-up or the Stripe checkout session. Returns rows added (0 when off or on error)."""
    if not pages_open() or not ref:
        return 0
    try:
        from db import legal as db

        ids = _version_ids()
        return db.record_acceptance(user, ids, checkout_session_id=ref, tier=tier or "", kind=kind) if ids else 0
    except Exception as e:  # never block the sign-up or the checkout
        log.warning("policy acceptance not recorded (%s, %s): %s", kind, ref[:24], type(e).__name__)
        return 0


def record_signup(email: str) -> int:
    return record(email, kind="signup", ref=f"signup:{email.strip().lower()}", tier="basic")


def record_checkout(user: str, url: str, plan: str) -> int:
    """For a new Stripe checkout (not the plan-change portal): the session id is in its URL."""
    m = _SESSION.search(url or "")
    return record(user, kind="subscription", ref=m.group(1), tier=plan) if m else 0
