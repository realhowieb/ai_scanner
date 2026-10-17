"""Invite a friend (db.referrals): share links, credit at sign-up, Pro trial when the
new account verifies its email, and the trial applied to the signed-in account.

    sign-up with ?ref=CODE  -> record_signup (credited, nothing granted yet)
    email verified          -> reward_verified: both people get REWARD_DAYS of Pro
    every request           -> effective_tier: a Starter account with a live grant is Pro

Rewards wait for email verification so a throwaway address can't earn anything, and
each account's trial is capped at CAP_DAYS ahead. Billing is untouched: a trial is
not a subscription, and the account is Starter again when it ends. Every call here
is best effort and never breaks sign-up, verification or sign-in.

Env: HSF_REFERRALS_ENABLED=0 turns the whole feature off (links, credit, trials),
HSF_REFERRAL_REWARD_DAYS (default 30; 0 = links still credit, no trial),
HSF_REFERRAL_CAP_DAYS (default 180).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

log = logging.getLogger("hsf_api.referrals")

GRANT_CACHE_S = 60
_grant_cache: Dict[str, tuple] = {}  # user -> (monotonic expiry, until datetime or None)
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("HSF_REFERRALS_ENABLED", "1").strip() != "0"


def reward_days() -> int:
    try:
        return max(0, min(365, int(os.environ.get("HSF_REFERRAL_REWARD_DAYS", "30"))))
    except ValueError:
        return 30


def cap_days() -> int:
    try:
        return max(1, min(730, int(os.environ.get("HSF_REFERRAL_CAP_DAYS", "180"))))
    except ValueError:
        return 180


def _forget(user: str) -> None:
    with _lock:
        _grant_cache.pop(user.strip().lower(), None)


def clear_cache() -> None:
    with _lock:
        _grant_cache.clear()


def grant_until(user: str) -> Optional[dt.datetime]:
    """The account's free-Pro end time, cached per process for GRANT_CACHE_S. None on
    any storage error (the account just keeps its own plan)."""
    key = user.strip().lower()
    if not key or not enabled():
        return None
    now = time.monotonic()
    with _lock:
        hit = _grant_cache.get(key)
    if hit and now < hit[0]:
        until = hit[1]
    else:
        try:
            from db import referrals as store

            until = store.grant_until(key)
        except Exception as e:
            log.debug("referral grant lookup failed: %s", type(e).__name__)
            until = None
        with _lock:
            if len(_grant_cache) > 5000:
                _grant_cache.clear()
            _grant_cache[key] = (now + GRANT_CACHE_S, until)
    return until if until and until > dt.datetime.now(dt.timezone.utc) else None


def effective_tier(user: str, tier: Optional[str], is_admin: bool = False) -> tuple:
    """(tier, trial end) for an account: Starter with a live grant reads as Pro."""
    t = str(tier or "basic").strip().lower()
    if is_admin or t not in ("", "basic"):
        return t, None
    until = grant_until(user)
    return ("pro", until) if until else (t or "basic", None)


def apply_to_account(account: Dict[str, Any]) -> Dict[str, Any]:
    """The signed-in account with any referral trial applied (tier, pro_trial_until)."""
    tier, until = effective_tier(str(account.get("username") or ""), account.get("tier"),
                                 bool(account.get("is_admin")))
    if until is None:
        return account
    return {**account, "tier": tier, "pro_trial_until": until.isoformat(timespec="seconds")}


def credit_signup(attribution: Optional[Dict[str, str]], email: str) -> bool:
    code = str((attribution or {}).get("ref") or "").strip().lower()
    if not code or not enabled():
        return False
    try:
        from db import referrals as store

        ok = store.record_signup(code, email)
    except Exception as e:
        log.warning(json.dumps({"event": "referral_credit_failed", "error": type(e).__name__}))
        return False
    if ok:
        log.info(json.dumps({"event": "referral_signup"}))
    return ok


def reward_on_verify(email: str) -> bool:
    if not enabled():
        return False
    try:
        from db import referrals as store

        referrer = store.reward_verified(email, days=reward_days(), cap_days=cap_days())
    except Exception as e:
        log.warning(json.dumps({"event": "referral_reward_failed", "error": type(e).__name__}))
        return False
    if referrer is None:
        return False
    _forget(email)
    _forget(referrer)
    log.info(json.dumps({"event": "referral_rewarded", "days": reward_days()}))
    return True


def summary(user: str) -> Dict[str, Any]:
    from db import referrals as store

    code = store.get_or_create_code(user)
    counts = store.stats(user)
    until = grant_until(user)
    return {"code": code, "path": f"/signup?ref={code}", "signed_up": counts["signed_up"],
            "verified": counts["verified"], "reward_days": reward_days(),
            "pro_trial_until": until.isoformat(timespec="seconds") if until else None}
