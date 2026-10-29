"""The plan an account actually gets: the highest of its subscription plan (users.tier,
written by billing sync), a free Pro trial from invites (api/referrals) and an active
Founding 100 membership (db/founding: Founder Pro -> Pro, Founder Elite -> Premium,
for five years from payment).

Founder access is never written into users.tier, so a subscription sync can't erase
it, and it ends by itself at expires_at. Lookups are cached per account for
CACHE_S, and skipped entirely until the first founding membership exists. Any storage
error falls back to the account's own plan (never more access than it paid for).
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("hsf_api.plan_access")

CACHE_S = 60
_cache: Dict[str, Tuple[float, Optional[str], Optional[dt.datetime]]] = {}
_any: Dict[str, Tuple[float, bool]] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        _any.clear()


def _rank(tier: Optional[str]) -> int:
    from domain.plans import TIER_ORDER

    return TIER_ORDER.get(str(tier or "basic").strip().lower(), 0)


def _any_founders(now: float) -> bool:
    with _lock:
        hit = _any.get("any")
    if hit and now < hit[0]:
        return hit[1]
    try:
        from db import founding

        found = founding.any_members()
    except Exception as e:
        log.debug("founding lookup unavailable: %s", type(e).__name__)
        found = False
    with _lock:
        _any["any"] = (now + CACHE_S, found)
    return found


def founder_access(user: str) -> Tuple[Optional[str], Optional[dt.datetime]]:
    """(founder tier, access end) for an active, unexpired membership, else (None, None)."""
    key = (user or "").strip().lower()
    if not key:
        return None, None
    now = time.monotonic()
    if not _any_founders(now):
        return None, None
    with _lock:
        hit = _cache.get(key)
    if hit and now < hit[0]:
        tier, until = hit[1], hit[2]
    else:
        try:
            from db import founding

            tier, until = founding.active_tier(key)
        except Exception as e:
            log.debug("founding lookup failed: %s", type(e).__name__)
            tier, until = None, None
        with _lock:
            if len(_cache) > 5000:
                _cache.clear()
            _cache[key] = (now + CACHE_S, tier, until)
    if tier and until and until > dt.datetime.now(dt.timezone.utc):
        return tier, until
    return None, None


def effective_tier(user: str, tier: Optional[str], is_admin: bool = False) -> Dict[str, Any]:
    """{"tier", "pro_trial_until", "founder_tier", "founder_until"} for an account."""
    from api import referrals
    from domain import founding as rules

    out: Dict[str, Any] = {"tier": str(tier or "basic").strip().lower() or "basic", "pro_trial_until": None,
                           "founder_tier": None, "founder_until": None}
    t, trial = referrals.effective_tier(user, tier, is_admin)
    out["tier"], out["pro_trial_until"] = t, trial
    if is_admin or out["tier"] == "admin":
        return out
    f_tier, f_until = founder_access(user)
    if f_tier:
        out["founder_tier"], out["founder_until"] = f_tier, f_until
        plan = rules.PLAN_FOR_TIER[f_tier]
        if _rank(plan) > _rank(out["tier"]):
            out["tier"], out["pro_trial_until"] = plan, None
    return out


def with_founder(user: str, plan: Optional[str]) -> str:
    """The higher of a stored plan and the account's founder plan, for background jobs
    (emails, alerts) that read users.tier directly. 'admin' stays 'admin'; any lookup
    error keeps the stored plan."""
    stored = str(plan or "basic").strip().lower() or "basic"
    if stored == "admin":
        return stored
    try:
        from domain import founding as rules

        f_tier, _ = founder_access(user)
        if f_tier:
            founder_plan = rules.PLAN_FOR_TIER[f_tier]
            if _rank(founder_plan) > _rank(stored):
                return founder_plan
    except Exception as e:
        log.debug("founder plan lookup failed: %s", type(e).__name__)
    return stored


def elite_alert_limit(founder_tier: Optional[str], base: int) -> int:
    """Founder Elite's alert cap (above Premium's); everyone else keeps `base`."""
    from domain import founding as rules

    return max(int(base), rules.ELITE_ALERT_LIMIT) if founder_tier == rules.FOUNDER_ELITE else int(base)


def founder_alert_limit(user: str, base: int) -> int:
    """`base` raised to Founder Elite's cap for an active Elite founder (background jobs)."""
    try:
        return elite_alert_limit(founder_access(user)[0], base)
    except Exception:
        return int(base)


def ai_daily_limit(user: str, base: int) -> int:
    """The daily AI allowance: Founder Elite gets ELITE_AI_MULTIPLIER times `base`."""
    from domain import founding as rules

    if base <= 0:
        return base
    try:
        if founder_access(user)[0] == rules.FOUNDER_ELITE:
            return int(base) * rules.ELITE_AI_MULTIPLIER
    except Exception:
        pass
    return int(base)


def apply_to_account(account: Dict[str, Any]) -> Dict[str, Any]:
    """The signed-in account with invite trials and founder access applied."""
    eff = effective_tier(str(account.get("username") or ""), account.get("tier"), bool(account.get("is_admin")))
    out = dict(account)
    out["tier"] = eff["tier"]
    if eff["pro_trial_until"]:
        out["pro_trial_until"] = eff["pro_trial_until"].isoformat(timespec="seconds")
    if eff["founder_tier"]:
        out["founder_tier"] = eff["founder_tier"]
        out["founder_until"] = eff["founder_until"].isoformat(timespec="seconds")
    return out
