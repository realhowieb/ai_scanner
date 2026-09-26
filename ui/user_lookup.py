"""P0-8 — look up only the signed-in user on each Scanner rerun.

The Scanner used to call `db.users.load_users()` on every rerun: a full read of
the users table (including every password hash) just to find one user's tier
for the legacy fallback in `_resolve_tier_state`. Tier Sync already does the
authoritative DB-first lookup, so the legacy map only needs this user's entry.

`load_user_map` returns the same shape `load_users()` did, restricted to one
user: `{username: {"name": ..., "tier": ...}}` (no password). Inactive or
unknown users map to `{}`, exactly as they were absent from `load_users()`;
`get_user_tier` then falls back to the local USERS_DB as before.
"""
from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


def _lookup(username: str) -> Dict[str, Dict[str, Any]]:
    key = (username or "").strip().lower()
    if not key:
        return {}
    from db.users import get_user_by_username

    rec = get_user_by_username(key)
    if not rec or rec.get("is_active") is False:
        return {}
    return {key: {"name": rec.get("full_name"), "tier": rec.get("tier") or "basic"}}


if st is not None:
    # Short TTL: a tier change (e.g. after checkout) shows within 30 s even if
    # Tier Sync's own lookup fails; cache_data hands each caller its own copy,
    # so Tier Sync's in-place update of the map can't leak between sessions.
    load_user_map = st.cache_data(ttl=30, show_spinner=False)(_lookup)
else:  # pragma: no cover
    load_user_map = _lookup


# ---- Run 71: session-level reuse of the resolved tier ---------------------------------------------
TIER_STATE_KEY = "_hsf_tier_state"
TIER_STATE_TTL_S = 30.0


def tier_state_is_fresh(cached: Any, username: str, *, now: float, session_tier: Any,
                        billing_return: bool) -> bool:
    """Reuse a cached tier state only for the same user, within the TTL, when no
    Stripe return is in progress and no other page changed the session's tier."""
    if not isinstance(cached, dict) or billing_return:
        return False
    if cached.get("user") != username or now - float(cached.get("at") or 0) >= TIER_STATE_TTL_S:
        return False
    state = cached.get("state") or {}
    if session_tier and str(session_tier).strip().lower() != str(state.get("tier_key") or "").strip().lower():
        return False
    return True


def session_tier_state(username: str, compute) -> Any:
    """Resolve the tier at most every TIER_STATE_TTL_S seconds per session
    (Tier Sync queries the DB); recompute immediately after checkout/portal
    returns or when another page updated the session tier. Never raises."""
    import time

    if st is None:
        return compute()
    try:
        billing_return = bool((st.query_params.get("checkout") or st.query_params.get("portal") or "").strip())
    except Exception:
        billing_return = True
    cached = st.session_state.get(TIER_STATE_KEY)
    now = time.time()
    if tier_state_is_fresh(cached, username, now=now, session_tier=st.session_state.get("tier_key"),
                           billing_return=billing_return):
        return cached["state"]
    state = compute()
    try:
        st.session_state[TIER_STATE_KEY] = {"user": username, "at": now, "state": state}
    except Exception:
        pass
    return state
