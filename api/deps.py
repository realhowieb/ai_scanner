"""Shared pieces of the API routes: the signed-in account, plan checks and common responses.

Route modules live in api/routes/; api/main.py builds the app from them."""
from __future__ import annotations

import datetime as dt
import hmac
import logging
import time
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Path, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api import account as acct
from api import store, tokens, user_data
from api.settings import Settings

log = logging.getLogger("hsf_api")


_UNAUTHORIZED = "Invalid or expired token"
_BAD_LOGIN = "Email or password is incorrect."


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _token_pair(username: str, settings: Settings, client: Optional[str],
                account: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    account = account if account is not None else store.get_account(username)
    if not account or account.get("is_active") is False or not account.get("password"):
        raise HTTPException(401, _UNAUTHORIZED)
    refresh = tokens.new_refresh_token()
    store.save_refresh_token(tokens.hash_refresh_token(refresh), username, settings.refresh_ttl_s, client)
    return {"access_token": tokens.create_access_token(username, settings, password_hash=account["password"]), "token_type": "bearer",
            "expires_in": settings.access_ttl_s, "refresh_token": refresh}


# Declared as a security scheme so /docs shows an Authorize button (paste the
# access token only; the docs page adds "Bearer ").
_bearer = HTTPBearer(auto_error=False, description="Access token from POST /v1/auth/login")


def current_account(request: Request,
                    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> Dict[str, Any]:
    """The signed-in, active account for a `Authorization: Bearer <access token>` header."""
    token = (creds.credentials if creds and (creds.scheme or "").lower() == "bearer" else "").strip()
    if not token:
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    settings = _settings(request)
    claims = tokens.access_claims(token, settings)
    username = str(claims.get("sub") or "").strip().lower() if claims else None
    # Security state must be fresh on every server. A cached row would preserve
    # access after a password reset, deactivation, or permission change elsewhere.
    account = store.get_account(username) if username else None
    if not account or account.get("is_active") is False:
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    expected = tokens.credential_version(username, account.get("password") or "", settings)
    if not hmac.compare_digest(claims["cv"], expected):
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    from api import plan_access

    _note_activity(username)
    return plan_access.apply_to_account({k: v for k, v in account.items() if k != "password"})


def _note_activity(username: str) -> None:
    """Last-active time for the come-back email (db/user_activity.py: one write an hour at most)."""
    try:
        from db.user_activity import touch

        touch(username)
    except Exception:
        pass


# Compatibility cache for non-authorizing callers; current_account bypasses it.
# Plan, admin and active-flag changes show up within ACCOUNT_CACHE_S. The password
# hash is never cached: endpoints that check a password read the row fresh.
ACCOUNT_CACHE_S = 15
_account_cache: Dict[str, tuple] = {}  # username -> (monotonic expiry, row without password)


def _recent_account(username: str) -> Optional[Dict[str, Any]]:
    key = username.strip().lower()
    now = time.monotonic()
    hit = _account_cache.get(key)
    if hit and now < hit[0]:
        return dict(hit[1])
    account = store.get_account(key)
    if len(_account_cache) > 1000:
        _account_cache.clear()
    if account:
        safe = {k: v for k, v in account.items() if k != "password"}
        _account_cache[key] = (now + ACCOUNT_CACHE_S, safe)
        return dict(safe)
    _account_cache.pop(key, None)
    return None


def forget_account(username: str) -> None:
    """Drop the cached row after this process changes the account."""
    _account_cache.pop((username or "").strip().lower(), None)


def _fresh_account(account: Dict[str, Any]) -> Dict[str, Any]:
    """The full row (with the password hash) for endpoints that check a password."""
    fresh = store.get_account(_user(account))
    if not fresh or fresh.get("is_active") is False:
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    return fresh


def entitlements_for(account: Dict[str, Any]) -> Dict[str, Any]:
    from domain.plans import ALERT_LIMIT_BY_TIER, TIER_ORDER, compute_entitlements, has_min_tier

    is_admin = bool(account.get("is_admin"))
    tier = str(account.get("tier") or "basic").strip().lower()
    tier = "admin" if is_admin else (tier if tier in TIER_ORDER else "basic")
    from api.plan_access import elite_alert_limit
    from domain import founding as founding_rules

    founder = account.get("founder_tier")
    ents = compute_entitlements(tier_obj=tier, is_admin=is_admin, has_min_tier_fn=has_min_tier)
    if founder:  # early access: features switched on for founders before their plan gets them
        for name in founding_rules.early_access_features():
            if name in ents:
                ents[name] = True
    return {"tier": tier, "is_admin": is_admin, "entitlements": ents,
            "alert_limit": elite_alert_limit(founder, ALERT_LIMIT_BY_TIER.get(tier, 1))}


def _capabilities(ent: Dict[str, Any]) -> Dict[str, Any]:
    from api.alert_rules import capabilities

    return capabilities(ent, watchlist_max=user_data.MAX_WATCHLISTS,
                        tickers_per_request=user_data.MAX_TICKERS_PER_REQUEST)


def _aware(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    """Query datetimes without a zone are UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)


class _rule_errors:
    """Map rule validation to 422 and plan gates to 403 inside a route."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        from api import alert_rules

        if exc_type is not None and issubclass(exc_type, alert_rules.RuleForbidden):
            raise HTTPException(403, str(exc)) from exc
        if exc_type is not None and issubclass(exc_type, alert_rules.RuleError):
            raise HTTPException(422, str(exc)) from exc
        return False


TICKER = Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$", description="Ticker symbol, e.g. AAPL or BRK.B")
_AUTH = {401: {"description": "Not signed in"}}
_OWNED = {**_AUTH, 404: {"description": "Not found (or not yours)"}}


def _me_out(account: Dict[str, Any]) -> Dict[str, Any]:
    from domain.plan_labels import plan_label

    ent = entitlements_for(account)
    return {"email": str(account["username"]).strip().lower(),
            "name": account.get("full_name") or None,
            "plan": ent["tier"], "plan_label": plan_label(ent["tier"]),
            "is_admin": ent["is_admin"], "alert_limit": ent["alert_limit"],
            "email_verified": acct.is_verified(str(account["username"]).strip().lower()),
            "entitlements": ent["entitlements"],
            "pro_trial_until": account.get("pro_trial_until"),
            "billing_attention": acct.billing_attention(str(account["username"]), ent["tier"])}


def _user(account: Dict[str, Any]) -> str:
    return str(account["username"]).strip().lower()


def require_feature(account: Dict[str, Any], feature: str) -> Dict[str, Any]:
    """The account's entitlements, or 403 with the web's upgrade wording."""
    ent = entitlements_for(account)
    if not ent["entitlements"].get(feature):
        try:
            from domain.upgrades import upgrade_message

            msg = upgrade_message(feature)
        except Exception:
            msg = "Your plan doesn't include this feature."
        raise HTTPException(403, msg)
    return ent


def require_min_plan(account: Dict[str, Any], plan: str, message: str) -> Dict[str, Any]:
    from domain.plans import has_min_tier

    ent = entitlements_for(account)
    if not (ent["is_admin"] or has_min_tier(ent["tier"], plan)):
        raise HTTPException(403, message)
    return ent


def _track(params: Dict[str, str], event: str, **kwargs: Any) -> None:
    """Best-effort acquisition event for the web app (never raises, never blocks)."""
    try:
        from ui.acquisition import attribution_from_params, track_event

        params = {str(k)[:40]: str(v)[:200] for k, v in (params or {}).items()}
        track_event(event, attribution=attribution_from_params(params, params.get("referrer")), **kwargs)
    except Exception:
        log.debug("acquisition event %s not recorded", event, exc_info=True)


def require_admin(account: Dict[str, Any]) -> Dict[str, Any]:
    """Admins only (the same users.is_admin flag the Streamlit admin pages use), else 403."""
    if not account.get("is_admin"):
        raise HTTPException(403, "Research endpoints are internal (admin only).")
    return account
_ADMIN = {**_AUTH, 403: {"description": "Admins only"}}
