"""Admin console (admins only): accounts, health and maintenance tools for the web app's Admin page."""
from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

from fastapi import Depends, FastAPI, HTTPException

from api import models
from api.deps import _ADMIN, _user, current_account

_TAG = "admin (admins only)"
_EXTRA = {"x-internal": True}
T = TypeVar("T")


def _admin(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
    if not account.get("is_admin"):
        raise HTTPException(403, "Admins only.")
    return account


def _act(fn: Callable[[], T]) -> T:
    from api.admin import AdminError

    try:
        return fn()
    except AdminError as e:
        raise HTTPException(422, str(e)) from None


def register(app: FastAPI) -> None:
    from api import admin

    responses = {**_ADMIN, 422: {"description": "Refused (the detail says why)"}}

    def get(path: str, model: Any, summary: str):
        return app.get(path, response_model=model, tags=[_TAG], openapi_extra=_EXTRA, responses=responses, summary=summary)

    def post(path: str, model: Any, summary: str):
        return app.post(path, response_model=model, tags=[_TAG], openapi_extra=_EXTRA, responses=responses, summary=summary)

    @get("/v1/admin/users", models.AdminUsers, "Every account with plan, active, admin and verification status")
    def admin_users(_a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return {"users": admin.users()}

    @post("/v1/admin/users", models.Message, "Create an account (email username, password rules as sign-up)")
    def admin_create_user(body: models.AdminCreateUser, a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return _act(lambda: admin.create_user(_user(a), body.email, body.full_name, body.password, body.tier, body.active))

    @post("/v1/admin/users/update", models.Message, "Change an account's plan or active flag")
    def admin_update_user(body: models.AdminUpdateUser, a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return _act(lambda: admin.update_user(_user(a), body.username, body.tier, body.is_active))

    @post("/v1/admin/users/verify", models.Message, "Mark an account's email verified")
    def admin_verify_user(body: models.AdminUserTarget, a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return _act(lambda: admin.mark_verified(_user(a), body.username))

    @post("/v1/admin/users/resend-verification", models.Message, "Email the account a new verification link")
    def admin_resend_verification(body: models.AdminUserTarget, _a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return _act(lambda: admin.resend_verification(body.username))

    @post("/v1/admin/users/role", models.Message, "Grant or revoke the admin role (not your own)")
    def admin_set_role(body: models.AdminSetRole, a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return _act(lambda: admin.set_admin(_user(a), body.username, body.is_admin))

    @get("/v1/admin/audit", models.AdminAudit, "Recent admin actions, newest first")
    def admin_audit(_a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return {"events": admin.audit_log()}

    @get("/v1/admin/health", models.AdminHealth, "System health snapshot and email delivery")
    def admin_health(_a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        out = admin.system_health()
        try:
            out["email"] = admin.email_health()
        except Exception:
            out["email"] = None
        return out

    @get("/v1/admin/providers", models.AdminProviders, "Live checks: database, Alpaca, FMP, Finnhub, billing")
    def admin_providers(_a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return admin.providers()

    @get("/v1/admin/activity", models.AdminActivity, "Recent scan errors, sign-in attempts and AI usage (30 days)")
    def admin_activity(_a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return _act(lambda: {"scan_errors": admin.scan_errors(), "login_attempts": admin.login_attempts(),
                             "ai_usage": admin.ai_usage()})

    @post("/v1/admin/db-check", models.AdminDbCheck, "Database reachability and earnings-table hygiene")
    def admin_db_check(_a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return admin.db_check()

    @get("/v1/admin/earnings-refresh", models.AdminEarningsRefresh, "The last manual earnings refresh")
    def admin_earnings_status(_a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return admin.earnings_refresh_status()

    @post("/v1/admin/earnings-refresh", models.AdminEarningsRefresh,
          "Refresh the earnings calendar now (FMP then Finnhub, in the background)")
    def admin_earnings_refresh(a: Dict[str, Any] = Depends(_admin)) -> Dict[str, Any]:
        return _act(lambda: admin.start_earnings_refresh(_user(a)))
