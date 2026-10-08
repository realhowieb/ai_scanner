"""HSF API service (P1-59). Run: uvicorn api.main:app

Endpoints (v1): GET /healthz · POST /v1/auth/login · POST /v1/auth/refresh ·
POST /v1/auth/logout · GET /v1/me · GET /v1/today · GET /v1/scans/latest ·
GET /v1/stocks/{ticker} · /v1/watchlists · /v1/alerts · sign-up, email
verification, password reset/change, email preferences, billing links. Full list in docs/API.md;
OpenAPI docs at /docs.
"""
from __future__ import annotations

import datetime as dt
import functools
import inspect
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from api import account as acct
from api import devices, models, ratelimit, store, tokens, user_data
from api.scans import json_safe
from api.settings import Settings, load_settings

log = logging.getLogger("hsf_api")
access_log = logging.getLogger("hsf_api.access")
if not access_log.handlers:  # uvicorn only configures its own loggers; one JSON line per request to stdout
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    access_log.addHandler(_h)
    access_log.setLevel(logging.INFO)
    access_log.propagate = False
_SAFE_RID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

_UNAUTHORIZED = "Invalid or expired token"
_BAD_LOGIN = "Email or password is incorrect."


class _ReleasingRoute(APIRoute):
    """Runs each endpoint, then ends the transaction on that worker thread's warm
    database connection. Several db.* helpers never call close(), so without this
    a worker thread sits idle in a transaction holding table locks until its next
    request (API acceptance run). Same thread as the endpoint, so it's the right
    connection."""

    def __init__(self, path: str, endpoint: Callable[..., Any], **kwargs: Any):
        if not inspect.iscoroutinefunction(endpoint):
            inner = endpoint

            @functools.wraps(inner)
            def endpoint(*args: Any, **kw: Any) -> Any:
                try:
                    return inner(*args, **kw)
                finally:
                    from db.engine import release_thread_connection

                    release_thread_connection()
        super().__init__(path, endpoint, **kwargs)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="HSFinest.AI API", version="1.0.0")
    app.router.route_class = _ReleasingRoute
    app.state.settings = settings
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                           allow_methods=["GET", "POST", "PATCH", "DELETE"],
                           allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
                           expose_headers=["X-Request-ID", "Retry-After"],  # 429/503 back-off
                           max_age=600)  # browsers cache the preflight for 10 minutes

    @app.middleware("http")
    async def _request_id(request: Request, call_next):
        """X-Request-ID on every response (a caller's own id is kept when it looks
        safe) and one JSON access-log line: method, route template, status, time.
        Never logs headers, bodies, query strings or tokens."""
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _SAFE_RID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            access_log.info(json.dumps({
                "request_id": rid, "method": request.method,
                "route": getattr(route, "path", None) or "unmatched", "status": status,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}))
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(store.DatabaseUnavailable)
    def _db_down(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": "Service temporarily unavailable. Try again shortly."},
                            status_code=503, headers={"Retry-After": "30"})

    try:  # connection drops mid-query are an outage too, not a server bug
        import psycopg

        app.add_exception_handler(psycopg.OperationalError, _db_down)
    except ImportError:  # pragma: no cover
        pass

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/healthz", response_model=models.Health)
    def healthz() -> Dict[str, bool]:
        """Liveness only: no database call (for the Render health check)."""
        return {"ok": True}

    @app.exception_handler(user_data.NotFound)
    def _not_found(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": f"No such {exc.args[0] if exc.args else 'item'}."}, status_code=404)

    @app.exception_handler(user_data.LimitReached)
    def _limit(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(user_data.Conflict)
    def _conflict(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    from api import custom_scans, scan_jobs

    @app.exception_handler(custom_scans.PlanError)
    def _plan_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(scan_jobs.ScanInProgress)
    def _scan_in_progress(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": "You already have a scan running. Check it with GET /v1/scans/{scan_id}.",
                             "scan_id": getattr(exc, "scan_id", None)}, status_code=409)

    @app.exception_handler(scan_jobs.ScanBusy)
    def _scan_busy(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": "Scans are busy right now. Try again in a minute."},
                            status_code=503, headers={"Retry-After": "60"})

    from api import ai as ai_mod

    @app.exception_handler(ai_mod.AIUnavailable)
    def _ai_unavailable(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503, headers={"Retry-After": "300"})

    @app.exception_handler(ai_mod.AILimit)
    def _ai_limit(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=429)

    @app.exception_handler(ai_mod.AIFailed)
    def _ai_failed(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    from api import trading as trading_mod

    @app.exception_handler(trading_mod.PaperUnavailable)
    def _paper_unavailable(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(trading_mod.PaperRejected)
    def _paper_rejected(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(trading_mod.TradeClosed)
    def _trade_closed(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(devices.InvalidDevice)
    def _bad_device(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(acct.AccountError)
    def _account_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=getattr(exc, "status", 400))

    @app.exception_handler(acct.BillingUnavailable)
    def _billing_down(_request: Request, exc: Exception) -> JSONResponse:
        log.warning("billing service call failed: %s", str(exc)[:120])
        return JSONResponse({"detail": "Billing is temporarily unavailable. Please try again in a minute."},
                            status_code=502)

    @app.get("/readyz", response_model=models.Ready,
             responses={503: {"description": "Database unavailable, or (strict=true) a scheduled scan was missed"}})
    def readyz(strict: bool = Query(False, description=(
            "Answer 503 when a scheduled full-market scan was missed, so an outside uptime "
            "monitor alerts on stale data as well as on a database outage"))) -> Any:
        """Readiness: the database answers, plus the latest market scan's age for
        freshness monitoring. /healthz stays the liveness check."""
        store.ping()
        now = dt.datetime.now(dt.timezone.utc)
        latest, age, fresh = None, None, {"stale": None, "expected_scan_at": None}
        try:
            from api.today import market_runs, scan_freshness

            runs = market_runs()
            created = runs[0]["created_at"] if runs else None
            if created is not None:
                latest = json_safe(created)
                age = round((now - created).total_seconds() / 60.0, 1)
            fresh = scan_freshness(created, now)
        except store.DatabaseUnavailable:
            raise
        except Exception:  # scan freshness is informational
            pass
        body = {"ok": True, "database": "ok", "latest_scan_at": latest, "scan_age_minutes": age, **fresh}
        if strict and fresh["stale"]:
            return JSONResponse({**body, "ok": False}, status_code=503)
        return body

    _routes(app)
    _data_routes(app)
    _account_routes(app)
    _device_routes(app)
    _scan_routes(app)
    _history_routes(app)
    _market_routes(app)
    _ai_routes(app)
    _trading_routes(app)
    _delete_account_route(app)
    return app


class LoginBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    client: Optional[str] = Field(default=None, max_length=80)


class RefreshBody(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class LogoutBody(RefreshBody):
    push_token: Optional[str] = Field(default=None, max_length=600,
                                      description="This device's push token, so it stops getting this account's pushes")


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _token_pair(username: str, settings: Settings, client: Optional[str]) -> Dict[str, Any]:
    refresh = tokens.new_refresh_token()
    store.save_refresh_token(tokens.hash_refresh_token(refresh), username, settings.refresh_ttl_s, client)
    return {"access_token": tokens.create_access_token(username, settings), "token_type": "bearer",
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
    username = tokens.verify_access_token(token, _settings(request))
    account = store.get_account(username) if username else None
    if not account or account.get("is_active") is False:
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    return account


def entitlements_for(account: Dict[str, Any]) -> Dict[str, Any]:
    from auth.tiering import TIER_ORDER, has_min_tier
    from ui.app_session import ALERT_LIMIT_BY_TIER, compute_entitlements

    is_admin = bool(account.get("is_admin"))
    tier = str(account.get("tier") or "basic").strip().lower()
    tier = "admin" if is_admin else (tier if tier in TIER_ORDER else "basic")
    return {"tier": tier, "is_admin": is_admin,
            "entitlements": compute_entitlements(tier_obj=tier, is_admin=is_admin, has_min_tier_fn=has_min_tier),
            "alert_limit": ALERT_LIMIT_BY_TIER.get(tier, 1)}


def _routes(app: FastAPI) -> None:
    @app.post("/v1/auth/login", response_model=models.TokenPair,
              responses={401: {"description": "Wrong email or password"}, 429: {"description": "Rate limited"},
                         503: {"description": "Database unavailable"}})
    def login(body: LoginBody, request: Request, _limited: None = Depends(ratelimit.limit("login"))) -> Dict[str, Any]:
        from db.users import is_login_rate_limited, record_login_attempt

        settings = _settings(request)
        email = body.email.strip().lower()
        if is_login_rate_limited(email):
            raise HTTPException(429, "Too many failed sign-in attempts. Try again in a few minutes.")
        account = store.get_account(email)
        if account is None:
            store.burn_password_check()
            ok = False
        else:
            ok = store.check_password(account, body.password) and account.get("is_active") is not False
        record_login_attempt(email, success=ok, failure_reason=None if ok else "api_login_failed")
        if not ok:
            raise HTTPException(401, _BAD_LOGIN)
        return _token_pair(str(account["username"]).strip().lower(), settings, body.client)

    @app.post("/v1/auth/refresh", response_model=models.TokenPair,
              responses={401: {"description": "Unknown, expired or reused refresh token"}})
    def refresh(body: RefreshBody, request: Request) -> Dict[str, Any]:
        settings = _settings(request)
        status, username = store.use_refresh_token(tokens.hash_refresh_token(body.refresh_token))
        if status == "reused":
            log.warning("refresh token reuse: all sessions revoked for one account")
            if username:
                devices.remove_all(username)  # P1-64: no pushes to a possibly stolen session
        elif status == "grace":
            log.info("refresh token retried within the grace window")
        account = store.get_account(username) if status in ("ok", "grace") and username else None
        if not account or account.get("is_active") is False:
            raise HTTPException(401, _UNAUTHORIZED)
        return _token_pair(str(account["username"]).strip().lower(), settings, None)

    @app.post("/v1/auth/logout", status_code=204)
    def logout(body: LogoutBody) -> None:
        username = store.revoke_refresh_token(tokens.hash_refresh_token(body.refresh_token))
        if username and body.push_token:
            devices.remove_token(username, body.push_token.strip())

    @app.get("/v1/me", response_model=models.Me, responses={401: {"description": "Not signed in"}})
    def me(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from ui.plan_labels import plan_label

        ent = entitlements_for(account)
        return {"email": str(account["username"]).strip().lower(),
                "name": account.get("full_name") or None,
                "plan": ent["tier"], "plan_label": plan_label(ent["tier"]),
                "is_admin": ent["is_admin"], "alert_limit": ent["alert_limit"],
                "email_verified": acct.is_verified(str(account["username"]).strip().lower()),
                "entitlements": ent["entitlements"]}

    @app.get("/v1/today", response_model=models.Today, responses={401: {"description": "Not signed in"}})
    def today(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api.today import build_today

        return build_today(dt.datetime.now(dt.timezone.utc), entitlements_for(account)["entitlements"])


TICKER = Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$", description="Ticker symbol, e.g. AAPL or BRK.B")
_AUTH = {401: {"description": "Not signed in"}}
_OWNED = {**_AUTH, 404: {"description": "Not found (or not yours)"}}


class WatchlistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    make_default: bool = False


class WatchlistUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    make_default: bool = Field(default=False, description="true makes this the default watchlist")


class TickersBody(BaseModel):
    tickers: List[str] = Field(min_length=1, max_length=user_data.MAX_TICKERS_PER_REQUEST)


class NoteBody(BaseModel):
    note: Optional[str] = Field(default=None, max_length=user_data.MAX_NOTE_LEN)


class AlertCreate(BaseModel):
    type: str = Field(description="breakout, watchlist, price, move, rvol, ema_cross or ewo_cross")
    ticker: Optional[str] = Field(default=None, max_length=12)
    threshold: Optional[float] = None
    direction: Optional[str] = Field(default=None, max_length=12)
    watchlist_only: bool = False


class AlertUpdate(BaseModel):
    enabled: bool


def _user(account: Dict[str, Any]) -> str:
    return str(account["username"]).strip().lower()


def _data_routes(app: FastAPI) -> None:
    # ---- step 5: scans and stock detail ----
    @app.get("/v1/market/tape", response_model=models.Tape, responses=_AUTH, summary="Price strip")
    def market_tape(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA and TSLA: last price and change vs the previous close."""
        from api.today import tape_quotes

        return {"quotes": tape_quotes()}

    @app.get("/v1/scans/latest", response_model=models.LatestScan, responses=_AUTH)
    def scans_latest(account: Dict[str, Any] = Depends(current_account),
                     limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0, le=10_000),
                     min_score: int = Query(0, ge=0, le=100),
                     signal: Optional[str] = Query(None, pattern="^(golden_cross|breakout|prebreakout|gapper|gainer)$",
                                                   description="Only setups with this signal")) -> Dict[str, Any]:
        """The latest market scan's HSF setups, ranked as in the Scanner, up to the plan's row cap."""
        from api.scans import latest_scan

        ent = entitlements_for(account)
        if signal == "prebreakout" and not ent["entitlements"].get("can_early_breakout"):
            raise HTTPException(403, "PreBreakout is a Premium feature.")
        return latest_scan(ent["entitlements"], ent["tier"], limit=limit, offset=offset,
                           min_score=min_score, signal=signal)

    @app.get("/v1/stocks/{ticker}", response_model=models.StockDetail, responses=_AUTH)
    def stock(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stock Intelligence for one ticker: score, signals, reasons and risks, lifecycle,
        daily bars, and your watchlists and alerts on it."""
        from api.scans import daily_bars, stock_detail

        t = ticker.strip().upper()
        user = _user(account)
        out = stock_detail(t, entitlements_for(account)["entitlements"])
        try:
            bars = daily_bars(t)
        except Exception:  # chart data is optional; the page still renders
            bars = {"bars": [], "as_of": None}
        out.update({"bars": bars["bars"], "bars_as_of": bars["as_of"],
                    "watchlists": user_data.watchlists_with(user, t),
                    "alerts": json_safe(user_data.alerts_for(user, t))})
        return out

    # ---- step 6: watchlists ----
    @app.get("/v1/watchlists", response_model=List[models.Watchlist], responses=_AUTH)
    def watchlists(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        return user_data.list_watchlists(_user(account))

    @app.post("/v1/watchlists", response_model=models.WatchlistDetail, status_code=201,
              responses={**_AUTH, 403: {"description": "Watchlist limit reached"},
                         409: {"description": "A watchlist with that name exists"}})
    def watchlist_create(body: WatchlistCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.create_watchlist(_user(account), body.name, body.make_default))

    @app.get("/v1/watchlists/{watchlist_id}", response_model=models.WatchlistDetail, responses=_OWNED)
    def watchlist_get(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.get_watchlist(_user(account), watchlist_id))

    @app.patch("/v1/watchlists/{watchlist_id}", response_model=models.WatchlistDetail,
               responses={**_OWNED, 409: {"description": "A watchlist with that name exists"}})
    def watchlist_update(watchlist_id: int, body: WatchlistUpdate,
                         account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.update_watchlist(_user(account), watchlist_id, name=body.name,
                                                    make_default=body.make_default))

    @app.delete("/v1/watchlists/{watchlist_id}", status_code=204, responses=_OWNED)
    def watchlist_delete(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.delete_watchlist(_user(account), watchlist_id)

    @app.post("/v1/watchlists/{watchlist_id}/tickers", response_model=models.TickersResult, responses=_OWNED)
    def watchlist_add(watchlist_id: int, body: TickersBody,
                      account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return user_data.add_tickers(_user(account), watchlist_id, body.tickers)

    @app.delete("/v1/watchlists/{watchlist_id}/tickers/{ticker}", status_code=204, responses=_OWNED)
    def watchlist_remove(watchlist_id: int, ticker: str = TICKER,
                         account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.remove_ticker(_user(account), watchlist_id, ticker.upper())

    @app.patch("/v1/watchlists/{watchlist_id}/tickers/{ticker}", status_code=204, responses=_OWNED)
    def watchlist_note(watchlist_id: int, body: NoteBody, ticker: str = TICKER,
                       account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.set_note(_user(account), watchlist_id, ticker.upper(), body.note)

    # ---- step 6: alerts ----
    @app.get("/v1/alerts", response_model=models.Alerts, responses=_AUTH)
    def alerts(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        ent = entitlements_for(account)
        items = user_data.list_alerts(_user(account))
        return json_safe({"limit": ent["alert_limit"], "used": len(items),
                          "email_enabled": bool(ent["entitlements"].get("can_email_alerts")), "alerts": items})

    @app.post("/v1/alerts", response_model=models.Alert, status_code=201,
              responses={**_AUTH, 403: {"description": "Plan alert limit reached"},
                         409: {"description": "You already have this alert"},
                         422: {"description": "Invalid type, ticker, threshold or direction"}})
    def alert_create(body: AlertCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Same types and rules as the web app. Free 1 alert, Pro 5, Premium 25."""
        ent = entitlements_for(account)
        try:
            created = user_data.create_alert(_user(account), ent["alert_limit"], body.type, ticker=body.ticker,
                                             threshold=body.threshold, direction=body.direction,
                                             watchlist_only=body.watchlist_only)
        except user_data.Conflict:
            raise
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return json_safe(created)

    @app.patch("/v1/alerts/{alert_id}", response_model=models.Alert, responses=_OWNED)
    def alert_update(alert_id: int, body: AlertUpdate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.set_alert_enabled(_user(account), alert_id, body.enabled))

    @app.delete("/v1/alerts/{alert_id}", status_code=204, responses=_OWNED)
    def alert_delete(alert_id: int, account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.delete_alert(_user(account), alert_id)

    @app.get("/v1/alerts/types", response_model=List[models.AlertType], responses=_AUTH)
    def alert_types(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        """The alert types and their input rules (what POST /v1/alerts validates), for building forms."""
        return user_data.alert_types()

    @app.get("/v1/alerts/events", response_model=List[models.AlertEvent], responses=_AUTH)
    def alert_events(account: Dict[str, Any] = Depends(current_account),
                     limit: int = Query(20, ge=1, le=100)) -> List[Dict[str, Any]]:
        """Your most recent fired alerts, newest first."""
        return json_safe(user_data.alert_events(_user(account), limit))


class SignupBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    username: str = Field(min_length=1, max_length=40, description="Shown in the app; can also be used to sign in on the web")
    accept_terms: bool = Field(description="The usage agreement checkbox on the web sign-up form")
    client: Optional[str] = Field(default=None, max_length=80)


class TokenBody(BaseModel):
    token: str = Field(min_length=10, max_length=128)


class EmailBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class ResetConfirmBody(BaseModel):
    token: str = Field(min_length=10, max_length=128)
    new_password: str = Field(min_length=1, max_length=256)


class PasswordChangeBody(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class EmailPrefsUpdate(BaseModel):
    digest: Optional[bool] = None
    evening: Optional[bool] = None
    alerts: Optional[bool] = None


class CheckoutBody(BaseModel):
    plan: str = Field(pattern="^(pro|premium)$")
    interval: str = Field(default="month", pattern="^(month|year)$")


class PortalBody(BaseModel):
    flow: Optional[str] = Field(default=None, pattern="^cancel$", description="'cancel' opens the cancellation screen")


_RESET_SENT = ("If that email is registered, a reset link has been sent. "
               "Check your inbox (and spam folder).")


def _account_routes(app: FastAPI) -> None:
    @app.post("/v1/auth/signup", response_model=models.SignupResult, status_code=201,
              responses={400: {"description": "Invalid input or password rule"}, 409: {"description": "Email or username taken"},
                         429: {"description": "Too many sign-ups from this address"}})
    def signup(body: SignupBody, request: Request, _l: None = Depends(ratelimit.limit("signup"))) -> Dict[str, Any]:
        """Create a Free account (same rules as the web form) and sign in. A verification
        email is sent; verifying is needed to upgrade and for alert emails."""
        res = acct.signup(body.email, body.password, body.username, body.accept_terms)
        return {**_token_pair(res["email"], _settings(request), body.client),
                "email": res["email"], "verification_sent": res["verification_sent"]}

    @app.post("/v1/auth/verify-email", response_model=models.Message,
              responses={400: {"description": "Invalid or expired link"}})
    def verify_email(body: TokenBody, _l: None = Depends(ratelimit.limit("verify"))) -> Dict[str, Any]:
        """Confirm an email address with the token from the verification link."""
        acct.verify_email(body.token)
        return {"message": "Email verified."}

    @app.post("/v1/me/verify-email", response_model=models.Message, responses={401: {"description": "Not signed in"}})
    def resend_verification(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Send a new verification link to the signed-in account's email (3 per hour)."""
        user = _user(account)
        if acct.is_verified(user):
            return {"message": "Your email is already verified."}
        ratelimit.check("verify_resend", user)
        if not acct.send_verification(user):
            raise HTTPException(503, "We couldn't send the email right now. Try again later.")
        return {"message": "Verification email sent. Check your inbox (and spam)."}

    @app.post("/v1/auth/password-reset", response_model=models.Message, status_code=202,
              responses={429: {"description": "Too many requests from this address"}})
    def password_reset(body: EmailBody, _l: None = Depends(ratelimit.limit("password_reset"))) -> Dict[str, Any]:
        """Email a reset link. Same answer whether or not the account exists."""
        acct.request_password_reset(body.email)
        return {"message": _RESET_SENT}

    @app.post("/v1/auth/password-reset/confirm", response_model=models.Message,
              responses={400: {"description": "Invalid link or password rule"}})
    def password_reset_confirm(body: ResetConfirmBody, _l: None = Depends(ratelimit.limit("verify"))) -> Dict[str, Any]:
        """Set a new password with the token from the reset link; signs the account out everywhere."""
        acct.confirm_password_reset(body.token, body.new_password)
        return {"message": "Password updated. You've been signed out on all devices; sign in with the new password."}

    @app.post("/v1/me/password", response_model=models.TokenPair,
              responses={400: {"description": "Wrong current password or password rule"}, 401: {"description": "Not signed in"}})
    def change_password(body: PasswordChangeBody, request: Request,
                        account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Change the password. Every other session (web and app) is signed out; this
        device gets a new token pair."""
        ratelimit.check("login", ratelimit.client_ip(request))
        acct.change_password(account, body.current_password, body.new_password)
        return _token_pair(_user(account), _settings(request), None)

    @app.get("/v1/me/email-preferences", response_model=models.EmailPrefs, responses=_AUTH)
    def email_prefs(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return acct.get_email_prefs(_user(account))

    @app.patch("/v1/me/email-preferences", response_model=models.EmailPrefs, responses=_AUTH)
    def email_prefs_update(body: EmailPrefsUpdate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return acct.set_email_prefs(_user(account), body.model_dump(exclude_none=True))

    @app.post("/v1/billing/checkout", response_model=models.BillingLink,
              responses={**_AUTH, 403: {"description": "Verify your email first"},
                         502: {"description": "Billing service unavailable"}})
    def billing_checkout(body: CheckoutBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stripe checkout for Pro or Premium (monthly or yearly when enabled). Existing
        subscribers get Stripe's plan-change screen instead (mode=portal). Open the URL in a browser."""
        return acct.checkout_url(_user(account), body.plan, body.interval)

    @app.post("/v1/billing/portal", response_model=models.BillingLink,
              responses={**_AUTH, 404: {"description": "No subscription yet"}, 502: {"description": "Billing service unavailable"}})
    def billing_portal(body: PortalBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Stripe Customer Portal: payment method, invoices, plan, cancellation (flow=cancel)."""
        return acct.portal_url(_user(account), body.flow)


class DeviceBody(BaseModel):
    push_token: str = Field(min_length=10, max_length=600, description="Token from APNs, FCM or Expo")
    platform: Literal["ios", "android"]
    provider: Optional[Literal["apns", "fcm", "expo"]] = Field(
        default=None, description="Default: expo for Expo tokens, apns on iOS, fcm on Android")
    device_name: Optional[str] = Field(default=None, max_length=80, description="Shown in the app's device list")
    app_version: Optional[str] = Field(default=None, max_length=40)


def _device_out(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    for k in ("created_at", "last_seen_at"):
        if isinstance(out.get(k), (dt.datetime, dt.date)):
            out[k] = json_safe(out[k])
    return out


def _device_routes(app: FastAPI) -> None:
    """P1-64: push devices. Register at every app start and after each sign-in,
    sign-up or password change (all sessions signed out also removes devices);
    re-registering the same token is a no-op apart from last_seen_at."""

    @app.post("/v1/me/devices", response_model=models.Device,
              responses={**_AUTH, 400: {"description": "Not a push token for that provider"}})
    def device_register(body: DeviceBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        token = body.push_token.strip()
        provider = devices.resolve_provider(token, body.platform, body.provider)
        name = (body.device_name or "").strip() or None
        version = (body.app_version or "").strip() or None
        return _device_out(devices.register(_user(account), token, provider, body.platform, name, version))

    @app.get("/v1/me/devices", response_model=List[models.Device], responses=_AUTH)
    def device_list(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        return [_device_out(d) for d in devices.list_devices(_user(account))]

    @app.delete("/v1/me/devices/{device_id}", status_code=204, responses=_OWNED)
    def device_remove(device_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        if not devices.remove(_user(account), device_id):
            raise user_data.NotFound("device")


class ScanFilters(BaseModel):
    """Same filters and ranges as the web's Custom Scan page (defaults match it)."""
    min_price: float = Field(1.0, ge=0.5, le=500)
    max_price: float = Field(1000.0, ge=1, le=5000)
    min_dollar_vol: float = Field(5_000_000.0, ge=0, le=1e12, description="20-day average dollar volume floor")
    min_gap: float = Field(1.0, ge=0, le=20, description="Used with apply_gap_filter (Pro+)")
    apply_gap_filter: bool = Field(False, description="Pro+")
    unusual_volume: bool = Field(False, description="Pro+")
    session: Literal["regular", "premarket", "afterhours"] = Field("regular", description="premarket / afterhours: Pro+")
    profile: Literal["regular", "aggressive", "conservative"] = "regular"
    top_n: Optional[int] = Field(None, ge=5, le=10_000,
                                 description="Rows to return; at most the plan's cap (Free 25, Pro 100, Premium 200). "
                                             "Default min(25, cap)")
    max_nasdaq: Optional[int] = Field(None, ge=100, le=100_000,
                                      description="NASDAQ ticker cap, Pro up to 4000 (default 1200); "
                                                  "ignored for Premium (full list)")
    max_combo: Optional[int] = Field(None, ge=100, le=100_000,
                                     description="Combo ticker cap, Pro up to 6000 (default 1000); "
                                                 "ignored for Premium (full list)")


class ScanCreate(BaseModel):
    universe: Literal["sp500", "nasdaq", "combo", "us_market", "watchlist", "ticker"] = Field(
        description="sp500 (every plan) · nasdaq, combo (Pro+) · us_market (Premium+) · "
                    "watchlist (one of yours) · ticker (one symbol)")
    ticker: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$", description="For universe=ticker")
    watchlist_id: Optional[int] = Field(None, ge=1, description="For universe=watchlist")
    score_all: bool = Field(False, description="Watchlist only: score every symbol, ignoring the screens (as on the web)")
    filters: ScanFilters = Field(default_factory=ScanFilters)


def _job_out(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {**row, "scan_id": row["id"]}
    return json_safe(out)


def _scan_routes(app: FastAPI) -> None:
    from api import custom_scans, scan_jobs

    _SCAN_ERRORS = {**_AUTH, 403: {"description": "Not in your plan (universe, rows, session or filter)"},
                    404: {"description": "Watchlist not found (or not yours)"},
                    409: {"description": "You already have a scan queued or running (body has scan_id)"},
                    422: {"description": "Invalid request"}, 429: {"description": "Too many scans this hour"},
                    503: {"description": "Scans busy or database unavailable (Retry-After)"}}

    @app.post("/v1/scans", response_model=models.ScanJob, status_code=202, responses=_SCAN_ERRORS,
              summary="Start a custom scan")
    def scan_create(body: ScanCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Queue a custom scan (the web's Custom Scan page). Plan rules are checked here; the
        scan runs in the background — poll GET /v1/scans/{scan_id} (every 2-5 s) until
        status is complete or failed. One scan per account at a time; 30 per hour."""
        user = _user(account)
        ratelimit.check("scan", user)
        try:
            params = custom_scans.plan_or_raise(body.model_dump(), entitlements_for(account))
        except custom_scans.PlanError:
            raise
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if params["universe"] == "watchlist":
            user_data.get_watchlist(user, int(params["watchlist_id"]))   # 404 when not yours
        job = scan_jobs.create_job(user, params["universe"], params)
        scan_jobs.submit(job["id"], lambda report: custom_scans.run_scan(params, user, report))
        return _job_out(job)

    @app.get("/v1/scans", response_model=List[models.ScanJob], responses=_AUTH, summary="Your recent custom scans")
    def scan_list(account: Dict[str, Any] = Depends(current_account),
                  limit: int = Query(10, ge=1, le=50)) -> List[Dict[str, Any]]:
        """Newest first, without results (fetch one by id for its rows). Kept 7 days."""
        return [_job_out(j) for j in scan_jobs.list_jobs(_user(account), limit)]

    @app.get("/v1/scans/{scan_id}", response_model=models.ScanJob, responses=_OWNED, summary="A custom scan's status and results")
    def scan_get(scan_id: str = Path(pattern="^[0-9a-f]{32}$"),
                 account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        job = scan_jobs.get_job(_user(account), scan_id)
        if job is None:
            raise user_data.NotFound("scan")
        return _job_out(job)

    @app.delete("/v1/scans/{scan_id}", response_model=models.ScanJob, responses=_OWNED, summary="Cancel a custom scan")
    def scan_cancel(scan_id: str = Path(pattern="^[0-9a-f]{32}$"),
                    account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Cancel your queued or running scan: it reads `failed` with error "Cancelled." and you
        can start another at once. A running scan stops at its next progress step. A scan that
        already finished is returned unchanged."""
        job = scan_jobs.cancel_job(_user(account), scan_id)
        if job is None:
            raise user_data.NotFound("scan")
        return _job_out(job)


def require_feature(account: Dict[str, Any], feature: str) -> Dict[str, Any]:
    """The account's entitlements, or 403 with the web's upgrade wording."""
    ent = entitlements_for(account)
    if not ent["entitlements"].get(feature):
        try:
            from ui.pricing import upgrade_message

            msg = upgrade_message(feature)
        except Exception:
            msg = "Your plan doesn't include this feature."
        raise HTTPException(403, msg)
    return ent


def _history_routes(app: FastAPI) -> None:
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


def _market_routes(app: FastAPI) -> None:
    from api import market

    @app.get("/v1/earnings", response_model=List[models.EarningsItem], responses={**_AUTH, 403: {"description": "Pro feature"}},
             summary="Upcoming earnings (Pro)")
    def earnings(account: Dict[str, Any] = Depends(current_account),
                 days: int = Query(7, ge=0, le=market.MAX_EARNINGS_DAYS),
                 tickers: Optional[str] = Query(None, max_length=4000,
                                                description="Comma-separated tickers to keep (e.g. a scan's rows)")) -> List[Dict[str, Any]]:
        """The web's earnings calendar: earnings in the next `days` days, soonest first."""
        require_feature(account, "can_earnings")
        wanted = [t for t in (tickers or "").split(",") if t.strip()][:500]
        return market.earnings(days, wanted)

    _DT = {**_AUTH, 403: {"description": "Pro feature"}}

    @app.get("/v1/day-trader", response_model=models.DayTrader, responses=_DT, summary="Live Day Trader monitor (Pro)")
    def day_trader(account: Dict[str, Any] = Depends(current_account),
                   source: Literal["watchlist", "movers", "movers_sp500", "movers_nasdaq", "premarket", "postmarket",
                                   "scan_picks", "megacaps", "custom"] = "watchlist",
                   symbols: Optional[str] = Query(None, max_length=2000, description="source=custom: comma-separated"),
                   watchlist_id: Optional[int] = Query(None, ge=1, description="source=watchlist; default: your default list")
                   ) -> Dict[str, Any]:
        """The web's Day Trader table: live quotes, gap, VWAP, relative volume and the day-trade
        score for a symbol source. Quotes are shared for 30 s and movers screens for 2 min;
        poll every 30-60 s while the market is open."""
        require_feature(account, "can_day_trader")
        watch: List[str] = []
        if source == "watchlist":
            user = _user(account)
            wid = watchlist_id or next((w["id"] for w in user_data.list_watchlists(user) if w["is_default"]), None)
            if wid is not None:
                watch = [i["ticker"] for i in user_data.get_watchlist(user, int(wid))["items"]]   # 404 if not yours
        return json_safe(market.day_trader(source, (symbols or "").split(","), watch))

    @app.get("/v1/day-trader/stair-steppers", response_model=models.StairSteppers, responses=_DT,
             summary="Smooth 1-minute trends (Pro)")
    def stair_steppers(account: Dict[str, Any] = Depends(current_account),
                       symbols: str = Query(..., max_length=2000, description="Comma-separated; the first 40 are checked"),
                       window: int = Query(45, description="Bars fitted: 10, 15, 20, 30, 45 or 60"),
                       direction: Literal["up", "down", "either"] = "up",
                       r2_min: float = Query(0.8, ge=0.5, le=0.99),
                       max_pullback: float = Query(1.0, ge=0.1, le=5.0),
                       min_trend: float = Query(0.5, ge=0.0, le=20.0)) -> Dict[str, Any]:
        """The web's Stair-steppers check: symbols moving in a tight, straight line on the
        1-minute chart. Descriptive only; not a prediction."""
        require_feature(account, "can_day_trader")
        from analytics.stair_step import WINDOW_OPTIONS

        if window not in WINDOW_OPTIONS:
            raise HTTPException(422, f"window must be one of {list(WINDOW_OPTIONS)}.")
        return json_safe(market.stair_steppers(symbols.split(","), window=window, direction=direction, r2_min=r2_min,
                                               max_pullback_pct=max_pullback, min_trend_pct_per_hour=min_trend))

    @app.get("/v1/brief", response_model=models.Brief, responses=_AUTH, summary="Market Brief")
    def brief(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """The web's Market Brief (same builder as the morning email): market backdrop, top
        opportunities with movement, gappers, movers, setups and catalysts. Cached 5 minutes.
        PreBreakout picks and model fields are Premium. AI narrative: /v1/ai (Premium);
        historical scorecard: /v1/track-record (Pro); your alerts: /v1/alerts/events."""
        return json_safe(market.brief(entitlements_for(account)["entitlements"]))


class AISummaryBody(BaseModel):
    run_id: Optional[int] = Field(None, ge=1, description="One of your saved scans (GET /v1/runs); default: the latest market scan")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AIChatBody(AISummaryBody):
    messages: List[ChatTurn] = Field(min_length=1, max_length=16,
                                     description="The conversation so far, oldest first, ending with the new question")


def _ai_routes(app: FastAPI) -> None:
    from api import ai

    _AI = {**_AUTH, 403: {"description": "Premium feature"}, 404: {"description": "Scan not found (or not yours)"},
           429: {"description": "Daily AI limit or hourly request limit reached"},
           502: {"description": "AI call failed"}, 503: {"description": "AI unavailable"}}

    def _premium(account: Dict[str, Any]) -> str:
        require_feature(account, "can_ai_notes")
        user = _user(account)
        ratelimit.check("ai", user)
        return user

    @app.post("/v1/ai/summary", response_model=models.AIText, responses=_AI, summary="AI scan summary (Premium)")
    def ai_summary(body: AISummaryBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Claude explains the top results in HSF Score order (the web's AI Scan Summary).
        Research commentary, not investment advice."""
        user = _premium(account)
        out = ai.summary(user, body.run_id, shared=body.run_id is None)
        if out["run_id"] is None and body.run_id is not None:
            raise user_data.NotFound("scan")
        return out

    @app.post("/v1/ai/chat", response_model=models.AIChatAnswer, responses=_AI, summary="Ask about a scan (Premium)")
    def ai_chat(body: AIChatBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Questions about one scan's results (the web's results chat). Send the conversation so
        far; the last message must be the user's question. Up to 8 prior turns are used."""
        user = _premium(account)
        if body.messages[-1].role != "user":
            raise HTTPException(422, "The last message must be the user's question.")
        out = ai.chat(user, body.run_id, [m.model_dump() for m in body.messages])
        if out["run_id"] is None and body.run_id is not None:
            raise user_data.NotFound("scan")
        return out

    @app.post("/v1/ai/notes/{ticker}", response_model=models.AIText, responses=_AI,
              summary="AI setup note for a ticker (Premium)")
    def ai_note(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Claude's note on one result of the latest market scan; text is null when the ticker
        isn't in it."""
        user = _premium(account)
        return ai.ticker_note(user, ticker.strip().upper())

    @app.get("/v1/ai/brief-narrative", response_model=models.AIText, responses=_AI,
             summary="AI Market Brief narrative (Premium)")
    def ai_brief_narrative(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """A 2-3 sentence brief written from the Market Brief's facts only."""
        user = _premium(account)
        return json_safe(ai.brief_narrative(user))


class JournalCreate(BaseModel):
    ticker: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")
    entry_price: float = Field(gt=0, le=1_000_000)
    shares: int = Field(ge=0, le=10_000_000)


class JournalClose(BaseModel):
    exit_price: float = Field(gt=0, le=1_000_000)


class PaperConnect(BaseModel):
    api_key: str = Field(min_length=8, max_length=128, description="Alpaca PAPER API key ID")
    api_secret: str = Field(min_length=8, max_length=256, description="Alpaca PAPER API secret (stored encrypted, never returned)")


class PaperOrderBody(BaseModel):
    ticker: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$")
    qty: int = Field(ge=1, le=100_000, description="Whole shares")
    confirm: Literal[True] = Field(description="Must be true: the user confirmed this order (the web's confirmation step)")


def require_min_plan(account: Dict[str, Any], plan: str, message: str) -> Dict[str, Any]:
    from auth.tiering import has_min_tier

    ent = entitlements_for(account)
    if not (ent["is_admin"] or has_min_tier(ent["tier"], plan)):
        raise HTTPException(403, message)
    return ent


def _trading_routes(app: FastAPI) -> None:
    from api import trading

    _PRO = {**_AUTH, 403: {"description": "Pro feature"}}
    _PREM = {**_AUTH, 403: {"description": "Premium feature"}, 503: {"description": "Paper trading unavailable on the server"}}
    _PRO_MSG = "Trade plans and logging trades are part of Pro."

    @app.get("/v1/journal", response_model=models.Journal, responses=_AUTH, summary="Your trade journal")
    def journal(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Your logged trades (open first), open ones marked to live quotes, with closed-trade stats."""
        return json_safe(trading.journal(_user(account)))

    @app.post("/v1/journal", status_code=201, responses=_PRO, summary="Log a trade (Pro)")
    def journal_log(body: JournalCreate, account: Dict[str, Any] = Depends(current_account)) -> None:
        require_min_plan(account, "pro", _PRO_MSG)
        trading.log(_user(account), body.ticker.upper(), body.entry_price, body.shares)

    @app.post("/v1/journal/{trade_id}/close", status_code=204, responses={**_PRO, **_OWNED, 409: {"description": "Already closed"}},
              summary="Close a logged trade (Pro)")
    def journal_close(body: JournalClose, trade_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        require_min_plan(account, "pro", _PRO_MSG)
        if not trading.close(_user(account), trade_id, body.exit_price):
            raise user_data.NotFound("trade")

    @app.delete("/v1/journal/{trade_id}", status_code=204, responses={**_PRO, **_OWNED}, summary="Delete a logged trade (Pro)")
    def journal_delete(trade_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        require_min_plan(account, "pro", _PRO_MSG)
        if not trading.delete(_user(account), trade_id):
            raise user_data.NotFound("trade")

    @app.get("/v1/stocks/{ticker}/plan", response_model=models.TradePlan, responses={**_PRO, 404: {"description": "Not in the latest scan"}},
             summary="Trade plan for a scan result (Pro)")
    def stock_plan(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account),
                   account_size: float = Query(10_000.0, ge=100, le=1e9), risk_pct: float = Query(1.0, gt=0, le=10)
                   ) -> Dict[str, Any]:
        """The web's trade plan: stop at half the 20-day volatility (2-8%), targets at 1.5R and 3R,
        size from your risk budget. Educational only, not advice."""
        require_min_plan(account, "pro", _PRO_MSG)
        out = trading.plan(ticker.strip().upper(), account_size, risk_pct)
        if out is None:
            raise HTTPException(404, "That ticker isn't in the latest market scan.")
        return json_safe(out)

    @app.get("/v1/paper/account", response_model=models.PaperStatus, responses=_PREM, summary="Paper account status (Premium)")
    def paper_account(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        require_feature(account, "can_paper_trade")
        return json_safe(trading.paper_status(_user(account)))

    @app.post("/v1/paper/account", response_model=models.PaperStatus, responses={**_PREM, 400: {"description": "Keys rejected"}},
              summary="Connect your Alpaca paper account (Premium)")
    def paper_connect(body: PaperConnect, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Validates the keys against Alpaca's paper endpoint, then stores them encrypted. Paper keys only."""
        require_feature(account, "can_paper_trade")
        user = _user(account)
        ratelimit.check("paper_connect", user)
        return json_safe(trading.connect(user, body.api_key, body.api_secret))

    @app.delete("/v1/paper/account", status_code=204, responses=_PREM, summary="Disconnect your paper account (Premium)")
    def paper_disconnect(account: Dict[str, Any] = Depends(current_account)) -> None:
        require_feature(account, "can_paper_trade")
        trading.disconnect(_user(account))

    @app.get("/v1/paper/activity", response_model=models.PaperActivity, responses=_PREM, summary="Paper positions and orders (Premium)")
    def paper_activity(account: Dict[str, Any] = Depends(current_account),
                       limit: int = Query(25, ge=1, le=100)) -> Dict[str, Any]:
        require_feature(account, "can_paper_trade")
        return json_safe(trading.activity(_user(account), limit))

    @app.post("/v1/paper/orders", response_model=models.PaperOrder, status_code=201,
              responses={**_PREM, 400: {"description": "Order rejected"}, 422: {"description": "confirm must be true"},
                         429: {"description": "Too many orders this hour"}},
              summary="Paper trade a setup (Premium)")
    def paper_order(body: PaperOrderBody, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """A whole-share market BUY sent to your Alpaca PAPER account (no real money), imported into
        your journal. Show the user what will be sent and send confirm=true only after they confirm."""
        require_feature(account, "can_paper_trade")
        user = _user(account)
        ratelimit.check("paper_order", user)
        return json_safe(trading.order(user, body.ticker.upper(), body.qty))


class DeleteAccountBody(BaseModel):
    password: str = Field(min_length=1, max_length=256)
    confirm: Literal["DELETE"] = Field(description='Type "DELETE": the user confirmed permanent deletion')


def _delete_account_route(app: FastAPI) -> None:
    @app.delete("/v1/me", status_code=204, summary="Delete your account",
                responses={**_AUTH, 400: {"description": "Wrong password"},
                           409: {"description": "Cancel your paid subscription first (or admin account)"},
                           422: {"description": 'confirm must be "DELETE"'}, 429: {"description": "Too many attempts"}})
    def delete_me(body: DeleteAccountBody, account: Dict[str, Any] = Depends(current_account)) -> None:
        """Permanently deletes your account and its data (watchlists, alerts, journal, paper keys,
        settings, saved scans, sessions, devices). Refused while a paid subscription is active:
        cancel it first via POST /v1/billing/portal {"flow": "cancel"}. Can't be undone."""
        ratelimit.check("delete_account", _user(account))
        acct.delete_account(account, body.password)


def _failing_app(message: str):
    """ASGI app that refuses to start with `message`, so uvicorn logs
    "Application startup failed" with the real reason and exits."""
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.failed", "message": message})
            return
        raise RuntimeError(message)

    return app


def _module_app():
    """Module-level app for `uvicorn api.main:app` (settings from the environment).
    Importing never raises, so tests can import create_app without the secret."""
    try:
        return create_app()
    except RuntimeError as e:
        log.error("HSF API not started: %s", e)
        return _failing_app(f"HSF API not started: {e}")


app = _module_app()
