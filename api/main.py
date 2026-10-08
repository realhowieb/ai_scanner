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
    _outcome_routes(app)
    _market_routes(app)
    _ai_routes(app)
    _trading_routes(app)
    _public_routes(app)
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
    account = _recent_account(username) if username else None
    if not account or account.get("is_active") is False:
        raise HTTPException(401, _UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    return account


# Every signed-in request reads the account row; the database is a cross-region
# round trip or three away, so the row is reused for a few seconds per process.
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
    from auth.tiering import TIER_ORDER, has_min_tier
    from ui.app_session import ALERT_LIMIT_BY_TIER, compute_entitlements

    is_admin = bool(account.get("is_admin"))
    tier = str(account.get("tier") or "basic").strip().lower()
    tier = "admin" if is_admin else (tier if tier in TIER_ORDER else "basic")
    return {"tier": tier, "is_admin": is_admin,
            "entitlements": compute_entitlements(tier_obj=tier, is_admin=is_admin, has_min_tier_fn=has_min_tier),
            "alert_limit": ALERT_LIMIT_BY_TIER.get(tier, 1)}


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


class AlertRuleCreate(BaseModel):
    rule_type: str = Field(description="See GET /v1/alerts/rules/types, e.g. HSF_SCORE_CROSS_ABOVE")
    ticker: Optional[str] = Field(default=None, max_length=12, description="One ticker, or set watchlist_id")
    watchlist_id: Optional[int] = Field(default=None, ge=1, description="Every symbol on this watchlist")
    threshold: Optional[float] = None
    value: Optional[str] = Field(default=None, max_length=40, description="SETUP_APPEARED: only this setup (optional)")
    delivery_channels: Optional[List[Literal["in_app", "email"]]] = Field(
        default=None, max_length=2, description="Default in_app; email is Pro+")
    cooldown_seconds: Optional[int] = Field(default=None, ge=0, le=7 * 86400,
                                            description="Default 1 day for level rules, 1 hour for transitions")
    enabled: bool = True


class AlertRuleUpdate(BaseModel):
    threshold: Optional[float] = None
    value: Optional[str] = Field(default=None, max_length=40)
    enabled: Optional[bool] = None
    delivery_channels: Optional[List[Literal["in_app", "email"]]] = Field(default=None, min_length=1, max_length=2)
    cooldown_seconds: Optional[int] = Field(default=None, ge=0, le=7 * 86400)


class AlertUpdate(BaseModel):
    enabled: bool


def _user(account: Dict[str, Any]) -> str:
    return str(account["username"]).strip().lower()


def _data_routes(app: FastAPI) -> None:
    # ---- step 5: scans and stock detail ----
    @app.get("/v1/today/me", response_model=models.TodayPersonal, responses=_AUTH, summary="Today: your sections")
    def today_personal(account: Dict[str, Any] = Depends(current_account),
                       seen: Optional[int] = Query(None, ge=1, description="Market run this browser last saw (from `marker`)"),
                       baseline: Optional[int] = Query(None, ge=1, description="Its baseline run (from `marker`)")) -> Dict[str, Any]:
        """New since your last visit and your watchlist against the latest market scan."""
        from api.today import build_personal

        return build_personal(_user(account), seen, baseline)

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
                                                   description="Only setups with this signal"),
                     sort: Literal["score", "chg_pct", "gap_pct", "rvol", "prob"] = Query(
                         "score", description="Order of the rows the plan sees (descending); the plan's rows are "
                                              "always its top HSF-ranked setups")) -> Dict[str, Any]:
        """The latest market scan's HSF setups, ranked as in the Scanner, up to the plan's row cap."""
        from api.scans import latest_scan

        ent = entitlements_for(account)
        if signal == "prebreakout" and not ent["entitlements"].get("can_early_breakout"):
            raise HTTPException(403, "PreBreakout is a Premium feature.")
        return latest_scan(ent["entitlements"], ent["tier"], limit=limit, offset=offset,
                           min_score=min_score, signal=signal, sort=sort)

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
        """The list with each ticker's row in the latest market scan (`items[].latest`)."""
        from api.scans import watchlist_scan_state

        out = user_data.get_watchlist(_user(account), watchlist_id)
        state = watchlist_scan_state([i["ticker"] for i in out["items"]], entitlements_for(account)["entitlements"])
        out = {**out, "scan_at": state["scan_at"], "scan_total": state.get("total"), "stale": state["stale"],
               "items": [{**i, "latest": state["rows"].get(str(i["ticker"]).upper())} for i in out["items"]]}
        return json_safe(out)

    @app.patch("/v1/watchlists/{watchlist_id}", response_model=models.WatchlistDetail,
               responses={**_OWNED, 409: {"description": "A watchlist with that name exists"}})
    def watchlist_update(watchlist_id: int, body: WatchlistUpdate,
                         account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return json_safe(user_data.update_watchlist(_user(account), watchlist_id, name=body.name,
                                                    make_default=body.make_default))

    @app.delete("/v1/watchlists/{watchlist_id}", status_code=204, responses=_OWNED)
    def watchlist_delete(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> None:
        """Alert rules on this watchlist are switched off (kept, so you can see why)."""
        from api import alert_rules
        from db import alert_rules as rule_store

        user_data.delete_watchlist(_user(account), watchlist_id)
        try:  # best effort: the evaluator skips rules whose watchlist is gone either way
            alert_rules._db(rule_store.disable_rules_for_watchlist, _user(account), watchlist_id)
        except Exception as e:
            log.warning(json.dumps({"event": "watchlist_rules_disable_failed", "error": type(e).__name__}))

    @app.get("/v1/watchlists/{watchlist_id}/intelligence", response_model=models.WatchlistIntelligence,
             responses=_OWNED, summary="HSF intelligence for every symbol on a watchlist")
    def watchlist_intelligence(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Score, rank and their change since the previous scan, setup, signals, PreBreakout
        (Premium), price, RVOL, EMA cross, freshness and active alert count for each symbol,
        from the latest saved market scan (no live quote calls). Fields with no canonical
        source are null and listed in `unavailable_fields`."""
        from api import watchlist_intel

        user = _user(account)
        wl = user_data.get_watchlist(user, watchlist_id)
        return json_safe(watchlist_intel.intelligence(user, wl, entitlements_for(account)["entitlements"]))

    @app.get("/v1/watchlists/{watchlist_id}/changes", response_model=models.WatchlistChanges,
             responses=_OWNED, summary="What changed on a watchlist since the previous scan")
    def watchlist_changes(watchlist_id: int, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """The canonical HSF changes (new, dropped, rising/falling, status, fading, signals
        incl. PreBreakout) between the two latest market scans for this list's symbols, with
        rank moves, plus your rule alerts on them since the previous scan."""
        from api import watchlist_intel

        user = _user(account)
        wl = user_data.get_watchlist(user, watchlist_id)
        return json_safe(watchlist_intel.changes(user, wl, entitlements_for(account)["entitlements"]))

    @app.post("/v1/watchlists/{watchlist_id}/tickers", response_model=models.TickersResult, responses=_OWNED)
    def watchlist_add(watchlist_id: int, body: TickersBody,
                      account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return user_data.add_tickers(_user(account), watchlist_id, body.tickers)

    @app.post("/v1/watchlists/{watchlist_id}/symbols", response_model=models.TickersResult, responses=_OWNED,
              summary="Add symbols (same as /tickers)")
    def watchlist_add_symbols(watchlist_id: int, body: TickersBody,
                              account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return user_data.add_tickers(_user(account), watchlist_id, body.tickers)

    @app.delete("/v1/watchlists/{watchlist_id}/tickers/{ticker}", status_code=204, responses=_OWNED)
    def watchlist_remove(watchlist_id: int, ticker: str = TICKER,
                         account: Dict[str, Any] = Depends(current_account)) -> None:
        user_data.remove_ticker(_user(account), watchlist_id, ticker.upper())

    @app.delete("/v1/watchlists/{watchlist_id}/symbols/{ticker}", status_code=204, responses=_OWNED,
                summary="Remove a symbol (same as /tickers/{ticker})")
    def watchlist_remove_symbol(watchlist_id: int, ticker: str = TICKER,
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

    @app.get("/v1/alerts/events", response_model=List[models.AlertEvent],
             responses={**_AUTH, 422: {"description": "Invalid cursor or filter"}})
    def alert_events(account: Dict[str, Any] = Depends(current_account),
                     limit: int = Query(20, ge=1, le=100),
                     ticker: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$"),
                     rule_id: Optional[int] = Query(None, ge=1, description="Only this rule's events"),
                     watchlist_id: Optional[int] = Query(None, ge=1, description="Only rule events from this watchlist"),
                     triggered_after: Optional[dt.datetime] = Query(None),
                     triggered_before: Optional[dt.datetime] = Query(None),
                     source: Optional[Literal["alert", "rule"]] = Query(None, description="alert: ticker alerts; rule: alert rules"),
                     cursor: Optional[str] = Query(None, max_length=200, description="The last event's cursor, for the next page")
                     ) -> List[Dict[str, Any]]:
        """Your fired alerts (ticker alerts and alert rules), newest first. Page with `cursor`."""
        from api import alert_rules

        try:
            return json_safe(alert_rules.list_events(
                _user(account), limit=limit, ticker=ticker.upper() if ticker else None, rule_id=rule_id,
                watchlist_id=watchlist_id, triggered_after=_aware(triggered_after),
                triggered_before=_aware(triggered_before), cursor=cursor, source=source))
        except alert_rules.RuleError as e:
            raise HTTPException(422, str(e)) from e

    # ---- alert rules (server-evaluated conditions on HSF intelligence) ----
    @app.get("/v1/alerts/rules", response_model=models.AlertRules, responses=_AUTH, summary="Your alert rules")
    def alert_rules_list(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Your alert rules, the plan's active-alert limit (shared with ticker alerts) and
        what the plan allows."""
        from api import alert_rules

        ent = entitlements_for(account)
        user = _user(account)
        return json_safe({"limit": ent["alert_limit"], "used": alert_rules.count_active(user),
                          "capabilities": _capabilities(ent), "rules": alert_rules.list_rules(user)})

    @app.get("/v1/alerts/rules/types", response_model=List[models.AlertRuleType], responses=_AUTH,
             summary="Alert rule types")
    def alert_rule_types(account: Dict[str, Any] = Depends(current_account)) -> List[Dict[str, Any]]:
        """The rule types, their thresholds and whether your plan includes each."""
        from api import alert_rules

        return alert_rules.rule_types(entitlements_for(account)["entitlements"])

    @app.post("/v1/alerts/rules", response_model=models.AlertRule, status_code=201, summary="Create an alert rule",
              responses={**_AUTH, 403: {"description": "Plan limit, rule type or channel not on your plan"},
                         404: {"description": "Watchlist not found (or not yours)"},
                         409: {"description": "You already have this rule"},
                         422: {"description": "Invalid rule"}})
    def alert_rule_create(body: AlertRuleCreate, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Set exactly one of `ticker` or `watchlist_id`. Rules count toward the plan's alert
        limit together with ticker alerts."""
        from api import alert_rules

        with _rule_errors():
            return json_safe(alert_rules.create_rule(_user(account), entitlements_for(account), body.model_dump()))

    @app.get("/v1/alerts/rules/{rule_id}", response_model=models.AlertRule, responses=_OWNED, summary="One alert rule")
    def alert_rule_get(rule_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api import alert_rules

        return json_safe(alert_rules.get_rule(_user(account), rule_id))

    @app.patch("/v1/alerts/rules/{rule_id}", response_model=models.AlertRule, summary="Change an alert rule",
               responses={**_OWNED, 403: {"description": "Plan limit or channel not on your plan"},
                          422: {"description": "Invalid change"}})
    def alert_rule_update(body: AlertRuleUpdate, rule_id: int = Path(ge=1),
                          account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Threshold, value, enabled, channels, cooldown. A new threshold or value starts the
        rule from a fresh baseline."""
        from api import alert_rules

        with _rule_errors():
            return json_safe(alert_rules.update_rule(_user(account), entitlements_for(account), rule_id,
                                                     body.model_dump(exclude_unset=True)))

    @app.delete("/v1/alerts/rules/{rule_id}", status_code=204, responses=_OWNED, summary="Delete an alert rule")
    def alert_rule_delete(rule_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        """Its past events stay in /v1/alerts/events."""
        from api import alert_rules

        alert_rules.delete_rule(_user(account), rule_id)

    @app.get("/v1/me/capabilities", response_model=models.Capabilities, responses=_AUTH,
             summary="Your plan's watchlist and alert limits")
    def my_capabilities(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        return _capabilities(entitlements_for(account))


class SignupBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    username: str = Field(min_length=1, max_length=40, description="Shown in the app; can also be used to sign in on the web")
    accept_terms: bool = Field(description="The usage agreement checkbox on the web sign-up form")
    client: Optional[str] = Field(default=None, max_length=80)
    attribution: Optional[Dict[str, str]] = Field(
        default=None, description="Web sign-ups: first-visit utm_* tags and referrer, for the acquisition funnel")


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
        if body.attribution is not None:
            _track(body.attribution, "signup_completed", username=res["email"], plan="basic")
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
        acct.change_password(_fresh_account(account), body.current_password, body.new_password)
        forget_account(_user(account))
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


class OutcomeParams:
    """Explicit, validated Outcome Intelligence filters. Nothing is filtered unless
    asked for, and the response echoes every filter back."""

    def __init__(self,
                 setup: Optional[str] = Query(None, max_length=40, description="Canonical setup (primary_setup), e.g. breakout"),
                 signal: Optional[str] = Query(None, max_length=40, description="Signal label, e.g. golden_cross"),
                 min_score: Optional[float] = Query(None, ge=0, le=100),
                 max_score: Optional[float] = Query(None, ge=0, le=100),
                 score_bucket: Optional[str] = Query(None, pattern=r"^\d{1,3}-\d{1,3}$", description="e.g. 80-89"),
                 score_version: Optional[str] = Query(None, max_length=20, description="HSF score version as frozen"),
                 start_date: Optional[dt.date] = Query(None, description="Observed on/after (YYYY-MM-DD, UTC)"),
                 end_date: Optional[dt.date] = Query(None, description="Observed on/before (YYYY-MM-DD, UTC)"),
                 certified_only: bool = Query(False, description="Only matured rows passing the canonical eligibility rule"),
                 matured_only: bool = Query(False, description="Only records matured at the horizon (counts drop pending)"),
                 unit: Literal["signal_day", "observation"] = Query(
                     "signal_day", description="signal_day: one record per ticker per entry day (default); observation: every frozen row")):
        from analytics import outcome_intelligence as oi

        self.unit = unit
        try:
            self.filters = oi.normalize_filters(setup=setup, signal=signal, min_score=min_score, max_score=max_score,
                                                score_bucket=score_bucket, score_version=score_version,
                                                start_date=start_date, end_date=end_date,
                                                certified_only=certified_only, matured_only=matured_only)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e


def _outcome_routes(app: FastAPI) -> None:
    from api import outcomes

    _PRO = {**_AUTH, 403: {"description": "Pro feature"}, 422: {"description": "Invalid filter"},
            503: {"description": "Database unavailable"}}
    HORIZON = Query(5, description="Trading days: 1, 3 or 5. 5 is the pre-declared primary horizon, not a data-chosen one")

    def _run(fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return json_safe(fn())
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    def _horizon(h: Optional[int]) -> Optional[int]:
        if h is not None and h not in (1, 3, 5):
            raise HTTPException(422, "horizon must be 1, 3 or 5.")
        return h

    @app.get("/v1/outcomes/summary", response_model=models.OutcomeSummary, responses=_PRO,
             summary="Outcome Intelligence: overall evidence (Pro)")
    def outcomes_summary(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                         horizon: int = HORIZON) -> Dict[str, Any]:
        """Every eligible HSF signal unless filters are given: counts, date range, raw and
        SPY-relative returns, win and beat rates, MFE/MAE, with sample sizes."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.summary(p.filters, h, p.unit))

    @app.get("/v1/outcomes/scores", response_model=models.OutcomeScores, responses=_PRO,
             summary="Outcome Intelligence: by HSF score bucket (Pro)")
    def outcomes_scores(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                        horizon: int = HORIZON,
                        buckets: Optional[str] = Query(None, max_length=120, pattern=r"^[0-9,\- ]+$",
                                                       description="e.g. 40-49,50-59,60-69 (default: canonical HSF buckets)")
                        ) -> Dict[str, Any]:
        """Every bucket in score order, weak ones included, plus a monotonicity check
        that lists each inversion (a lower bucket beating a higher one)."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.scores(p.filters, h, p.unit, buckets))

    @app.get("/v1/outcomes/horizons", response_model=models.OutcomeHorizons, responses=_PRO,
             summary="Outcome Intelligence: by horizon (Pro)")
    def outcomes_horizons(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends()) -> Dict[str, Any]:
        """The same metrics for 1, 3 and 5 trading days. No horizon is singled out."""
        require_feature(account, "can_track_record")
        return _run(lambda: outcomes.horizons(p.filters, p.unit))

    @app.get("/v1/outcomes/setups", response_model=models.OutcomeGroups, responses=_PRO,
             summary="Outcome Intelligence: by setup or signal (Pro)")
    def outcomes_setups(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                        horizon: int = HORIZON,
                        group_by: Literal["setup", "signal"] = Query("setup")) -> Dict[str, Any]:
        """Groups ordered by sample size (never by performance), each with its counts."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.setups(p.filters, h, p.unit, group_by))

    @app.get("/v1/outcomes/timeseries", response_model=models.OutcomeTimeseries, responses=_PRO,
             summary="Outcome Intelligence: through time (Pro)")
    def outcomes_timeseries(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                            horizon: int = HORIZON,
                            period: Literal["day", "week", "month"] = Query("week")) -> Dict[str, Any]:
        """Per observation period (grouped by when the signal was observed)."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.timeseries(p.filters, h, p.unit, period))

    @app.get("/v1/outcomes/symbols/{ticker}", response_model=models.OutcomeSymbol, responses=_PRO,
             summary="Outcome Intelligence: one ticker's HSF history (Pro)")
    def outcomes_symbol(ticker: str = TICKER, account: Dict[str, Any] = Depends(current_account),
                        p: OutcomeParams = Depends(),
                        horizon: Optional[int] = Query(None, description="1, 3 or 5; omit for all three"),
                        page: int = Query(1, ge=1, le=10_000), page_size: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
        """Aggregate evidence per horizon plus the observation-level records, newest first."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        return _run(lambda: outcomes.symbol(ticker.strip().upper(), p.filters, h, p.unit, page, page_size))

    @app.get("/v1/outcomes/query", response_model=models.OutcomeQuery, responses=_PRO,
             summary="Outcome Intelligence: filtered evidence with records (Pro)")
    def outcomes_query(account: Dict[str, Any] = Depends(current_account), p: OutcomeParams = Depends(),
                       ticker: Optional[str] = Query(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$"),
                       horizon: int = HORIZON,
                       page: int = Query(1, ge=1, le=10_000), page_size: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
        """Any combination of the explicit filters, one horizon: metrics plus paged records."""
        require_feature(account, "can_track_record")
        h = _horizon(horizon)
        f = {**p.filters, "ticker": ticker.strip().upper() if ticker else None}
        return _run(lambda: outcomes.query(f, h, p.unit, page, page_size))


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


FUNNEL_EVENTS = ("landing_visit", "primary_cta_click", "signup_started")


class FunnelEvent(BaseModel):
    event: Literal["landing_visit", "primary_cta_click", "signup_started"]
    attribution: Dict[str, str] = Field(default_factory=dict, max_length=12,
                                        description="utm_* tags and referrer from the visitor's first page")
    surface: Optional[str] = Field(default=None, max_length=40, description="Which button or page")


class UnsubscribeBody(BaseModel):
    token: str = Field(min_length=10, max_length=64)
    kind: Literal["digest", "evening", "alerts", "all"]


def _track(params: Dict[str, str], event: str, **kwargs: Any) -> None:
    """Best-effort acquisition event for the web app (never raises, never blocks)."""
    try:
        from ui.acquisition import attribution_from_params, track_event

        params = {str(k)[:40]: str(v)[:200] for k, v in (params or {}).items()}
        track_event(event, attribution=attribution_from_params(params, params.get("referrer")), **kwargs)
    except Exception:
        log.debug("acquisition event %s not recorded", event, exc_info=True)


def _public_routes(app: FastAPI) -> None:
    """Signed-out endpoints for the web app: plans and pricing, funnel events and the
    emailed unsubscribe link."""

    @app.get("/v1/plans", response_model=models.Plans, summary="Plans and pricing (public)")
    def plans() -> Dict[str, Any]:
        """The plan comparison the landing and pricing pages show, from the same source as
        the Billing page (ui.pricing), so copy can't drift from what each plan gets."""
        from ui import pricing as p

        highlights = p.plan_highlights()
        tiers = [{"id": t, "name": p.TIER_NAMES[t], "price": p.PRICES[t], "yearly_price": p.YEARLY_PRICES.get(t),
                  "tagline": p.TAGLINES[t], "alert_limit": int(p.ALERT_LIMIT_BY_TIER.get(t, 1)),
                  "highlights": highlights.get(t, [])} for t in p.TIERS]
        rows = [{"label": label, **{t: (int(p.ALERT_LIMIT_BY_TIER.get(t, 1)) if flag == p.ALERTS else p.included(flag, t))
                                     for t in p.TIERS}} for label, flag in p.ROWS]
        return {"tiers": tiers, "rows": rows}

    @app.post("/v1/events", status_code=202, responses={429: {"description": "Too many events from this address"}},
              summary="Record a signed-out funnel event")
    def funnel_event(body: FunnelEvent, _l: None = Depends(ratelimit.limit("events"))) -> None:
        """Landing visit, call-to-action click or sign-up started, with the visitor's utm tags.
        Stores no email, name or IP. Best effort: always accepted."""
        _track(body.attribution, body.event, metadata={"surface": body.surface or "web", "app": "web"})

    def _unsub_user(token: str) -> str:
        from db.email_prefs import user_for_token

        user = user_for_token(token)
        if not user:
            raise HTTPException(400, "This unsubscribe link isn't valid. Sign in and open Account to change your emails.")
        return user

    def _unsub_state(user: str) -> Dict[str, Any]:
        from db.email_prefs import get_prefs
        from ui.log_privacy import mask_email

        return {"email": mask_email(user), "prefs": get_prefs(user)}

    _UNSUB = {400: {"description": "Invalid link"}, 429: {"description": "Too many attempts"}}

    @app.get("/v1/email-preferences/unsubscribe", response_model=models.UnsubscribeState, responses=_UNSUB,
             summary="Email settings behind an unsubscribe link")
    def unsubscribe_state(t: str = Query(min_length=10, max_length=64), _l: None = Depends(ratelimit.limit("unsubscribe"))
                          ) -> Dict[str, Any]:
        """Which emails the link's account gets. Changes nothing (email scanners open links)."""
        return _unsub_state(_unsub_user(t))

    @app.post("/v1/email-preferences/unsubscribe", response_model=models.UnsubscribeState,
              responses={**_UNSUB, 503: {"description": "Couldn't save"}}, summary="Unsubscribe with an emailed link")
    def unsubscribe(body: UnsubscribeBody, _l: None = Depends(ratelimit.limit("unsubscribe"))) -> Dict[str, Any]:
        """Turns off one kind of email, or all of them, for the link's account. Account emails
        (verification, password reset) still go out."""
        from db.email_prefs import KINDS, set_prefs

        user = _unsub_user(body.token)
        kinds = KINDS if body.kind == "all" else (body.kind,)
        if not set_prefs(user, **{k: False for k in kinds}):
            raise HTTPException(503, "Couldn't save that right now. Please try again in a minute.")
        return _unsub_state(user)


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
        acct.delete_account(_fresh_account(account), body.password)
        forget_account(_user(account))


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


def _warm_caches() -> None:
    """Build the Market Brief, then the Day Trader "Top movers" table (the page's default
    source), once in the background after a (re)start, so the first visitor after a deploy
    or a free-plan wake-up doesn't wait 20-50 s for either. One after the other on one
    thread to keep the start-up memory peak low. HSF_WARM_BRIEF=0 / HSF_WARM_DAY_TRADER=0
    turn each off."""
    import os
    import threading

    if os.environ.get("RENDER", "").strip().lower() != "true":
        return
    jobs = []
    if os.environ.get("HSF_WARM_BRIEF", "1").strip() != "0":
        jobs.append(("brief", lambda m: m._brief_core()))
    if os.environ.get("HSF_WARM_DAY_TRADER", "1").strip() != "0":
        jobs.append(("day trader", lambda m: m.day_trader("movers")))
    if not jobs:
        return

    def run() -> None:
        from api import market

        for name, job in jobs:
            try:
                job(market)
            except Exception as e:  # warming is best effort; the first visitor builds it instead
                log.warning("%s warm-up failed: %s", name, str(e)[:120])

    threading.Thread(target=run, name="cache-warmup", daemon=True).start()


def _start_realtime_alerts() -> None:
    """P1-60: run the real-time price-alert worker here, now that hsf-api is on an
    always-on plan. No-op unless REALTIME_ALERTS_ENABLED=1 (leave it off on the
    billing service; the shared last_fired_at throttle covers any overlap). The same loop
    evaluates alert rules (api.alert_rules)."""
    try:
        from api.alert_rules import worker_pass
        from billing_service.realtime_alerts import register_pass_hook, start_background_worker

        register_pass_hook(worker_pass)  # alert rules ride the same loop (HSF_ALERT_RULES_ENABLED=0 stops them)
        start_background_worker()
    except Exception as e:  # alerts are best effort; the API serves regardless
        log.warning("realtime alerts worker failed to start: %s", str(e)[:120])


def _module_app():
    """Module-level app for `uvicorn api.main:app` (settings from the environment).
    Importing never raises, so tests can import create_app without the secret."""
    try:
        app = create_app()
    except RuntimeError as e:
        log.error("HSF API not started: %s", e)
        return _failing_app(f"HSF API not started: {e}")
    _warm_caches()
    _start_realtime_alerts()
    return app


app = _module_app()
