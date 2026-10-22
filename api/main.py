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
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.routing import APIRoute

from api import account as acct
from api import devices, models, routes, store, user_data

# Re-exported: tests and older callers reach these through api.main.
from api.deps import _account_cache, _recent_account, current_account, entitlements_for, forget_account  # noqa: F401
from api.etag import ETagMiddleware
from api.memory import memory_status, rss_mb, tune_allocator
from api.scans import json_safe
from api.settings import Settings, load_settings

tune_allocator()  # cap malloc arenas before any worker thread starts (api/memory.py)

log = logging.getLogger("hsf_api")
access_log = logging.getLogger("hsf_api.access")
if not access_log.handlers:  # uvicorn only configures its own loggers; one JSON line per request to stdout
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    access_log.addHandler(_h)
    access_log.setLevel(logging.INFO)
    access_log.propagate = False
_SAFE_RID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


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


READYZ_RETRY_S = 2.0

# Sent on every answer. The API speaks JSON to the web server and the phone app, so
# nothing it returns should be sniffed as another type, framed, or leak a referrer.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000",
}


def docs_enabled() -> bool:
    """/docs and /openapi.json list every endpoint, admin ones included. They're on for
    local work and off on Render unless HSF_API_DOCS=1. The web client's copy of the
    contract comes from scripts/export_openapi.py, not from the live service."""
    raw = os.environ.get("HSF_API_DOCS", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return os.environ.get("RENDER", "").strip().lower() != "true"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    docs = docs_enabled()
    app = FastAPI(title="HSFinest.AI API", version="1.0.0",
                  docs_url="/docs" if docs else None, redoc_url="/redoc" if docs else None,
                  openapi_url="/openapi.json" if docs else None,
                  responses={422: {"model": models.ValidationErrorResponse, "description": "Invalid request fields"}})
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException

    from api.errors import http_response, validation_response

    app.add_exception_handler(RequestValidationError, validation_response)
    app.add_exception_handler(HTTPException, http_response)
    app.router.route_class = _ReleasingRoute
    app.state.settings = settings
    # Scan, history and stock responses are large JSON; the web BFF's fetch() accepts
    # gzip and unpacks it, so this only shrinks what crosses the network.
    # Inside GZip so the ETag is a hash of the plain JSON (api/etag.py).
    app.add_middleware(ETagMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
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
        from db.traffic import scope

        try:
            with scope("api.request") as db_metrics:
                response = await call_next(request)
                status = response.status_code
        finally:
            route = request.scope.get("route")
            from api.operations import record_request

            record_request(rid, request.method, getattr(route, "path", None) or "unmatched",
                           status, (time.perf_counter() - t0) * 1000)
            access_log.info(json.dumps({
                "request_id": rid, "method": request.method,
                "route": getattr(route, "path", None) or "unmatched", "status": status,
                "db_traffic": db_metrics.snapshot(),
                "ms": round((time.perf_counter() - t0) * 1000, 1), "rss_mb": rss_mb()}))
        response.headers["X-Request-ID"] = rid
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
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
    def root() -> Any:
        if docs:
            return RedirectResponse("/docs")
        return {"ok": True, "service": "HSFinest.AI API"}

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
        return JSONResponse({"detail": str(exc), "code": "paper_unavailable",
                             "request_id": getattr(_request.state, "request_id", None)}, status_code=503)

    @app.exception_handler(trading_mod.PaperRejected)
    def _paper_rejected(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc), "code": "paper_rejected",
                             "request_id": getattr(_request.state, "request_id", None)}, status_code=400)

    @app.exception_handler(trading_mod.OrderConflict)
    def _paper_conflict(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc), "code": "order_conflict",
                             "request_id": getattr(_request.state, "request_id", None)}, status_code=409)

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

    @app.exception_handler(acct.VerificationUnavailable)
    def _verification_down(_request: Request, exc: Exception) -> JSONResponse:
        log.warning("email verification lookup failed: %s", type(exc.__cause__ or exc).__name__)
        return JSONResponse({"detail": "We can't confirm your email verification right now. "
                                       "Please try again in a few minutes.",
                             "code": "verification_unavailable"},
                            status_code=503, headers={"Retry-After": "30"})

    @app.get("/readyz", response_model=models.Ready,
             responses={503: {"description": "Database unavailable, or (strict=true) a scheduled scan was missed"}})
    def readyz(strict: bool = Query(False, description=(
            "Answer 503 when a scheduled full-market scan was missed, so an outside uptime "
            "monitor alerts on stale data as well as on a database outage"))) -> Any:
        """Readiness: the database answers, plus the latest market scan's age for
        freshness monitoring. /healthz stays the liveness check. One failed database
        ping is retried after READYZ_RETRY_S: a database waking from idle shouldn't
        page an uptime monitor. db_ping_ms is the time of the answered ping."""
        t0 = time.perf_counter()
        try:
            store.ping()
        except store.DatabaseUnavailable:
            time.sleep(READYZ_RETRY_S)
            t0 = time.perf_counter()
            store.ping()  # a second failure is a real outage: 503 from the handler
        db_ms = round((time.perf_counter() - t0) * 1000, 1)
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
        body = {"ok": True, "database": "ok", "db_ping_ms": db_ms, "latest_scan_at": latest,
                "scan_age_minutes": age, **fresh, **memory_status()}
        from api import shared_cache

        kv = shared_cache.status()  # informational: the API works without Key Value
        body["key_value"] = {"configured": kv["configured"], "ok": kv["ok"], "ping_ms": kv["ping_ms"]}
        if strict and fresh["stale"]:
            return JSONResponse({**body, "ok": False}, status_code=503)
        return body

    async def _cap_threads() -> None:
        """Requests run on a shared thread pool, and each busy thread keeps its own warm
        database connection and memory. Starlette's default is 40; HSF_API_THREADS
        (default 24) keeps connections and memory in check on a 512 MB instance."""
        import anyio.to_thread

        try:
            n = int(os.environ.get("HSF_API_THREADS", "24").strip() or 24)
        except ValueError:
            n = 24
        if n > 0:
            anyio.to_thread.current_default_thread_limiter().total_tokens = n

    app.router.on_startup.append(_cap_threads)

    routes.register_all(app)
    from api import billing

    billing.register(app)
    return app


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


def _market():
    from api import market

    return market


def _scans():
    from api import scans

    return scans


def _history():
    from api import history as _h

    return _h


def _warm_caches() -> None:
    """Build the Market Brief, then the Day Trader "Top movers" table (the page's default
    source), then what Stock Intelligence and Track record pages share, once in the
    background after a (re)start, so the first visitor after a deploy doesn't wait for
    them. One after the other on one thread to keep the start-up memory peak low.
    HSF_WARM_BRIEF=0 / HSF_WARM_DAY_TRADER=0 / HSF_WARM_STOCK=0 / HSF_WARM_TRACK_RECORD=0
    turn each off."""
    import os
    import threading

    if os.environ.get("RENDER", "").strip().lower() != "true":
        return
    jobs = []
    if os.environ.get("HSF_WARM_BRIEF", "1").strip() != "0":
        jobs.append(("brief", lambda: _market()._brief_core()))
    if os.environ.get("HSF_WARM_DAY_TRADER", "1").strip() != "0":
        jobs.append(("day trader", lambda: _market().day_trader("movers")))
    if os.environ.get("HSF_WARM_STOCK", "1").strip() != "0":
        jobs.append(("stock pages", lambda: _scans().warm_stock_pages()))
    if os.environ.get("HSF_WARM_TRACK_RECORD", "1").strip() != "0":
        jobs.append(("track record", lambda: _history().track_record()))
    if not jobs:
        return

    def run() -> None:
        for name, job in jobs:
            try:
                job()
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
        from api.webpush import alert_fired
        from billing_service.realtime_alerts import register_fire_hook, register_pass_hook, start_background_worker

        register_pass_hook(worker_pass)  # alert rules ride the same loop (HSF_ALERT_RULES_ENABLED=0 stops them)
        register_fire_hook(alert_fired)  # browser notifications for price alerts (off until VAPID keys are set)
        start_background_worker()
    except Exception as e:  # alerts are best effort; the API serves regardless
        log.warning("realtime alerts worker failed to start: %s", str(e)[:120])


def _register_telegram_webhook() -> None:
    """Point the Telegram bot at this API once it's deployed (no-op until the TELEGRAM_* settings exist)."""
    if os.environ.get("RENDER", "").strip().lower() != "true":
        return
    import threading

    def run() -> None:
        try:
            from api.chat_delivery import register_telegram_webhook

            register_telegram_webhook()
        except Exception as e:  # best effort; Telegram linking just stays unavailable
            log.warning("telegram webhook registration failed: %s", type(e).__name__)

    threading.Thread(target=run, name="telegram-webhook", daemon=True).start()


def _start_day_trader_keep_warm() -> None:
    """Keep the Day Trader movers table warm during market hours (api.market)."""
    try:
        _market().start_day_trader_keep_warm()
    except Exception as e:  # best effort; the first visitor builds it instead
        log.warning("day trader keep-warm failed to start: %s", type(e).__name__)


def _start_scan_watchdog() -> None:
    """P2-80: flag (and re-run, with a token) a scan slot that didn't run (api/scan_watchdog.py)."""
    try:
        from api.scan_watchdog import start

        start()
    except Exception as e:  # best effort; the daily health report still catches missed slots
        log.warning("scan watchdog failed to start: %s", type(e).__name__)


def _module_app():
    """Module-level app for `uvicorn api.main:app` (settings from the environment).
    Importing never raises, so tests can import create_app without the secret."""
    try:
        app = create_app()
    except RuntimeError as e:
        log.error("HSF API not started: %s", e)
        return _failing_app(f"HSF API not started: {e}")
    # One stuck query mustn't hold a request thread forever (db/engine.py). Jobs keep no limit.
    os.environ.setdefault("HSF_DB_STATEMENT_TIMEOUT_MS", "30000")
    from api.monitoring import init_api_monitoring

    init_api_monitoring()
    _warm_caches()
    _start_realtime_alerts()
    _register_telegram_webhook()
    _start_scan_watchdog()
    _start_day_trader_keep_warm()
    if os.getenv("BILLING_MODE", "external").strip().lower() == "embedded":
        from billing_service.main import start_reconciliation_worker

        start_reconciliation_worker()
    return app


app = _module_app()
