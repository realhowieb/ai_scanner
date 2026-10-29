import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import psycopg2
import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_log = logging.getLogger("billing_service")


def _validate_email(email: str) -> str:
    """Normalize and validate an email from a Stripe webhook. Raises ValueError if invalid."""
    email = (email or "").strip().lower()
    if not email or not _EMAIL_RE.match(email):
        raise ValueError(f"Invalid or missing email from webhook: {email!r}")
    return email

def _start_realtime_alerts() -> None:
    """Start the real-time price-alert worker (no-op unless enabled via env)."""
    try:
        from billing_service.realtime_alerts import start_background_worker
    except ImportError:  # running as a flat module dir on Render
        try:
            from realtime_alerts import start_background_worker  # type: ignore
        except ImportError:
            return
    try:
        start_background_worker()
    except Exception as e:
        _log.warning("realtime alerts worker failed to start: %s", e)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # P2-23: lifespan replaces FastAPI's deprecated startup-event decorator.
    _start_realtime_alerts()
    yield


app = FastAPI(lifespan=_lifespan)


# ---------- ENV ----------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

STRIPE_PRICE_PRO = os.getenv("STRIPE_PRICE_PRO", "").strip()
STRIPE_PRICE_PREMIUM = os.getenv("STRIPE_PRICE_PREMIUM", "").strip()


def _price_ids(name: str) -> frozenset:
    """Comma-separated Stripe price ids from an env var (blank entries dropped)."""
    return frozenset(p.strip() for p in os.getenv(name, "").split(",") if p.strip())


# Price change 2026-10 ($19/$39 -> $25/$40): Stripe prices can't be edited, so
# new prices get new ids. Subscribers still on an older price keep their plan
# (grandfathered): list the previous ids here, comma-separated. Without this an
# old price id would map to Free on the next subscription webhook.
STRIPE_PRICE_PRO_LEGACY = _price_ids("STRIPE_PRICE_PRO_LEGACY")
STRIPE_PRICE_PREMIUM_LEGACY = _price_ids("STRIPE_PRICE_PREMIUM_LEGACY")
# Yearly prices ($250 / $400). Optional: unset means checkout is monthly only.
STRIPE_PRICE_PRO_YEARLY = os.getenv("STRIPE_PRICE_PRO_YEARLY", "").strip()
STRIPE_PRICE_PREMIUM_YEARLY = os.getenv("STRIPE_PRICE_PREMIUM_YEARLY", "").strip()

APP_SUCCESS_URL = os.getenv("APP_SUCCESS_URL", "").strip()  # e.g. https://yourapp.com
APP_CANCEL_URL = os.getenv("APP_CANCEL_URL", "").strip()    # e.g. https://yourapp.com
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()        # Neon Postgres URL
APP_PORTAL_RETURN_URL = os.getenv("APP_PORTAL_RETURN_URL", "").strip()  # e.g. https://hsfinestai.streamlit.app/billing
# Other apps that may send return links (success_url / return_url overrides):
# the web app. Comma-separated hosts; links to them must be https. The host of
# APP_SUCCESS_URL (Streamlit) is always allowed.
APP_RETURN_HOSTS = frozenset(
    h.strip().lower()
    for h in os.getenv("APP_RETURN_HOSTS", "app.hsfinest.com,hsf-web-v1.onrender.com").split(",")
    if h.strip()
)

if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY env var")

stripe.api_key = STRIPE_SECRET_KEY
stripe.max_network_retries = 1
if hasattr(stripe, "RequestsClient"):
    stripe.default_http_client = stripe.RequestsClient(timeout=15)
print(
    "[billing_service] starting | "
    f"prices: pro={'set' if bool(STRIPE_PRICE_PRO) else 'missing'}, "
    f"premium={'set' if bool(STRIPE_PRICE_PREMIUM) else 'missing'} | "
    f"db={'set' if bool(DATABASE_URL) else 'missing'} | "
    f"success_url={'set' if bool(APP_SUCCESS_URL) else 'missing'} | "
    f"cancel_url={'set' if bool(APP_CANCEL_URL) else 'missing'} | "
    f"webhook_secret={'set' if bool(STRIPE_WEBHOOK_SECRET) else 'missing'}"
)


# ---------- DB helpers ----------

def _append_qp(url: str, key: str, value: str) -> str:
    base = (url or "").strip()
    if not base:
        raise ValueError("Redirect URL is required")
    # Idempotent: don't append the param if it's already present (avoids
    # ?portal=return&portal=return when APP_*_URL already carries it).
    if f"{key}=" in base:
        return base
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{key}={value}"


def _allowed_return_url(candidate: str) -> str | None:
    """A caller-supplied return link, if it points at one of our apps: the host of
    APP_SUCCESS_URL, or an https link to a host in APP_RETURN_HOSTS. Anything
    else is ignored so this never becomes an open redirect."""
    try:
        url = urlparse(candidate or "")
        base_host = urlparse(APP_SUCCESS_URL or "").netloc
        if not url.netloc:
            return None
        if base_host and url.netloc == base_host:
            return candidate
        if url.scheme == "https" and url.netloc.lower() in APP_RETURN_HOSTS:
            return candidate
    except Exception:
        pass
    return None


def _cancel_url_for(success_url: str) -> str:
    """Checkout cancel link: back to the app the customer came from. APP_CANCEL_URL
    unless the success link points at another allowed app (the web app)."""
    url = urlparse(success_url or "")
    if url.netloc and url.netloc != urlparse(APP_SUCCESS_URL or "").netloc:
        return f"{url.scheme}://{url.netloc}/?checkout=cancel"
    return _append_qp(APP_CANCEL_URL, "checkout", "cancel")


def _required_env_missing(*names: str) -> list[str]:
    values = {
        "STRIPE_SECRET_KEY": STRIPE_SECRET_KEY,
        "STRIPE_WEBHOOK_SECRET": STRIPE_WEBHOOK_SECRET,
        "STRIPE_PRICE_PRO": STRIPE_PRICE_PRO,
        "STRIPE_PRICE_PREMIUM": STRIPE_PRICE_PREMIUM,
        "APP_SUCCESS_URL": APP_SUCCESS_URL,
        "APP_CANCEL_URL": APP_CANCEL_URL,
        "DATABASE_URL": DATABASE_URL,
    }
    return [name for name in names if not values.get(name)]


def _require_env(*names: str) -> None:
    missing = _required_env_missing(*names)
    if missing:
        raise HTTPException(500, f"Missing required billing env vars: {', '.join(missing)}")


def _db_status() -> dict:
    status = {"reachable": False, "error": None}
    if not DATABASE_URL:
        status["error"] = "DATABASE_URL is missing"
        return status
    try:
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        status["reachable"] = True
    except Exception as e:
        status["error"] = str(e)[:200]
    return status


def _record_paid_conversion(email: str, plan: str, *, stripe_event_id: str | None = None) -> None:
    """Best-effort acquisition conversion event. Stores a user hash, never email."""
    try:
        user_hash = hashlib.sha256(str(email or "").strip().lower().encode("utf-8")).hexdigest()[:24]
        metadata = json.dumps({"stripe_event_id": stripe_event_id or "", "billing_surface": "stripe_webhook"})
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS acquisition_events (
                        id SERIAL PRIMARY KEY,
                        event_name TEXT NOT NULL,
                        source TEXT DEFAULT 'direct',
                        utm_source TEXT,
                        utm_medium TEXT,
                        utm_campaign TEXT,
                        utm_content TEXT,
                        utm_term TEXT,
                        referrer_domain TEXT,
                        user_hash TEXT,
                        plan TEXT,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        occurred_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_acquisition_events_name_time "
                    "ON acquisition_events (event_name, occurred_at DESC)"
                )
                cur.execute(
                    """
                    INSERT INTO acquisition_events (event_name, source, user_hash, plan, metadata)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    """,
                    ("successful_paid_conversion", "other_unknown", user_hash, plan, metadata),
                )
            conn.commit()
    except Exception as exc:
        _log.warning("paid conversion analytics failed: %s", type(exc).__name__)


def _normalize_db_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    # If sslmode is already present, keep it.
    if "sslmode=" in u:
        return u
    # Append sslmode=require
    if "?" in u:
        return u + "&sslmode=require"
    return u + "?sslmode=require"

# NOTE: This assumes users.username == email (lowercase).
# If you later decouple username/email, update this to match on a dedicated email column.
def _set_user_plan_by_email(
    *,
    email: str,
    tier: str,
    stripe_customer_id: Optional[str] = None,
    stripe_subscription_id: Optional[str] = None,
    stripe_price_id: Optional[str] = None,
) -> None:
    tier = (tier or "basic").strip().lower()
    if tier not in {"basic", "pro", "premium", "admin"}:
        tier = "basic"

    email_key = (email or "").strip().lower()
    if not email_key:
        raise ValueError("email is required to update plan")

    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET tier = %s,
                    stripe_customer_id = COALESCE(%s, stripe_customer_id),
                    stripe_subscription_id = COALESCE(%s, stripe_subscription_id),
                    stripe_price_id = COALESCE(%s, stripe_price_id),
                    plan_updated_at = %s
                WHERE username = %s
                """,
                (
                    tier,
                    stripe_customer_id,
                    stripe_subscription_id,
                    stripe_price_id,
                    datetime.now(timezone.utc),
                    email_key,
                ),
            )
            if cur.rowcount == 0:
                raise LookupError(f"No user row updated for email={email_key}")
        conn.commit()


def _db_conn():
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL env var")
    try:
        from db.traffic import connect_options, count
    except ImportError:  # standalone Render billing root has no scanner package
        return psycopg2.connect(_normalize_db_url(DATABASE_URL), connect_timeout=8)

    conn = psycopg2.connect(_normalize_db_url(DATABASE_URL), connect_timeout=8, **connect_options("psycopg2"))
    count(connections_opened=1)
    return conn


def _get_user_by_email(email: str) -> dict:
    email_key = (email or "").strip().lower()
    if not email_key:
        return {}
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, tier, stripe_customer_id FROM users WHERE username = %s LIMIT 1",
                (email_key,),
            )
            row = cur.fetchone()
    if not row:
        return {}
    return {"username": row[0], "tier": row[1], "stripe_customer_id": row[2]}


def _get_username_by_customer(customer_id: str) -> Optional[str]:
    """The HSF account that stored this Stripe customer id at checkout, if any."""
    cid = (customer_id or "").strip()
    if not cid:
        return None
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username FROM users WHERE stripe_customer_id = %s LIMIT 1",
                (cid,),
            )
            row = cur.fetchone()
    return (row[0] or "").strip().lower() if row and row[0] else None


def _account_for_customer(customer_id: str, context: str) -> str:
    """HSF username for a subscription webhook (P1-53).

    The Stripe customer id stored at checkout identifies the account; the Stripe
    customer's email can change (e.g. edited in the Billing Portal), so it is only
    a fallback for accounts that never stored a customer id, and never for an
    account already linked to a different customer. Raises HTTPException(500) when
    no account can be resolved, so Stripe retries instead of the event being lost.
    """
    try:
        username = _get_username_by_customer(customer_id)
    except Exception as exc:
        raise HTTPException(500, f"Customer lookup failed for {context}: {type(exc).__name__}")
    if username:
        return username
    try:
        cust = stripe.Customer.retrieve(customer_id)
        email = _validate_email(cust.get("email") or "")
    except Exception as exc:
        raise HTTPException(500, f"Missing customer email for {context}: {exc}")
    try:
        user = _get_user_by_email(email)
    except Exception as exc:
        raise HTTPException(500, f"Account lookup failed for {context}: {type(exc).__name__}")
    linked = (user.get("stripe_customer_id") or "").strip() if user else ""
    if linked and linked != (customer_id or "").strip():
        _log.warning("%s: customer email matches an account linked to a different Stripe customer", context)
        raise HTTPException(500, f"No account is linked to this Stripe customer for {context}")
    return email


def _price_to_plan(price_id: str) -> str:
    price_id = (price_id or "").strip()
    if not price_id:
        return "basic"
    if price_id in {STRIPE_PRICE_PRO, STRIPE_PRICE_PRO_YEARLY} or price_id in STRIPE_PRICE_PRO_LEGACY:
        return "pro"
    if price_id in {STRIPE_PRICE_PREMIUM, STRIPE_PRICE_PREMIUM_YEARLY} or price_id in STRIPE_PRICE_PREMIUM_LEGACY:
        return "premium"
    return "basic"


_billing_engine = None
_reconciliation_thread = None


def start_reconciliation_worker():
    """Hourly recovery for missed deliveries. DB leases prevent duplicate work."""
    import threading

    global _reconciliation_thread
    if os.getenv("BILLING_MODE", "external").strip().lower() != "embedded":
        return
    if os.getenv("BILLING_RECONCILIATION_ENABLED", "1").strip() == "0":
        return
    if _reconciliation_thread and _reconciliation_thread.is_alive():
        return

    def run():
        wake = threading.Event()
        while True:
            try:
                result = billing_engine().reconcile_due()
                _log.info("billing reconciliation: checked=%s failed=%s", result["checked"], result["failed"])
            except Exception as exc:
                _log.warning("billing reconciliation unavailable: %s", type(exc).__name__)
            wake.wait(3600)

    _reconciliation_thread = threading.Thread(target=run, name="billing-reconciliation", daemon=True)
    _reconciliation_thread.start()


def billing_engine():
    global _billing_engine
    if _billing_engine is None:
        import sys
        try:
            from billing_service.engine import Engine
            from billing_service.state import PostgresStore
        except ImportError:  # standalone billing service with a flat Render root
            from engine import Engine
            from state import PostgresStore
        _billing_engine = Engine(sys.modules[__name__], PostgresStore(_db_conn))
    return _billing_engine


def sync_account_subscription(email: str) -> dict:
    return billing_engine().sync(_validate_email(email))


def billing_status(email: str) -> dict:
    return billing_engine().status(_validate_email(email))


def _is_legacy_price(price_id: str) -> bool:
    price_id = (price_id or "").strip()
    return bool(price_id) and (price_id in STRIPE_PRICE_PRO_LEGACY or price_id in STRIPE_PRICE_PREMIUM_LEGACY)


# ---------- Caller authentication (Run 83) ----------
# The Streamlit app mints a single-use, 10-minute "billing" token for the
# signed-in account (ui/auth_tokens.py) and sends it in this header. We consume
# it against the shared database to learn WHICH account is asking, then resolve
# that account's Stripe customer server-side. A client-supplied email or
# customer id is never trusted. Keep the SQL identical to ui/auth_tokens.py.
AUTH_HEADER = "X-HSF-Auth"
_TOKENS_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS hsf_auth_tokens ("
    " token_hash text PRIMARY KEY,"
    " username text NOT NULL,"
    " purpose text NOT NULL,"
    " expires_at timestamptz NOT NULL,"
    " created_at timestamptz NOT NULL DEFAULT now())"
)
_TOKENS_CONSUME_SQL = (
    "DELETE FROM hsf_auth_tokens WHERE token_hash = %s AND purpose = %s AND expires_at > %s "
    "RETURNING username"
)


def _consume_billing_token(token: str) -> Optional[str]:
    """Username for a valid, unused billing token (and invalidate it), else None."""
    raw = (token or "").strip()
    if not raw or len(raw) > 256:
        return None
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_TOKENS_SCHEMA_SQL)
            cur.execute(_TOKENS_CONSUME_SQL, (digest, "billing", datetime.now(timezone.utc)))
            row = cur.fetchone()
    user = (row[0] if row else "") or ""
    return user.strip().lower() or None


def _authenticated_user(request: Request, payload: dict) -> str:
    """The HSF account making this request. Fails closed before any Stripe call."""
    token = request.headers.get(AUTH_HEADER) or ""
    if not token.strip():
        raise HTTPException(401, "Please sign in again to manage billing.")
    try:
        user = _consume_billing_token(token)
    except Exception as exc:
        _log.warning("billing token check failed: %s", type(exc).__name__)
        raise HTTPException(503, "Account verification is temporarily unavailable. Please try again.")
    if not user:
        raise HTTPException(401, "Your sign-in could not be verified. Please sign in again.")
    claimed = (payload.get("email") or "").strip().lower()
    if claimed and claimed != user:
        raise HTTPException(403, "This billing request does not match your account.")
    return user


# ---------- API ----------
def _email_setup() -> dict:
    """P1-39: live-alert email settings on Render, for the Admin page. Only whether
    each setting is present and the sender DOMAIN (visible on every email anyway) —
    never the user, password or full address. Mirrors ui/email_setup.describe_smtp."""
    from email.utils import parseaddr

    names = ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "SMTP_FROM")
    missing = [n for n in names if not os.getenv(n, "").strip()]
    addr = parseaddr(os.getenv("SMTP_FROM", "").strip())[1]
    domain = addr.rpartition("@")[2].lower() if "@" in addr else None
    return {"configured": not missing, "missing": missing, "sender_domain": domain}


@app.get("/healthz")
def healthz():
    """Liveness for the Render health check: the process is up and serving.

    Deliberately no database or Stripe call: Render polls this every few
    seconds, and a DB check there would keep the Neon compute awake 24/7 and
    restart the service on every brief Neon blip. Use /health for the full
    readiness check (env + database)."""
    return {"ok": True}


@app.get("/health")
def health():
    missing = _required_env_missing(
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRICE_PRO",
        "STRIPE_PRICE_PREMIUM",
        "APP_SUCCESS_URL",
        "APP_CANCEL_URL",
        "DATABASE_URL",
    )
    db = _db_status()
    ok = not missing and bool(db["reachable"])
    body = {"ok": ok, "missing_env": missing, "db": db, "email": _email_setup()}
    if not ok:
        return JSONResponse(body, status_code=503)
    body["features"] = ["url_override", "idempotent_qp", "billing_readiness"]
    return body


@app.get("/debug/status")
def debug_status():
    # Run 83: operator-only. Hidden unless BILLING_DEBUG_STATUS=1 on the service.
    if os.getenv("BILLING_DEBUG_STATUS", "").strip() != "1":
        raise HTTPException(404, "Not Found")
    # Do not return secrets; only whether they are set.
    status = {
        "ok": False,
        "env": {
            "STRIPE_SECRET_KEY": bool(STRIPE_SECRET_KEY),
            "STRIPE_WEBHOOK_SECRET": bool(STRIPE_WEBHOOK_SECRET),
            "STRIPE_PRICE_PRO": bool(STRIPE_PRICE_PRO),
            "STRIPE_PRICE_PREMIUM": bool(STRIPE_PRICE_PREMIUM),
            "APP_SUCCESS_URL": bool(APP_SUCCESS_URL),
            "APP_CANCEL_URL": bool(APP_CANCEL_URL),
            "DATABASE_URL": bool(DATABASE_URL),
        },
        "db": _db_status(),
    }
    # Real-time alert worker introspection (enabled/alive/env presence).
    try:
        try:
            from billing_service.realtime_alerts import worker_status
        except ImportError:
            from realtime_alerts import worker_status  # type: ignore
        status["realtime_alerts"] = worker_status()
    except Exception as e:
        status["realtime_alerts"] = {"error": f"{type(e).__name__}: {e}"}
    required = (
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRICE_PRO",
        "STRIPE_PRICE_PREMIUM",
        "APP_SUCCESS_URL",
        "APP_CANCEL_URL",
        "DATABASE_URL",
    )
    status["missing_env"] = _required_env_missing(*required)
    status["billing_preflight"] = _billing_preflight()
    status["ok"] = not status["missing_env"] and bool(status["db"]["reachable"])
    return status


# Go-live pre-flight. Amounts in cents; keep in step with config.TIERS_CONFIG
# (a test checks). A mismatch here means Render points at the wrong price.
_EXPECTED_PRICES = (
    ("STRIPE_PRICE_PRO", "month", 2500),
    ("STRIPE_PRICE_PREMIUM", "month", 4000),
    ("STRIPE_PRICE_PRO_YEARLY", "year", 25000),
    ("STRIPE_PRICE_PREMIUM_YEARLY", "year", 40000),
)
_WEBHOOK_EVENTS = (
    "checkout.session.completed",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.paid",
    "invoice.payment_failed",
)


def _stripe_key_mode(key: str) -> str:
    if key.startswith(("sk_live_", "rk_live_")):
        return "live"
    if key.startswith(("sk_test_", "rk_test_")):
        return "test"
    return "unknown"


def _check_price(price_id: str, mode: str, interval: Optional[str], amount: Optional[int]) -> dict:
    """Problems with one configured price id (empty list = fine). Never raises."""
    out = {"id_set": True, "problems": []}
    try:
        p = stripe.Price.retrieve(price_id)
    except Exception as e:  # wrong mode, typo or deleted: Stripe says "No such price"
        out["problems"].append(f"not found with this key ({type(e).__name__})")
        return out
    recurring = p.get("recurring") or {}
    if mode in ("live", "test") and bool(p.get("livemode")) != (mode == "live"):
        out["problems"].append(f"price is {'live' if p.get('livemode') else 'test'} but key is {mode}")
    if not p.get("active", True):
        out["problems"].append("price is archived")
    if interval and recurring.get("interval") != interval:
        out["problems"].append(f"interval {recurring.get('interval')!r}, expected {interval!r}")
    if amount is not None and p.get("unit_amount") != amount:
        out["problems"].append(f"amount {p.get('unit_amount')}, expected {amount}")
    if (p.get("currency") or "usd") != "usd":
        out["problems"].append(f"currency {p.get('currency')!r}, expected 'usd'")
    return out


def _check_webhooks() -> dict:
    """Is a Stripe webhook endpoint pointed at this service with the
    three events? Restricted keys may not list endpoints; that's reported, not fatal."""
    try:
        endpoints = list(stripe.WebhookEndpoint.list(limit=100).auto_paging_iter())
    except Exception as e:
        return {"checked": False, "problems": [f"could not list endpoints ({type(e).__name__})"]}
    expected_url = os.getenv("BILLING_WEBHOOK_URL", "").strip().rstrip("/")
    expected_path = "/billing/webhook" if os.getenv("BILLING_MODE", "").strip().lower() == "embedded" else "/webhook"
    matching = [ep for ep in endpoints
                if urlparse(str(ep.get("url") or "")).path.rstrip("/") == expected_path
                and (not expected_url or str(ep.get("url") or "").rstrip("/") == expected_url)
                and ep.get("status") == "enabled"]
    if not matching:
        urls = sorted({str(ep.get("url") or "") for ep in endpoints})
        return {"checked": True, "problems": [f"no enabled endpoint at {expected_url or expected_path}"], "urls": urls}
    events = set()
    for ep in matching:
        events.update(ep.get("enabled_events") or [])
    missing = [] if "*" in events else [e for e in _WEBHOOK_EVENTS if e not in events]
    return {"checked": True, "problems": [f"missing events: {', '.join(missing)}"] if missing else [],
            "urls": sorted(str(ep.get("url")) for ep in matching)}


def _billing_preflight() -> dict:
    """What would break a real checkout: key mode, webhook secret, each price id
    (exists in this key's mode, active, right interval and amount) and the
    webhook endpoint. Reads Stripe only; never returns key or price values."""
    mode = _stripe_key_mode(STRIPE_SECRET_KEY)
    problems = []
    if mode == "unknown":
        problems.append("STRIPE_SECRET_KEY is not an sk_/rk_ live or test key")
    if STRIPE_WEBHOOK_SECRET and not STRIPE_WEBHOOK_SECRET.startswith("whsec_"):
        problems.append("STRIPE_WEBHOOK_SECRET does not start with whsec_")
    prices = {}
    for name, interval, amount in _EXPECTED_PRICES:
        price_id = os.getenv(name, "").strip()
        if not price_id:
            prices[name] = {"id_set": False, "problems": []}
            if not name.endswith("_YEARLY"):
                problems.append(f"{name} not set")
            continue
        prices[name] = _check_price(price_id, mode, interval, amount)
        problems += [f"{name}: {p}" for p in prices[name]["problems"]]
    for name in ("STRIPE_PRICE_PRO_LEGACY", "STRIPE_PRICE_PREMIUM_LEGACY"):
        for i, price_id in enumerate(sorted(_price_ids(name))):
            key = f"{name}[{i}]"
            # Legacy prices may be archived and are old amounts: only mode matters.
            prices[key] = _check_price(price_id, mode, None, None)
            prices[key]["problems"] = [p for p in prices[key]["problems"] if p != "price is archived"]
            problems += [f"{key}: {p}" for p in prices[key]["problems"]]
    yearly = [bool(os.getenv(n, "").strip()) for n in ("STRIPE_PRICE_PRO_YEARLY", "STRIPE_PRICE_PREMIUM_YEARLY")]
    if any(yearly) and not all(yearly):
        problems.append("only one yearly price is set; set both or neither")
    webhooks = _check_webhooks()
    # An endpoint we couldn't list is a warning (check it by hand), not a failure.
    warnings = [] if webhooks["checked"] else [f"webhook: {p}" for p in webhooks["problems"]]
    if webhooks["checked"]:
        problems += [f"webhook: {p}" for p in webhooks["problems"]]
    return {"ok": not problems, "stripe_mode": mode, "problems": problems,
            "warnings": warnings, "prices": prices, "webhooks": webhooks}


def _plan_change_flow(subscription, price_id: str, return_url: str, *, interval: str = "month") -> Optional[dict]:
    """Stripe portal flow_data that confirms switching `subscription` to `price_id`
    and redirects to `return_url` afterwards; None when there's nothing to switch
    (no item, unknown ids, or already on that price)."""
    try:
        items = ((subscription.get("items") or {}).get("data") or [])
        item = items[0] if items else {}
        current = (item.get("price") or {}).get("id")
        if not subscription.get("id") or not item.get("id") or not price_id or current == price_id:
            return None
        # Already on this plan at an older (grandfathered) price and asking for the
        # same plan monthly: don't offer a "switch" onto the new, higher price.
        # Switching between monthly and yearly on the same plan stays allowed.
        same_plan = _price_to_plan(current) != "basic" and _price_to_plan(current) == _price_to_plan(price_id)
        if same_plan and _is_legacy_price(current) and interval != "year":
            return None
        return {
            "type": "subscription_update_confirm",
            "subscription_update_confirm": {
                "subscription": subscription["id"],
                "items": [{"id": item["id"], "price": price_id, "quantity": 1}],
            },
            "after_completion": {"type": "redirect", "redirect": {"return_url": return_url}},
        }
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def _subscription_cancel_flow(subscription_id: str, return_url: str) -> Optional[dict]:
    """Stripe portal flow that confirms cancellation, then returns to HSF."""
    if not subscription_id or not return_url:
        return None
    return {
        "type": "subscription_cancel",
        "subscription_cancel": {"subscription": subscription_id},
        "after_completion": {"type": "redirect", "redirect": {"return_url": return_url}},
    }


def _founder_alumni_coupon(email: str) -> Optional[str]:
    """Founding members whose five years ended get the founder rate (a Stripe coupon)."""
    try:
        try:
            from billing_service.founding_checkout import alumni_coupon
        except ImportError:  # standalone Render root
            from founding_checkout import alumni_coupon  # type: ignore
        return alumni_coupon(email)
    except Exception:
        return None


@app.post("/create-checkout-session")
def create_checkout_session(payload: dict, request: Request):
    _require_env("STRIPE_PRICE_PRO", "STRIPE_PRICE_PREMIUM", "APP_SUCCESS_URL", "APP_CANCEL_URL", "DATABASE_URL")
    return checkout_for_account(_authenticated_user(request, payload), payload)


def checkout_for_account(email: str, payload: dict):
    _require_env("STRIPE_PRICE_PRO", "STRIPE_PRICE_PREMIUM", "APP_SUCCESS_URL", "APP_CANCEL_URL", "DATABASE_URL")
    if payload.get("interval") == "year" and not (
        STRIPE_PRICE_PRO_YEARLY if payload.get("plan") == "pro" else STRIPE_PRICE_PREMIUM_YEARLY
    ):
        raise HTTPException(400, "Yearly billing isn't available yet.")
    try:
        return billing_engine().checkout(_validate_email(email), payload)
    except LookupError as exc:
        raise HTTPException(404, "No app user exists for this email. Please sign up before upgrading.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, "Checkout is temporarily unavailable. Please try again.") from exc


def _create_checkout_for_account(email: str, payload: dict, *, user=None, idempotency_key=None, expires_at=None):
    """
    Called only after the HTTP adapter or API has authenticated the account.

    Payload example:
    {
      "email": "user@email.com",
      "plan": "pro" | "premium",
      "interval": "month" | "year"   (optional; yearly needs STRIPE_PRICE_*_YEARLY)
    }
    """
    _require_env(
        "STRIPE_PRICE_PRO",
        "STRIPE_PRICE_PREMIUM",
        "APP_SUCCESS_URL",
        "APP_CANCEL_URL",
        "DATABASE_URL",
    )

    email = _validate_email(email)
    plan = (payload.get("plan") or "").strip().lower()

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email is required")

    if plan not in {"pro", "premium"}:
        raise HTTPException(400, "Plan must be 'pro' or 'premium'")

    interval = (payload.get("interval") or "month").strip().lower()
    if interval not in {"month", "year"}:
        raise HTTPException(400, "Interval must be 'month' or 'year'")

    # Optional success_url / return_url overrides — only honored if they point at
    # one of our apps (see _allowed_return_url; prevents an open redirect).
    success_url = _append_qp(APP_SUCCESS_URL, "checkout", "success")
    success_override = _allowed_return_url((payload.get("success_url") or "").strip())
    if success_override:
        success_url = success_override

    portal_return_url = _append_qp(APP_PORTAL_RETURN_URL or APP_SUCCESS_URL, "portal", "return")
    return_override = _allowed_return_url((payload.get("return_url") or "").strip())
    if return_override:
        portal_return_url = return_override

    if interval == "year":
        price_id = STRIPE_PRICE_PRO_YEARLY if plan == "pro" else STRIPE_PRICE_PREMIUM_YEARLY
        if not price_id:
            raise HTTPException(400, "Yearly billing isn't available yet.")
    else:
        price_id = STRIPE_PRICE_PRO if plan == "pro" else STRIPE_PRICE_PREMIUM

    if user is None:
        try:
            user = _get_user_by_email(email)
        except Exception as e:
            _log.warning("checkout user lookup failed: %s", type(e).__name__)
            raise HTTPException(503, "Account lookup is temporarily unavailable. Please try again.")
    if not user:
        raise HTTPException(404, "No app user exists for this email. Please sign up before upgrading.")
    customer_id = user.get("stripe_customer_id") if user else None

    # If this customer already has an active subscription, DO NOT create a second subscription.
    # Send them to Stripe Billing Portal to upgrade/downgrade/cancel on the existing subscription.
    if customer_id:
        try:
            subs = stripe.Subscription.list(customer=customer_id, status="all", limit=100)
            if subs.get("has_more"):
                raise HTTPException(409, "Subscription history needs review before changing plans.")
            current = [sub for sub in subs.get("data", []) if sub.get("status") not in {"canceled", "incomplete_expired"}]
            if current:
                # P2-48: open Stripe's "confirm this plan change" screen for the
                # requested plan and send the customer straight back to the app
                # when they confirm (no "Return to site" click). Any problem ->
                # the plain portal, exactly as before.
                portal = None
                flow = _plan_change_flow(current[0], price_id, portal_return_url, interval=interval)
                if flow:
                    try:
                        portal = stripe.billing_portal.Session.create(
                            customer=customer_id,
                            return_url=portal_return_url,
                            flow_data=flow,
                        )
                    except Exception as e:
                        _log.warning("plan-change portal flow failed, using plain portal: %s", type(e).__name__)
                        portal = None
                if portal is None:
                    portal = stripe.billing_portal.Session.create(
                        customer=customer_id,
                        return_url=portal_return_url,
                    )
                return {"portal_url": portal.url, "mode": "portal"}
        except Exception as e:
            # A failed lookup or portal must never create a second subscription.
            _log.warning("existing subscription check/portal failed: %s", type(e).__name__)
            raise HTTPException(502, "Subscription management is temporarily unavailable. Please try again.")

    try:
        options = {}
        if idempotency_key:
            options["idempotency_key"] = idempotency_key
        if expires_at:
            options["expires_at"] = expires_at
        # The founder rate is for yearly plans only.
        coupon = _founder_alumni_coupon(email) if interval == "year" else None
        if coupon:  # Stripe allows a coupon or promotion codes, not both
            options["discounts"] = [{"coupon": coupon}]
        else:
            options["allow_promotion_codes"] = True
        session = stripe.checkout.Session.create(
            **options,
            mode="subscription",
            client_reference_id=hashlib.sha256(email.encode()).hexdigest(),
            customer=customer_id or None,
            customer_email=None if customer_id else email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=_cancel_url_for(success_url),
            metadata={
                "user_email": email,
                "requested_plan": plan,
                "interval": interval,
            },
        )
        session_id = getattr(session, "id", None)
        return {"checkout_url": session.url, "mode": "checkout", "session_id": session_id if isinstance(session_id, str) else None}
    except Exception as e:
        # Surface the message so curl shows something useful.
        _log.warning("create-checkout-session failed: %s", type(e).__name__)
        raise HTTPException(500, "Checkout is temporarily unavailable. Please try again.")


@app.post("/create-portal-session")
def create_portal_session(payload: dict, request: Request):
    return portal_for_account(_authenticated_user(request, payload), payload)


def portal_for_account(email: str, payload: dict):
    """
    Called only after the HTTP adapter or API has authenticated the account.
    Optional payload: { "return_url": "<URL on one of our apps>", "flow": "cancel" }.
    Any "email" must match the authenticated account; the Stripe customer
    always comes from that account's own users row, never from the request.
    """
    _require_env("APP_SUCCESS_URL", "DATABASE_URL")
    email = _validate_email(email)
    try:
        user = _get_user_by_email(email)
    except Exception as e:
        _log.warning("portal user lookup failed: %s", type(e).__name__)
        raise HTTPException(503, "Account lookup is temporarily unavailable. Please try again.")
    customer_id = (user or {}).get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(400, "No subscription found for this account yet.")

    flow = (payload.get("flow") or "").strip().lower()
    if flow not in {"", "cancel"}:
        raise HTTPException(400, "Unsupported billing portal flow.")

    return_url = _append_qp(APP_PORTAL_RETURN_URL or APP_SUCCESS_URL, "portal", "return")
    override = _allowed_return_url((payload.get("return_url") or "").strip())
    if override:
        return_url = override

    try:
        portal_kwargs = {"customer": customer_id, "return_url": return_url}
        if flow == "cancel":
            subscriptions = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
            active = (subscriptions or {}).get("data") or []
            subscription_id = (active[0] or {}).get("id") if active else None
            flow_data = _subscription_cancel_flow(subscription_id, return_url)
            if not flow_data:
                raise HTTPException(400, "No active subscription is available to cancel.")
            portal_kwargs["flow_data"] = flow_data
        portal = stripe.billing_portal.Session.create(**portal_kwargs)
        return {"portal_url": portal.url}
    except HTTPException:
        raise
    except Exception as e:
        _log.warning("portal session create failed: %s", type(e).__name__)
        raise HTTPException(502, "The billing portal is temporarily unavailable. Please try again.")


@app.post("/webhook")
async def stripe_webhook(request: Request):
    # Stripe and psycopg2 are synchronous; keep them off the shared API event loop.
    payload = await request.body()
    return await run_in_threadpool(process_webhook, payload, request.headers.get("stripe-signature"))


def process_webhook(payload: bytes, sig: Optional[str]):
    _require_env("STRIPE_WEBHOOK_SECRET", "DATABASE_URL")
    return billing_engine().webhook(payload, sig)
