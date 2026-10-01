import hashlib
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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

APP_SUCCESS_URL = os.getenv("APP_SUCCESS_URL", "").strip()  # e.g. https://yourapp.com
APP_CANCEL_URL = os.getenv("APP_CANCEL_URL", "").strip()    # e.g. https://yourapp.com
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()        # Neon Postgres URL
APP_PORTAL_RETURN_URL = os.getenv("APP_PORTAL_RETURN_URL", "").strip()  # e.g. https://hsfinestai.streamlit.app/billing

if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY env var")

stripe.api_key = STRIPE_SECRET_KEY
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


def _ensure_processed_events_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stripe_processed_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                processed_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
    conn.commit()


def _is_event_processed(conn, event_id: str) -> bool:
    """Return True if this Stripe event_id was already handled (idempotency guard)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM stripe_processed_events WHERE event_id = %s LIMIT 1",
            (event_id,),
        )
        return cur.fetchone() is not None


def _mark_event_processed(conn, event_id: str, event_type: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stripe_processed_events (event_id, event_type)
            VALUES (%s, %s)
            ON CONFLICT (event_id) DO NOTHING
            """,
            (event_id, event_type),
        )
    conn.commit()


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
    return psycopg2.connect(_normalize_db_url(DATABASE_URL), connect_timeout=8)


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


def _price_to_plan(price_id: str) -> str:
    if price_id == STRIPE_PRICE_PRO:
        return "pro"
    if price_id == STRIPE_PRICE_PREMIUM:
        return "premium"
    return "basic"


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
    status["ok"] = not status["missing_env"] and bool(status["db"]["reachable"])
    return status


def _plan_change_flow(subscription, price_id: str, return_url: str) -> Optional[dict]:
    """Stripe portal flow_data that confirms switching `subscription` to `price_id`
    and redirects to `return_url` afterwards; None when there's nothing to switch
    (no item, unknown ids, or already on that price)."""
    try:
        items = ((subscription.get("items") or {}).get("data") or [])
        item = items[0] if items else {}
        current = (item.get("price") or {}).get("id")
        if not subscription.get("id") or not item.get("id") or not price_id or current == price_id:
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


@app.post("/create-checkout-session")
async def create_checkout_session(payload: dict, request: Request):
    """
    Payload example:
    {
      "email": "user@email.com",
      "plan": "pro" | "premium"
    }
    """
    _require_env(
        "STRIPE_PRICE_PRO",
        "STRIPE_PRICE_PREMIUM",
        "APP_SUCCESS_URL",
        "APP_CANCEL_URL",
        "DATABASE_URL",
    )

    email = _authenticated_user(request, payload)
    plan = (payload.get("plan") or "").strip().lower()

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email is required")

    if plan not in {"pro", "premium"}:
        raise HTTPException(400, "Plan must be 'pro' or 'premium'")

    # Optional success_url / return_url overrides — only honored if they point at
    # the same host as APP_SUCCESS_URL (prevents this becoming an open redirect).
    def _validate_same_host(candidate: str) -> str | None:
        try:
            from urllib.parse import urlparse
            base_host = urlparse(APP_SUCCESS_URL or "").netloc
            if base_host and urlparse(candidate).netloc == base_host:
                return candidate
        except Exception:
            pass
        return None

    success_url = _append_qp(APP_SUCCESS_URL, "checkout", "success")
    success_override = _validate_same_host((payload.get("success_url") or "").strip())
    if success_override:
        success_url = success_override

    portal_return_url = _append_qp(APP_PORTAL_RETURN_URL or APP_SUCCESS_URL, "portal", "return")
    return_override = _validate_same_host((payload.get("return_url") or "").strip())
    if return_override:
        portal_return_url = return_override

    price_id = STRIPE_PRICE_PRO if plan == "pro" else STRIPE_PRICE_PREMIUM

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
            subs = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
            if subs and subs.get("data"):
                # P2-48: open Stripe's "confirm this plan change" screen for the
                # requested plan and send the customer straight back to the app
                # when they confirm (no "Return to site" click). Any problem ->
                # the plain portal, exactly as before.
                portal = None
                flow = _plan_change_flow(subs["data"][0], price_id, portal_return_url)
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
            # If portal creation fails for any reason, fall back to Checkout (but only if needed)
            print(f"[billing_service] portal redirect failed (falling back to checkout): {e}")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id or None,
            customer_email=None if customer_id else email,
            line_items=[{"price": price_id, "quantity": 1}],
            allow_promotion_codes=True,
            success_url=success_url,
            cancel_url=_append_qp(APP_CANCEL_URL, "checkout", "cancel"),
            metadata={
                "user_email": email,
                "requested_plan": plan,
            },
        )
        return {"checkout_url": session.url, "mode": "checkout"}
    except Exception as e:
        # Surface the message so curl shows something useful.
        _log.warning("create-checkout-session failed: %s", type(e).__name__)
        raise HTTPException(500, "Checkout is temporarily unavailable. Please try again.")


@app.post("/create-portal-session")
async def create_portal_session(payload: dict, request: Request):
    """
    Requires the X-HSF-Auth billing token of the signed-in account (Run 83).
    Optional payload: { "return_url": "<same-host URL>", "flow": "cancel" }.
    Any "email" must match the authenticated account; the Stripe customer
    always comes from that account's own users row, never from the request.
    """
    _require_env("APP_SUCCESS_URL", "DATABASE_URL")
    email = _authenticated_user(request, payload)
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
    override = (payload.get("return_url") or "").strip()
    if override:
        try:
            from urllib.parse import urlparse
            base_host = urlparse(APP_SUCCESS_URL or "").netloc
            if base_host and urlparse(override).netloc == base_host:
                return_url = override
        except Exception:
            pass

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
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(500, "Missing STRIPE_WEBHOOK_SECRET env var")
    _require_env("DATABASE_URL")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig, secret=STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Webhook signature verification failed: {e}")

    etype = event.get("type")
    event_id = event.get("id", "")
    data = event.get("data", {}).get("object", {})

    # Idempotency: skip already-processed events (Stripe retries on non-2xx).
    if DATABASE_URL and event_id:
        try:
            with _db_conn() as conn:
                _ensure_processed_events_table(conn)
                if _is_event_processed(conn, event_id):
                    _log.info("Skipping already-processed event %s (%s)", event_id, etype)
                    return JSONResponse({"received": True, "type": etype, "note": "already_processed"})
        except Exception as _idem_err:
            _log.warning("Idempotency check failed for %s: %s", event_id, _idem_err)

    # 1) Initial checkout completion
    if etype == "checkout.session.completed":
        raw_email = (
            (data.get("metadata", {}) or {}).get("user_email")
            or (data.get("customer_details") or {}).get("email")
            or data.get("customer_email")
            or ""
        )
        try:
            email = _validate_email(raw_email)
        except ValueError as exc:
            _log.warning("checkout.session.completed: %s", exc)
            raise HTTPException(500, str(exc))
        customer_id = data.get("customer")
        subscription_id = data.get("subscription")

        # Find the price from the subscription
        try:
            sub = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
            price_id = sub["items"]["data"][0]["price"]["id"]
            plan = _price_to_plan(price_id)
        except Exception:
            price_id = None
            plan = (data.get("metadata", {}) or {}).get("requested_plan") or "basic"
            plan = plan.strip().lower()
        if plan not in {"pro", "premium"}:
            raise HTTPException(500, f"Could not determine paid plan for checkout session {data.get('id')}")

        try:
            _set_user_plan_by_email(
                email=email,
                tier=plan,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                stripe_price_id=price_id,
            )
            _record_paid_conversion(email, plan, stripe_event_id=event_id)
        except Exception as e:
            raise HTTPException(500, f"DB update failed: {e}")

    # 2) Subscription updates (upgrade/downgrade, cancellation scheduling, etc.)
    elif etype == "customer.subscription.updated":
        subscription_id = data.get("id")
        customer_id = data.get("customer")

        status = (data.get("status") or "").strip().lower()
        cancel_at_period_end = bool(data.get("cancel_at_period_end"))

        # We need user_email; safest is to look it up from Stripe customer email
        try:
            cust = stripe.Customer.retrieve(customer_id)
            email = _validate_email(cust.get("email") or "")
        except Exception as exc:
            raise HTTPException(500, f"Missing customer email for subscription update: {exc}")

        if not email:
            raise HTTPException(500, "Missing customer email for subscription update")

        # Immediate cancellation -> downgrade to basic now
        if status == "canceled":
            try:
                _set_user_plan_by_email(
                    email=email,
                    tier="basic",
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                    stripe_price_id=None,
                )
            except Exception as e:
                raise HTTPException(500, f"DB update failed: {e}")
            return JSONResponse({"received": True, "type": etype, "action": "downgraded_basic_immediate_cancel"})

        # Cancel scheduled at period end -> keep current tier until Stripe sends subscription.deleted
        if cancel_at_period_end:
            return JSONResponse({"received": True, "type": etype, "action": "cancel_scheduled_keep_tier"})

        # Otherwise: active subscription update -> map price to plan
        price_id = None
        try:
            items = data.get("items", {}).get("data", [])
            if items:
                price_id = items[0].get("price", {}).get("id")
        except Exception:
            price_id = None

        plan = _price_to_plan(price_id or "")
        try:
            _set_user_plan_by_email(
                email=email,
                tier=plan,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                stripe_price_id=price_id,
            )
            _record_paid_conversion(email, plan, stripe_event_id=event_id)
        except Exception as e:
            raise HTTPException(500, f"DB update failed: {e}")

    # 3) Subscription cancelled → downgrade to Basic (unless admin)
    elif etype == "customer.subscription.deleted":
        customer_id = data.get("customer")
        try:
            cust = stripe.Customer.retrieve(customer_id)
            email = _validate_email(cust.get("email") or "")
        except Exception as exc:
            raise HTTPException(500, f"Missing customer email for subscription deletion: {exc}")

        if not email:
            raise HTTPException(500, "Missing customer email for subscription deletion")
        try:
            _set_user_plan_by_email(
                email=email,
                tier="basic",
                stripe_customer_id=customer_id,
                stripe_subscription_id=data.get("id"),
                stripe_price_id=None,
            )
        except Exception as e:
            raise HTTPException(500, f"DB update failed: {e}")

    # Mark the event processed so retries are skipped.
    if DATABASE_URL and event_id:
        try:
            with _db_conn() as conn:
                _mark_event_processed(conn, event_id, etype or "")
        except Exception as _mark_err:
            _log.warning("Failed to mark event %s processed: %s", event_id, _mark_err)

    return JSONResponse({"received": True, "type": etype})
