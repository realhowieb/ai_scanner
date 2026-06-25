import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import stripe

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_log = logging.getLogger("billing_service")


def _validate_email(email: str) -> str:
    """Normalize and validate an email from a Stripe webhook. Raises ValueError if invalid."""
    email = (email or "").strip().lower()
    if not email or not _EMAIL_RE.match(email):
        raise ValueError(f"Invalid or missing email from webhook: {email!r}")
    return email

# ---------- DB helpers ----------

def _append_qp(url: str, key: str, value: str) -> str:
    base = (url or "").strip() or "https://example.com"
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{key}={value}"
import psycopg2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ---------- ENV ----------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()

STRIPE_PRICE_PRO = os.getenv("STRIPE_PRICE_PRO", "").strip()
STRIPE_PRICE_PREMIUM = os.getenv("STRIPE_PRICE_PREMIUM", "").strip()

APP_SUCCESS_URL = os.getenv("APP_SUCCESS_URL", "").strip()  # e.g. https://yourapp.com
APP_CANCEL_URL = os.getenv("APP_CANCEL_URL", "").strip()    # e.g. https://yourapp.com
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()        # Neon Postgres URL
APP_PORTAL_RETURN_URL = os.getenv("APP_PORTAL_RETURN_URL", "").strip()  # e.g. https://hsf-beta.streamlit.app/billing

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
                print(f"[billing_service][WARN] No user row updated for email={email_key}")
        conn.commit()


def _db_conn():
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL env var")
    return psycopg2.connect(_normalize_db_url(DATABASE_URL), connect_timeout=8)


def _get_user_by_email(email: str) -> dict:
    email_key = (email or "").strip().lower()
    if not email_key:
        return {}
    try:
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
    except Exception as e:
        # DB issues should not block checkout creation.
        print(f"[billing_service] DB lookup failed for {email_key}: {e}")
        return {}


def _price_to_plan(price_id: str) -> str:
    if price_id == STRIPE_PRICE_PRO:
        return "pro"
    if price_id == STRIPE_PRICE_PREMIUM:
        return "premium"
    return "basic"


# ---------- API ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug/status")
def debug_status():
    # Do not return secrets; only whether they are set.
    status = {
        "ok": True,
        "env": {
            "STRIPE_SECRET_KEY": bool(STRIPE_SECRET_KEY),
            "STRIPE_WEBHOOK_SECRET": bool(STRIPE_WEBHOOK_SECRET),
            "STRIPE_PRICE_PRO": bool(STRIPE_PRICE_PRO),
            "STRIPE_PRICE_PREMIUM": bool(STRIPE_PRICE_PREMIUM),
            "APP_SUCCESS_URL": bool(APP_SUCCESS_URL),
            "APP_CANCEL_URL": bool(APP_CANCEL_URL),
            "DATABASE_URL": bool(DATABASE_URL),
        },
        "db": {"reachable": False, "error": None},
    }

    if DATABASE_URL:
        try:
            with _db_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            status["db"]["reachable"] = True
        except Exception as e:
            status["db"]["error"] = str(e)[:200]

    return status


@app.post("/create-checkout-session")
async def create_checkout_session(payload: dict):
    """
    Payload example:
    {
      "email": "user@email.com",
      "plan": "pro" | "premium"
    }
    """
    if not STRIPE_PRICE_PRO or not STRIPE_PRICE_PREMIUM:
        raise HTTPException(500, "Missing STRIPE_PRICE_PRO/STRIPE_PRICE_PREMIUM env vars")

    email = (payload.get("email") or "").strip().lower()
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

    success_url = _append_qp(APP_SUCCESS_URL or "https://example.com", "checkout", "success")
    success_override = _validate_same_host((payload.get("success_url") or "").strip())
    if success_override:
        success_url = success_override

    portal_return_url = _append_qp(
        APP_PORTAL_RETURN_URL or APP_SUCCESS_URL or "https://example.com", "portal", "return"
    )
    return_override = _validate_same_host((payload.get("return_url") or "").strip())
    if return_override:
        portal_return_url = return_override

    price_id = STRIPE_PRICE_PRO if plan == "pro" else STRIPE_PRICE_PREMIUM

    # Use existing customer if we have it
    user = _get_user_by_email(email)  # safe: returns {} on DB failure
    customer_id = user.get("stripe_customer_id") if user else None

    # If this customer already has an active subscription, DO NOT create a second subscription.
    # Send them to Stripe Billing Portal to upgrade/downgrade/cancel on the existing subscription.
    if customer_id:
        try:
            subs = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
            if subs and subs.get("data"):
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
            cancel_url=_append_qp(APP_CANCEL_URL or "https://example.com", "checkout", "cancel"),
            metadata={
                "user_email": email,
                "requested_plan": plan,
            },
        )
        return {"checkout_url": session.url, "mode": "checkout"}
    except Exception as e:
        # Surface the message so curl shows something useful.
        print(f"[billing_service] create-checkout-session failed: {e}")
        raise HTTPException(500, f"Create checkout session failed: {e}")


@app.post("/create-portal-session")
async def create_portal_session(payload: dict):
    """
    Payload example:
    { "email": "user@email.com" }
    """
    email = (payload.get("email") or "").strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Valid email is required")

    user = _get_user_by_email(email)
    customer_id = user.get("stripe_customer_id")
    if not customer_id:
        raise HTTPException(400, "No Stripe customer found for this user yet")

    return_url = _append_qp(
        APP_PORTAL_RETURN_URL or APP_SUCCESS_URL or "https://example.com", "portal", "return"
    )
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
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return {"portal_url": portal.url}
    except Exception as e:
        raise HTTPException(500, f"Stripe error: {e}")


@app.post("/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(500, "Missing STRIPE_WEBHOOK_SECRET env var")

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
            return JSONResponse({"received": True, "type": etype, "note": str(exc)})
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

        try:
            _set_user_plan_by_email(
                email=email,
                tier=plan,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                stripe_price_id=price_id,
            )
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
            email = (cust.get("email") or "").strip().lower()
        except Exception:
            email = ""

        if not email:
            return JSONResponse({"received": True, "type": etype, "note": "missing customer email"})

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
        except Exception as e:
            raise HTTPException(500, f"DB update failed: {e}")

    # 3) Subscription cancelled → downgrade to Basic (unless admin)
    elif etype == "customer.subscription.deleted":
        customer_id = data.get("customer")
        try:
            cust = stripe.Customer.retrieve(customer_id)
            email = (cust.get("email") or "").strip().lower()
        except Exception:
            email = ""

        if email:
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