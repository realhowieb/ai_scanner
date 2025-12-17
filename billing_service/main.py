import os
import json
from datetime import datetime, timezone
from typing import Optional

import stripe
import psycopg2
from fastapi import FastAPI, Request, HTTPException
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

if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY env var")

stripe.api_key = STRIPE_SECRET_KEY


# ---------- DB helpers ----------
def _db_conn():
    if not DATABASE_URL:
        raise RuntimeError("Missing DATABASE_URL env var")
    return psycopg2.connect(DATABASE_URL)


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
        conn.commit()


def _get_user_by_email(email: str) -> dict:
    email_key = (email or "").strip().lower()
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


# ---------- API ----------
@app.get("/health")
def health():
    return {"ok": True}


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

    price_id = STRIPE_PRICE_PRO if plan == "pro" else STRIPE_PRICE_PREMIUM

    # Use existing customer if we have it
    user = _get_user_by_email(email)
    customer_id = user.get("stripe_customer_id") if user else None

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id or None,
            customer_email=None if customer_id else email,
            line_items=[{"price": price_id, "quantity": 1}],
            allow_promotion_codes=True,
            success_url=(APP_SUCCESS_URL or "https://example.com") + "?checkout=success",
            cancel_url=(APP_CANCEL_URL or "https://example.com") + "?checkout=cancel",
            metadata={
                "user_email": email,
                "requested_plan": plan,
            },
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(500, f"Stripe error: {e}")


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

    try:
        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=(APP_SUCCESS_URL or "https://example.com") + "?portal=return",
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
    data = event.get("data", {}).get("object", {})

    # 1) Initial checkout completion
    if etype == "checkout.session.completed":
        email = (data.get("metadata", {}) or {}).get("user_email") or ""
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

    # 2) Subscription updates (upgrade/downgrade, payment issues, etc.)
    elif etype == "customer.subscription.updated":
        subscription_id = data.get("id")
        customer_id = data.get("customer")
        price_id = None
        try:
            items = data.get("items", {}).get("data", [])
            if items:
                price_id = items[0].get("price", {}).get("id")
        except Exception:
            price_id = None

        # We need user_email; safest is to look it up from Stripe customer email
        try:
            cust = stripe.Customer.retrieve(customer_id)
            email = (cust.get("email") or "").strip().lower()
        except Exception:
            email = ""

        if email:
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

    return JSONResponse({"received": True, "type": etype})