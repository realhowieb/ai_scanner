"""Alerts in Telegram and Discord, next to email and browser push (Pro and above).

Telegram: the owner creates a bot with @BotFather and sets TELEGRAM_BOT_TOKEN,
TELEGRAM_BOT_USERNAME and TELEGRAM_WEBHOOK_SECRET on the API. On each start the API points
the bot at its own /v1/telegram/webhook (register_telegram_webhook; the URL comes from
HSF_PUBLIC_API_URL or Render's RENDER_EXTERNAL_URL), so no manual step is needed;
scripts/telegram_set_webhook.py does the same by hand. A user taps "Connect Telegram", which opens
t.me/<bot>?start=<one-time code>; Telegram posts that /start to /v1/telegram/webhook
and the chat is linked. /stop in the chat unlinks it.

Discord: the user pastes a channel webhook URL (Channel settings -> Integrations ->
Webhooks). Nothing to set up on the server.

Every alert that fires (price alerts and alert rules, through api.webpush.alert_fired) is
sent to each linked chat, except during the account's quiet hours. Best effort: a failed
chat message never affects the alert or the other channels. Destinations are never
returned in full by the API.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import secrets
from typing import Any, Dict, List, Optional

from api.store import _conn
from db.engine import schema_once

log = logging.getLogger("hsf_api")

KINDS = ("telegram", "discord")
LINK_TTL_MIN = 15
DISCORD_URL = re.compile(r"^https://(?:canary\.|ptb\.)?(?:discord\.com|discordapp\.com)/api/webhooks/\d{5,30}/[\w-]{20,200}$")
TELEGRAM_API = "https://api.telegram.org"
TIMEOUT_S = 5.0


class ChatError(ValueError):
    """Bad input from the user (400)."""


def telegram_ready() -> bool:
    return all(os.environ.get(k, "").strip() for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_USERNAME",
                                                        "TELEGRAM_WEBHOOK_SECRET"))


def enabled() -> bool:
    return os.environ.get("HSF_CHAT_ALERTS_ENABLED", "1").strip() != "0"


# ---- storage ----------------------------------------------------------------------------------------
@schema_once
def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("hsf_schema_chat_destinations",))
        cur.execute("""CREATE TABLE IF NOT EXISTS hsf_chat_destinations (
            user_id TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
            kind TEXT NOT NULL, target TEXT NOT NULL, label TEXT,
            last_status TEXT, last_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY (user_id, kind))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS hsf_chat_link_codes (
            code TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
            expires_at TIMESTAMPTZ NOT NULL)""")
    conn.commit()


def _run(sql: str, args: tuple, fetch: bool = False) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(sql, args)
            rows: List[Dict[str, Any]] = []
            if fetch and cur.description:
                cols = [d.name for d in cur.description]
                rows = [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in cur.fetchall()]
        conn.commit()
        return rows
    finally:
        conn.close()


def destinations(user: str) -> List[Dict[str, Any]]:
    return _run("SELECT kind, target, label, last_status, last_at FROM hsf_chat_destinations WHERE user_id = %s",
                (user,), True)


def _save(user: str, kind: str, target: str, label: Optional[str]) -> None:
    _run("""INSERT INTO hsf_chat_destinations (user_id, kind, target, label) VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, kind) DO UPDATE SET target = EXCLUDED.target, label = EXCLUDED.label,
            last_status = NULL, last_at = NULL, created_at = NOW()""", (user, kind, target, label))


def remove(user: str, kind: str) -> bool:
    return bool(_run("DELETE FROM hsf_chat_destinations WHERE user_id = %s AND kind = %s RETURNING kind",
                     (user, kind), True))


def _mark(user: str, kind: str, status: str) -> None:
    try:
        _run("UPDATE hsf_chat_destinations SET last_status = %s, last_at = NOW() WHERE user_id = %s AND kind = %s",
             (status, user, kind))
    except Exception:
        pass


def summary(user: str) -> Dict[str, Any]:
    """What the Alerts page shows: per kind, whether it can be used and is connected."""
    rows = {r["kind"]: r for r in destinations(user)}

    def one(kind: str, available: bool) -> Dict[str, Any]:
        r = rows.get(kind)
        when = r.get("last_at") if r else None
        return {"available": available and enabled(), "connected": bool(r), "label": (r or {}).get("label"),
                "last_status": (r or {}).get("last_status"),
                "last_at": when.isoformat() if isinstance(when, (dt.datetime, dt.date)) else when}

    return {"telegram": one("telegram", telegram_ready()), "discord": one("discord", True)}


# ---- linking -----------------------------------------------------------------------------------------
def telegram_link(user: str) -> Dict[str, Any]:
    """A one-time deep link that connects the chat it's opened in to this account."""
    if not telegram_ready():
        raise ChatError("Telegram alerts aren't set up on the server yet.")
    code = secrets.token_urlsafe(18)
    _run("DELETE FROM hsf_chat_link_codes WHERE user_id = %s OR expires_at < NOW()", (user,))
    _run("INSERT INTO hsf_chat_link_codes (code, user_id, expires_at) VALUES (%s, %s, NOW() + make_interval(mins => %s))",
         (code, user, LINK_TTL_MIN))
    bot = os.environ["TELEGRAM_BOT_USERNAME"].strip().lstrip("@")
    return {"url": f"https://t.me/{bot}?start={code}", "expires_in_minutes": LINK_TTL_MIN}


def set_discord(user: str, webhook_url: str) -> Dict[str, Any]:
    url = (webhook_url or "").strip()
    if not DISCORD_URL.match(url):
        raise ChatError("That isn't a Discord webhook URL. In Discord: Channel settings → Integrations → Webhooks → Copy Webhook URL.")
    ok, _ = send_discord(url, "✅ HSFinest.AI alerts will post in this channel.")
    if not ok:
        raise ChatError("Discord didn't accept a test message on that webhook. Check the URL and try again.")
    _save(user, "discord", url, "Discord channel")
    return summary(user)


def handle_telegram_update(update: Dict[str, Any]) -> None:
    """A message to the bot. /start <code> links the chat, /stop unlinks it."""
    msg = update.get("message") or {}
    chat = msg.get("chat") or {}
    text = str(msg.get("text") or "").strip()
    chat_id = chat.get("id")
    if chat_id is None or not text.startswith("/"):
        return
    command, _, arg = text.partition(" ")
    command = command.split("@", 1)[0].lower()
    if command == "/start":
        rows = _run("DELETE FROM hsf_chat_link_codes WHERE code = %s AND expires_at > NOW() RETURNING user_id",
                    (arg.strip()[:64],), True) if arg.strip() else []
        if not rows:
            send_telegram(str(chat_id), "This link has expired. Open Alerts → Delivery on HSFinest.AI and tap Connect Telegram again.")
            return
        user = rows[0]["user_id"]
        name = chat.get("title") or chat.get("username") or chat.get("first_name") or "Telegram chat"
        _save(user, "telegram", str(chat_id), str(name)[:60])
        send_telegram(str(chat_id), "✅ Connected. Your HSFinest.AI alerts will arrive here. Send /stop to turn them off.")
    elif command == "/stop":
        _run("DELETE FROM hsf_chat_destinations WHERE kind = 'telegram' AND target = %s", (str(chat_id),))
        send_telegram(str(chat_id), "Alerts turned off for this chat. Reconnect any time from Alerts → Delivery.")


def register_telegram_webhook(*, client=None) -> bool:
    """Point the bot at this API's webhook with the shared secret. Called once at start-up;
    setWebhook is idempotent, so a changed secret or URL is picked up on the next deploy.
    Pending updates are kept. Never logs the token."""
    if not (enabled() and telegram_ready()):
        return False
    base = (os.environ.get("HSF_PUBLIC_API_URL", "") or os.environ.get("RENDER_EXTERNAL_URL", "")).strip().rstrip("/")
    if not base.startswith("https://"):
        log.warning(json.dumps({"event": "telegram_webhook_skipped", "reason": "no https base url"}))
        return False
    import httpx

    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    own = client is None
    client = client or httpx.Client(timeout=10.0)
    try:
        r = client.post(f"{TELEGRAM_API}/bot{token}/setWebhook",
                        json={"url": f"{base}/v1/telegram/webhook",
                              "secret_token": os.environ["TELEGRAM_WEBHOOK_SECRET"].strip(),
                              "allowed_updates": ["message"]})
        ok = r.status_code == 200 and bool((r.json() or {}).get("ok"))
    except Exception as e:
        log.warning(json.dumps({"event": "telegram_webhook_failed", "error": type(e).__name__}))
        return False
    finally:
        if own:
            client.close()
    log.info(json.dumps({"event": "telegram_webhook_set" if ok else "telegram_webhook_failed",
                         "status": r.status_code, "url": f"{base}/v1/telegram/webhook"}))
    return ok


# ---- sending -----------------------------------------------------------------------------------------
def send_telegram(chat_id: str, text: str, *, client=None) -> tuple:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return False, None
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        r = client.post(f"{TELEGRAM_API}/bot{token}/sendMessage",
                        json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True})
        return 200 <= r.status_code < 300, r.status_code
    except Exception as e:
        log.warning(json.dumps({"event": "telegram_send_failed", "error": type(e).__name__}))
        return False, None
    finally:
        if own:
            client.close()


def send_discord(url: str, text: str, *, client=None) -> tuple:
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=False)
    try:
        r = client.post(url, json={"content": text[:1900], "allowed_mentions": {"parse": []}})
        return 200 <= r.status_code < 300, r.status_code
    except Exception as e:
        log.warning(json.dumps({"event": "discord_send_failed", "error": type(e).__name__}))
        return False, None
    finally:
        if own:
            client.close()


def _app_url() -> str:
    return (os.environ.get("APP_BASE_URL", "").strip() or "https://app.hsfinest.com").rstrip("/")


def send_all(user: str, text: str) -> int:
    """Send to every linked chat. A destination that is gone (Telegram 403 = bot blocked or removed,
    Discord 404 = webhook deleted) is removed. Returns how many accepted it."""
    sent = 0
    for d in destinations(user):
        if d["kind"] == "telegram":
            ok, code = send_telegram(d["target"], text)
            gone = code == 403
        elif d["kind"] == "discord":
            ok, code = send_discord(d["target"], text)
            gone = code in (401, 404)
        else:
            continue
        if gone:
            remove(user, d["kind"])
        else:
            _mark(user, d["kind"], "accepted" if ok else "failed")
        sent += 1 if ok else 0
    return sent


def alert_fired(user_id: str, message: str) -> int:
    """Hook for both alert kinds. Never raises."""
    user = str(user_id).strip().lower()
    if not enabled():
        return 0
    try:
        if not destinations(user):
            return 0
        from api import store, webpush
        from api.workspace import is_quiet

        if not webpush.plan_allows(store.get_account(user)) or is_quiet(user):
            return 0
        return send_all(user, f"⚡ HSF alert: {message}\n{_app_url()}/alerts")
    except Exception as e:
        log.warning(json.dumps({"event": "chat_alert_failed", "error": type(e).__name__}))
        return 0
