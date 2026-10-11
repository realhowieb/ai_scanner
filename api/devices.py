"""Push-notification devices for the mobile app (P1-64).

The app registers the push token its platform gives it (APNs on iOS, FCM on
Android, or an Expo push token) against the signed-in account; the web app
registers browser push subscriptions here too (provider "webpush", see
api.webpush, which is the only sender so far). Senders read `devices_for_user()`.

A token belongs to at most one account: registering it again (another sign-in
on the same phone) moves it. Devices are removed when the app signs out with
its push token, and every device of an account is removed when all of its
sessions are signed out (password change or reset, refresh-token theft), so a
signed-out phone never keeps receiving that account's alerts.

Writes only its own `api_push_devices` table.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from api.store import _conn
from db.engine import schema_once

MAX_DEVICES = 10  # per account; the least recently seen ones go first
PROVIDERS = ("apns", "fcm", "expo")
PLATFORMS = ("ios", "android")

_TOKEN_RULES = {
    "apns": re.compile(r"^[0-9A-Fa-f]{64,200}$"),
    "fcm": re.compile(r"^[A-Za-z0-9_:\-]{32,512}$"),
    "expo": re.compile(r"^Expo(?:nent)?PushToken\[[A-Za-z0-9_\-]{10,100}\]$"),
}


class InvalidDevice(ValueError):
    """The token doesn't look like a push token for that provider (400)."""


def resolve_provider(token: str, platform: str, provider: Optional[str]) -> str:
    """The provider given, else Expo for Expo tokens, APNs on iOS, FCM on Android."""
    if provider:
        chosen = provider
    elif token.startswith(("ExponentPushToken[", "ExpoPushToken[")):
        chosen = "expo"
    else:
        chosen = "apns" if platform == "ios" else "fcm"
    if not _TOKEN_RULES[chosen].match(token):
        raise InvalidDevice(f"That doesn't look like a {chosen.upper()} push token.")
    if chosen == "apns" and platform != "ios":
        raise InvalidDevice("APNs tokens are only valid on iOS.")
    return chosen


def _rows(cur) -> List[Dict[str, Any]]:
    cols = [d.name for d in cur.description]
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, r)) for r in cur.fetchall()]


@schema_once
def ensure_devices_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS api_push_devices (
            id BIGSERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            token TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            platform TEXT NOT NULL,
            device_name TEXT,
            app_version TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS api_push_devices_user ON api_push_devices (username)")
    conn.commit()
    cur.close()


def register(username: str, token: str, provider: str, platform: str,
             device_name: Optional[str], app_version: Optional[str]) -> Dict[str, Any]:
    """Add or refresh this device for the account (idempotent). Keeps the newest
    MAX_DEVICES devices per account."""
    conn = _conn()
    try:
        ensure_devices_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"api_push_devices:{username}",))
        cur.execute(
            "INSERT INTO api_push_devices (username, token, provider, platform, device_name, app_version) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (token) DO UPDATE SET "
            "  created_at = CASE WHEN api_push_devices.username = EXCLUDED.username "
            "                    THEN api_push_devices.created_at ELSE NOW() END, "
            "  username = EXCLUDED.username, provider = EXCLUDED.provider, platform = EXCLUDED.platform, "
            "  device_name = EXCLUDED.device_name, app_version = EXCLUDED.app_version, last_seen_at = NOW() "
            "RETURNING id, provider, platform, device_name, app_version, created_at, last_seen_at",
            (username, token, provider, platform, device_name, app_version),
        )
        device = _rows(cur)[0]
        cur.execute(
            "DELETE FROM api_push_devices WHERE username = %s AND id NOT IN ("
            "  SELECT id FROM api_push_devices WHERE username = %s "
            "  ORDER BY last_seen_at DESC, id DESC LIMIT %s)",
            (username, username, MAX_DEVICES),
        )
        conn.commit()
        cur.close()
        return device
    finally:
        conn.close()


def list_devices(username: str) -> List[Dict[str, Any]]:
    """The account's devices, newest first, without their push tokens."""
    conn = _conn()
    try:
        ensure_devices_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT id, provider, platform, device_name, app_version, created_at, last_seen_at "
                    "FROM api_push_devices WHERE username = %s ORDER BY last_seen_at DESC, id DESC", (username,))
        rows = _rows(cur)
        conn.rollback()
        cur.close()
        return rows
    finally:
        conn.close()


def devices_for_user(username: str) -> List[Dict[str, Any]]:
    """Push targets for the alert sender: provider, platform and token."""
    conn = _conn()
    try:
        ensure_devices_schema(conn)
        cur = conn.cursor()
        cur.execute("SELECT id, provider, platform, token FROM api_push_devices WHERE username = %s "
                    "ORDER BY last_seen_at DESC", (username,))
        rows = _rows(cur)
        conn.rollback()
        cur.close()
        return rows
    finally:
        conn.close()


def _delete(sql: str, params: tuple) -> int:
    conn = _conn()
    try:
        ensure_devices_schema(conn)
        cur = conn.cursor()
        cur.execute(sql, params)
        n = cur.rowcount or 0
        conn.commit()
        cur.close()
        return int(n)
    finally:
        conn.close()


def remove(username: str, device_id: int) -> bool:
    """Remove one of the account's devices; False when it isn't theirs or doesn't exist."""
    return _delete("DELETE FROM api_push_devices WHERE username = %s AND id = %s", (username, int(device_id))) > 0


def remove_token(username: str, token: str) -> int:
    """Remove the device with this push token if it belongs to the account (sign-out)."""
    return _delete("DELETE FROM api_push_devices WHERE username = %s AND token = %s", (username, token))


def remove_web_endpoint(username: str, endpoint: str) -> int:
    """Remove this browser's push subscription (it turned notifications off or signed out)."""
    return _delete("DELETE FROM api_push_devices WHERE username = %s AND provider = 'webpush' "
                   "AND (token::jsonb ->> 'endpoint') = %s", (username, endpoint))


def remove_all(username: str) -> int:
    """Remove every device of the account (all sessions signed out)."""
    return _delete("DELETE FROM api_push_devices WHERE username = %s", (username,))
