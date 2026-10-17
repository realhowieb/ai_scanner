"""iPhone app push notifications for fired alerts (Apple Push Notification service).

Off until hsf-api has an APNs auth key: APNS_KEY_P8 (the .p8 file's contents, from
developer.apple.com > Keys with "Apple Push Notifications service" ticked), APNS_KEY_ID
and APNS_TEAM_ID. APNS_TOPIC is the app's bundle id (default com.hsfinest.app) and
APNS_SANDBOX=1 sends to Apple's development server (Xcode debug builds; TestFlight and
App Store builds use production). HSF_APNS_ENABLED=0 turns it off without removing the key.

The app registers its device token with POST /v1/me/devices (provider "apns"). Like
browser notifications, iPhone notifications are part of Pro (api.webpush.plan_allows).
Apple answers 410, or 400 BadDeviceToken / DeviceTokenNotForTopic, for a token that's
no longer valid; that row is deleted. APNs only speaks HTTP/2 (httpx with h2).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("hsf_api.apns")

PROVIDER = "apns"
DEFAULT_TOPIC = "com.hsfinest.app"
TTL_S = 4 * 3600                # same as browser pushes: older than 4 hours is no longer news
_JWT_REFRESH_S = 45 * 60        # Apple accepts a provider token for up to an hour
_GONE_REASONS = {"BadDeviceToken", "DeviceTokenNotForTopic", "Unregistered"}

_jwt_lock = threading.Lock()
_jwt_cache: Dict[str, Any] = {}


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _key_pem() -> str:
    # Render env vars may hold the PEM with literal "\n" instead of newlines.
    return _env("APNS_KEY_P8").replace("\\n", "\n")


def enabled() -> bool:
    return (bool(_key_pem() and _env("APNS_KEY_ID") and _env("APNS_TEAM_ID"))
            and os.environ.get("HSF_APNS_ENABLED", "1").strip() != "0")


def topic() -> str:
    return _env("APNS_TOPIC") or DEFAULT_TOPIC


def host() -> str:
    return "api.sandbox.push.apple.com" if _env("APNS_SANDBOX") == "1" else "api.push.apple.com"


def provider_token(now: Optional[float] = None) -> str:
    """The ES256 provider token Apple wants in `authorization: bearer`, reused for 45 minutes."""
    import jwt

    now = time.time() if now is None else now
    key_id, team = _env("APNS_KEY_ID"), _env("APNS_TEAM_ID")
    with _jwt_lock:
        cached = _jwt_cache.get("token")
        if cached and _jwt_cache.get("kid") == key_id and now - _jwt_cache.get("iat", 0) < _JWT_REFRESH_S:
            return cached
        token = jwt.encode({"iss": team, "iat": int(now)}, _key_pem(), algorithm="ES256", headers={"kid": key_id})
        _jwt_cache.update(token=token, kid=key_id, iat=int(now))
        return token


def payload(title: str, body: str, url: str) -> Dict[str, Any]:
    return {"aps": {"alert": {"title": title[:80], "body": body[:240]}, "sound": "default",
                    "thread-id": "hsf-alert"},
            "url": url if url.startswith("/") else "/alerts"}


def send(token: str, message: Dict[str, Any], *, client=None) -> Tuple[bool, Optional[int], Optional[str]]:
    """Push one message to one device. (delivered, status, Apple's reason when refused)."""
    if not enabled():
        return False, None, None
    import httpx

    headers = {"authorization": f"bearer {provider_token()}", "apns-topic": topic(),
               "apns-push-type": "alert", "apns-priority": "10",
               "apns-expiration": str(int(time.time()) + TTL_S)}
    own = client is None
    client = client or httpx.Client(http2=True, timeout=5.0)
    try:
        r = client.post(f"https://{host()}/3/device/{token}", content=json.dumps(message, separators=(",", ":")),
                        headers=headers)
        reason = None
        if r.status_code != 200:
            try:
                reason = str(r.json().get("reason") or "") or None
            except Exception:
                reason = None
        return r.status_code == 200, r.status_code, reason
    except Exception as e:
        log.warning(json.dumps({"event": "apns_send_failed", "error": type(e).__name__}))
        return False, None, None
    finally:
        if own:
            client.close()


def _gone(status: Optional[int], reason: Optional[str]) -> bool:
    return status == 410 or (status == 400 and reason in _GONE_REASONS)


def notify_user(user_id: str, title: str, body: str, url: str = "/alerts") -> int:
    """Push to every iPhone the account signed in to the app on. Returns how many Apple
    accepted. Best effort: never raises."""
    if not enabled():
        return 0
    user = str(user_id).strip().lower()
    try:
        from api import devices

        targets: List[Dict[str, Any]] = [d for d in devices.devices_for_user(user) if d.get("provider") == PROVIDER]
    except Exception as e:
        log.warning(json.dumps({"event": "apns_targets_failed", "error": type(e).__name__}))
        return 0
    if not targets:
        return 0
    try:
        from api import store
        from api.webpush import plan_allows

        if not plan_allows(store.get_account(user)):
            return 0
    except Exception as e:
        log.warning(json.dumps({"event": "apns_plan_check_failed", "error": type(e).__name__}))
        return 0
    message = payload(title, body, url)
    sent = 0
    import httpx

    with httpx.Client(http2=True, timeout=5.0) as client:
        for d in targets:
            ok, status, reason = send(str(d["token"]), message, client=client)
            sent += 1 if ok else 0
            if not ok and status is not None:
                log.warning(json.dumps({"event": "apns_refused", "status": status, "reason": reason}))
            if _gone(status, reason):
                try:
                    from api import devices

                    devices.remove(user, int(d["id"]))
                except Exception:
                    pass
    return sent
