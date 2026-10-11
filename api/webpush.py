"""Browser push notifications for fired alerts (Web Push, RFC 8030/8291/8292).

Off until hsf-api has a VAPID key pair: VAPID_PRIVATE_KEY (the raw P-256 private
key, base64url, as `python scripts/generate_vapid_keys.py` prints it) and
VAPID_SUBJECT (a mailto: or https: contact push services can reach).
HSF_WEB_PUSH_ENABLED=0 turns it off again without removing the keys. The public key is
derived from the private one and handed to browsers by GET /v1/web-push/config.

A browser subscription is stored in api_push_devices (provider "webpush",
platform "web") with its endpoint and keys as the token, so password changes and
"sign out everywhere" remove it like a phone. Push services answer 404/410 for a
subscription the browser dropped; that row is deleted. Payloads are encrypted
with aes128gcm using `cryptography` (already a dependency); nothing new to install.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

log = logging.getLogger("hsf_api.webpush")

PROVIDER = "webpush"
PLATFORM = "web"
TTL_S = 4 * 3600          # a push that can't be delivered within 4 hours is no longer news
RECORD_SIZE = 4096
_JWT_TTL_S = 12 * 3600
_B64 = re.compile(r"^[A-Za-z0-9_\-]+={0,2}$")


class InvalidSubscription(ValueError):
    """Not a browser push subscription (400)."""


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


# ---- keys -------------------------------------------------------------------------------------------
def _private_key():
    raw = os.environ.get("VAPID_PRIVATE_KEY", "").strip()
    if not raw:
        return None
    try:
        from cryptography.hazmat.primitives.asymmetric import ec

        return ec.derive_private_key(int.from_bytes(_b64d(raw), "big"), ec.SECP256R1())
    except Exception:
        log.warning(json.dumps({"event": "webpush_bad_vapid_key"}))
        return None


def _public_bytes(key) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)


def public_key() -> Optional[str]:
    key = _private_key()
    return _b64e(_public_bytes(key)) if key is not None else None


def enabled() -> bool:
    return (public_key() is not None and bool(os.environ.get("VAPID_SUBJECT", "").strip())
            and os.environ.get("HSF_WEB_PUSH_ENABLED", "1").strip() != "0")


def config() -> Dict[str, Any]:
    on = enabled()
    return {"enabled": on, "public_key": public_key() if on else None}


# ---- subscriptions ----------------------------------------------------------------------------------
def subscription_token(endpoint: str, p256dh: str, auth: str) -> str:
    """Validate a PushSubscription and return it as the stored token (canonical JSON)."""
    endpoint = endpoint.strip()
    parts = urlsplit(endpoint)
    if parts.scheme != "https" or not parts.netloc or len(endpoint) > 1000:
        raise InvalidSubscription("That isn't a browser push subscription.")
    if not (_B64.match(p256dh) and _B64.match(auth)):
        raise InvalidSubscription("That isn't a browser push subscription.")
    try:
        key, secret = _b64d(p256dh), _b64d(auth)
    except Exception as e:
        raise InvalidSubscription("That isn't a browser push subscription.") from e
    if len(key) != 65 or key[0] != 4 or len(secret) != 16:
        raise InvalidSubscription("That isn't a browser push subscription.")
    return json.dumps({"endpoint": endpoint, "keys": {"auth": auth, "p256dh": p256dh}},
                      sort_keys=True, separators=(",", ":"))


def endpoint_of(token: str) -> Optional[str]:
    try:
        return str(json.loads(token)["endpoint"])
    except Exception:
        return None


# ---- encryption (RFC 8291, aes128gcm) ---------------------------------------------------------------
def _hmac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def encrypt(payload: bytes, p256dh: str, auth: str, *, salt: Optional[bytes] = None, server_key=None) -> bytes:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    ua_public = _b64d(p256dh)
    auth_secret = _b64d(auth)
    server_key = server_key or ec.generate_private_key(ec.SECP256R1())
    as_public = _public_bytes(server_key)
    ua_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), ua_public)
    ecdh = server_key.exchange(ec.ECDH(), ua_key)
    ikm = _hmac(_hmac(auth_secret, ecdh), b"WebPush: info\x00" + ua_public + as_public + b"\x01")
    salt = salt or os.urandom(16)
    prk = _hmac(salt, ikm)
    cek = _hmac(prk, b"Content-Encoding: aes128gcm\x00\x01")[:16]
    nonce = _hmac(prk, b"Content-Encoding: nonce\x00\x01")[:12]
    if len(payload) > RECORD_SIZE - 17 - 86:
        raise ValueError("push payload too large")
    body = AESGCM(cek).encrypt(nonce, payload + b"\x02", None)
    header = salt + RECORD_SIZE.to_bytes(4, "big") + bytes([len(as_public)]) + as_public
    return header + body


def _vapid_header(endpoint: str, key) -> str:
    import jwt

    parts = urlsplit(endpoint)
    subject = os.environ.get("VAPID_SUBJECT", "").strip()
    token = jwt.encode({"aud": f"{parts.scheme}://{parts.netloc}", "exp": int(time.time()) + _JWT_TTL_S,
                        "sub": subject}, key, algorithm="ES256")
    return f"vapid t={token}, k={_b64e(_public_bytes(key))}"


# ---- sending ----------------------------------------------------------------------------------------
def send(token: str, message: Dict[str, Any], *, client=None) -> Tuple[bool, Optional[int]]:
    """Push one message to one subscription. (delivered, status); status 404/410 = gone."""
    key = _private_key()
    if key is None:
        return False, None
    sub = json.loads(token)
    body = encrypt(json.dumps(message, separators=(",", ":")).encode(), sub["keys"]["p256dh"], sub["keys"]["auth"])
    headers = {"Content-Encoding": "aes128gcm", "Content-Type": "application/octet-stream",
               "TTL": str(TTL_S), "Urgency": "high", "Authorization": _vapid_header(sub["endpoint"], key)}
    import httpx

    own = client is None
    client = client or httpx.Client(timeout=5.0)
    try:
        r = client.post(sub["endpoint"], content=body, headers=headers)
        return 200 <= r.status_code < 300, r.status_code
    except Exception as e:
        log.warning(json.dumps({"event": "webpush_send_failed", "error": type(e).__name__}))
        return False, None
    finally:
        if own:
            client.close()


def notify_user(user_id: str, title: str, body: str, url: str = "/alerts") -> int:
    """Push to every browser the account turned notifications on in. Returns how many
    accepted it. Best effort: never raises."""
    if not enabled():
        return 0
    try:
        from api import devices

        targets: List[Dict[str, Any]] = [d for d in devices.devices_for_user(str(user_id).strip().lower())
                                         if d.get("provider") == PROVIDER]
    except Exception as e:
        log.warning(json.dumps({"event": "webpush_targets_failed", "error": type(e).__name__}))
        return 0
    if not targets:
        return 0
    message = {"title": title[:80], "body": body[:240], "url": url if url.startswith("/") else "/alerts",
               "tag": "hsf-alert"}
    sent = 0
    import httpx

    with httpx.Client(timeout=5.0) as client:
        for d in targets:
            try:
                ok, status = send(d["token"], message, client=client)
            except Exception as e:
                log.warning(json.dumps({"event": "webpush_send_failed", "error": type(e).__name__}))
                continue
            sent += 1 if ok else 0
            if status in (404, 410):
                try:
                    from api import devices

                    devices.remove(str(user_id).strip().lower(), int(d["id"]))
                except Exception:
                    pass
    return sent


def alert_fired(user_id: str, message: str) -> None:
    """Hook for both alert kinds (price alerts and alert rules)."""
    notify_user(user_id, "HSF alert", message, "/alerts")
