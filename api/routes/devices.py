"""Push devices: phones (APNs, FCM, Expo) and browsers (Web Push)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path
from pydantic import BaseModel, Field

from api import devices, models, user_data
from api.deps import _AUTH, _OWNED, _user, current_account
from api.scans import json_safe


class DeviceBody(BaseModel):
    push_token: str = Field(min_length=10, max_length=600, description="Token from APNs, FCM or Expo")
    platform: Literal["ios", "android"]
    provider: Optional[Literal["apns", "fcm", "expo"]] = Field(
        default=None, description="Default: expo for Expo tokens, apns on iOS, fcm on Android")
    device_name: Optional[str] = Field(default=None, max_length=80, description="Shown in the app's device list")
    app_version: Optional[str] = Field(default=None, max_length=40)


class WebPushKeys(BaseModel):
    p256dh: str = Field(min_length=80, max_length=100)
    auth: str = Field(min_length=16, max_length=30)


class WebPushSubscribe(BaseModel):
    endpoint: str = Field(min_length=12, max_length=1000, description="PushSubscription.endpoint")
    keys: WebPushKeys
    device_name: Optional[str] = Field(default=None, max_length=80, description="e.g. Chrome on Mac")


class WebPushUnsubscribe(BaseModel):
    endpoint: str = Field(min_length=12, max_length=1000)


def _device_out(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    for k in ("created_at", "last_seen_at"):
        if isinstance(out.get(k), (dt.datetime, dt.date)):
            out[k] = json_safe(out[k])
    return out


def register(app: FastAPI) -> None:
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

    @app.get("/v1/web-push/config", response_model=models.WebPushConfig, responses=_AUTH,
             summary="Browser notifications: whether they're on and the server key")
    def web_push_config(account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        from api import webpush

        return webpush.config(webpush.plan_allows(account))

    @app.post("/v1/me/web-push", response_model=models.Device,
              responses={**_AUTH, 400: {"description": "Not a push subscription"},
                         403: {"description": "Browser notifications are part of Pro"},
                         503: {"description": "Browser notifications aren't set up on the server"}},
              summary="Turn on alert notifications in this browser")
    def web_push_subscribe(body: WebPushSubscribe, account: Dict[str, Any] = Depends(current_account)) -> Dict[str, Any]:
        """Every alert that fires for you (price alerts and alert rules) is also pushed to this
        browser. Pass the browser's PushSubscription (endpoint and keys). Pro and above."""
        from api import webpush

        if not webpush.plan_allows(account):
            raise HTTPException(403, webpush.UPGRADE)
        if not webpush.enabled():
            raise HTTPException(503, "Browser notifications aren't available yet.")
        try:
            token = webpush.subscription_token(body.endpoint, body.keys.p256dh, body.keys.auth)
        except webpush.InvalidSubscription as e:
            raise HTTPException(400, str(e)) from e
        return _device_out(devices.register(_user(account), token, webpush.PROVIDER, webpush.PLATFORM,
                                            (body.device_name or "").strip() or None, None))

    @app.delete("/v1/me/web-push", status_code=204, responses=_AUTH,
                summary="Turn off alert notifications in this browser")
    def web_push_unsubscribe(body: WebPushUnsubscribe, account: Dict[str, Any] = Depends(current_account)) -> None:
        devices.remove_web_endpoint(_user(account), body.endpoint.strip())

    @app.delete("/v1/me/devices/{device_id}", status_code=204, responses=_OWNED)
    def device_remove(device_id: int = Path(ge=1), account: Dict[str, Any] = Depends(current_account)) -> None:
        if not devices.remove(_user(account), device_id):
            raise user_data.NotFound("device")
