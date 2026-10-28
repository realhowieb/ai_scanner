"""The API's route groups. Each module's register(app) adds its endpoints.

register_all keeps the original order, which is also the order of paths in the
OpenAPI document (web/openapi.json).
"""
from __future__ import annotations

from fastapi import FastAPI


def register_all(app: FastAPI) -> None:
    from api.routes import (
        account,
        admin,
        ai,
        auth,
        data,
        devices,
        founding,
        history,
        market,
        onboarding,
        outcomes,
        public,
        research,
        scans,
        trading,
        workspace,
    )

    workspace.register(app)
    auth.register(app)
    data.register(app)
    account.register(app)
    devices.register(app)
    scans.register(app)
    history.register(app)
    outcomes.register(app)
    market.register(app)
    ai.register(app)
    trading.register(app)
    public.register(app)
    research.register(app)
    research.register_ml(app)
    admin.register(app)
    onboarding.register(app)
    founding.register(app)
    account.register_delete(app)
