"""Account-owned research preferences. Each PATCH merges sections atomically."""
from __future__ import annotations

import datetime as dt
import json
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.store import _conn
from billing_service.notification_preferences import quiet_at
from db.engine import schema_once


class StrictModel(BaseModel):
    @field_validator("scanner_views", "custom_views", check_fields=False)
    @classmethod
    def unique_names(cls, v):
        if v is not None and len({view.name for view in v}) != len(v):
            raise ValueError("Each saved view needs a different name.")
        return v

    @field_validator("name", check_fields=False)
    @classmethod
    def trim_name(cls, v):
        if not v.strip():
            raise ValueError("Give this view a name.")
        return v.strip()

    model_config = ConfigDict(extra="forbid")


class ScannerView(StrictModel):
    name: str = Field(min_length=1, max_length=60)
    signal: Literal["", "breakout", "gapper", "golden_cross", "gainer", "prebreakout", "earnings"] = ""
    when: Literal["today", "tomorrow", "week"] = Field(default="today", description="Earnings window (signal earnings)")
    min: Literal[0, 40, 60, 75] = 0
    sort: Literal["score", "chg_pct", "gap_pct", "rvol", "prob"] = "score"
    size: Literal[25, 50, 100] = 25


class CustomView(StrictModel):
    name: str = Field(min_length=1, max_length=60)
    universe: Literal["sp500", "nasdaq", "combo", "us_market", "watchlist", "ticker", "earnings"] = "sp500"
    earnings_window: Literal["today", "tomorrow", "week"] = "today"
    ticker: str = Field(default="", max_length=10, pattern=r"^$|^[A-Za-z][A-Za-z0-9.\-]{0,9}$")
    watchlist_id: int | None = Field(default=None, ge=1)
    session: Literal["regular", "premarket", "afterhours"] = "regular"
    top_n: Literal[10, 25, 50, 100, 200] = 25
    min_price: float | None = Field(default=None, ge=0, le=1000000, allow_inf_nan=False)
    max_price: float | None = Field(default=None, ge=0, le=1000000, allow_inf_nan=False)
    min_dollar_vol: float | None = Field(default=None, ge=0, le=1e15, allow_inf_nan=False)
    unusual: bool = False
    gap: bool = False
    min_gap: float | None = Field(default=None, ge=0, le=10000, allow_inf_nan=False)
    score_all: bool = False


Column = Literal["setup", "spark", "last", "change", "gap", "rvol", "prob"]


class TablePreferences(StrictModel):
    columns: list[Column] = Field(default_factory=lambda: ["setup", "spark", "last", "change", "gap", "rvol", "prob"], max_length=7)
    compact: bool = False

    @field_validator("columns")
    @classmethod
    def unique_columns(cls, v):
        return list(dict.fromkeys(v))


class QuietHours(StrictModel):
    enabled: bool = False
    start: str = Field(default="22:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(default="07:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    timezone: str = Field(default="America/New_York", max_length=64)

    @field_validator("timezone")
    @classmethod
    def timezone_exists(cls, v):
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Choose a valid time zone.") from None
        return v


class WorkspacePreferences(StrictModel):
    scanner_views: list[ScannerView] = Field(default_factory=list, max_length=20)
    custom_views: list[CustomView] = Field(default_factory=list, max_length=20)
    table: TablePreferences = Field(default_factory=TablePreferences)
    quiet_hours: QuietHours = Field(default_factory=QuietHours)


class WorkspacePatch(StrictModel):
    scanner_views: list[ScannerView] | None = Field(default=None, max_length=20)
    custom_views: list[CustomView] | None = Field(default=None, max_length=20)
    table: TablePreferences | None = None
    quiet_hours: QuietHours | None = None


@schema_once
def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("hsf_schema_workspace",))
        cur.execute("""CREATE TABLE IF NOT EXISTS hsf_workspace_preferences (
            user_id TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE, data JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
    conn.commit()


def preferences(user: str, changes: dict | None = None) -> dict:
    conn = _conn()
    try:
        ensure_schema(conn)
        with conn.cursor() as cur:
            if changes:
                cur.execute("INSERT INTO hsf_workspace_preferences (user_id, data) VALUES (%s, %s::jsonb) "
                            "ON CONFLICT (user_id) DO UPDATE SET data = hsf_workspace_preferences.data || EXCLUDED.data, "
                            "updated_at = NOW() RETURNING data", (user.strip().lower(), json.dumps(changes)))
            else:
                cur.execute("SELECT data FROM hsf_workspace_preferences WHERE user_id = %s", (user.strip().lower(),))
            row = cur.fetchone()
            data = (row["data"] if isinstance(row, dict) else row[0]) if row else {}
        conn.commit()
        return WorkspacePreferences.model_validate(data).model_dump()
    finally:
        conn.close()


def is_quiet(user: str, now: dt.datetime | None = None) -> bool:
    q = preferences(user)["quiet_hours"]
    return quiet_at(q, now)
