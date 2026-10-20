"""Bounded, read-only data access for the Admin analytics dashboard."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from db.engine import get_neon_conn

MAX_EVENTS = 20_000
MAX_RUNS = 20_000
MAX_RESEARCH_ROWS = 100_000
MAX_USERS = 100_000


def _value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _query(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> tuple[list[Any], str | None]:
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() or []
        cur.close()
        return list(rows), None
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return [], type(exc).__name__


def load_admin_data(
    *,
    days: int = 90,
    research_limit: int = MAX_RESEARCH_ROWS,
    include_research: bool = True,
) -> dict[str, Any]:
    """Load one cached-dashboard bundle without creating or altering schema."""
    conn = get_neon_conn()
    if conn is None:
        return {"available": False, "errors": {"database": "unavailable"}}

    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, int(days)))
    errors: dict[str, str] = {}
    try:
        user_rows, error = _query(
            conn,
            "SELECT tier, is_active, created_at, stripe_subscription_id FROM users "
            "ORDER BY created_at DESC LIMIT %s",
            (MAX_USERS,),
        )
        if error:
            errors["users"] = error
        users = [{
            "tier": _value(row, "tier", 0),
            "is_active": bool(_value(row, "is_active", 1)),
            "created_at": _value(row, "created_at", 2),
            "has_subscription": bool(_value(row, "stripe_subscription_id", 3)),
        } for row in user_rows]

        event_rows, error = _query(
            conn,
            "SELECT event_name, user_hash, plan, occurred_at FROM acquisition_events "
            "WHERE occurred_at >= %s ORDER BY occurred_at DESC LIMIT %s",
            (since, MAX_EVENTS),
        )
        if error:
            errors["acquisition_events"] = error
        events = [{
            "event_name": _value(row, "event_name", 0),
            "user_hash": _value(row, "user_hash", 1),
            "plan": _value(row, "plan", 2),
            "occurred_at": _value(row, "occurred_at", 3),
        } for row in event_rows]

        run_rows, error = _query(
            conn,
            "SELECT username, name, label, row_count, duration_sec, is_snapshot, created_at "
            "FROM runs WHERE created_at >= %s ORDER BY created_at DESC LIMIT %s",
            (since, MAX_RUNS),
        )
        if error:
            errors["runs"] = error
        runs = [{
            "username": _value(row, "username", 0),
            "name": _value(row, "name", 1),
            "label": _value(row, "label", 2),
            "row_count": _value(row, "row_count", 3),
            "duration_sec": _value(row, "duration_sec", 4),
            "is_snapshot": bool(_value(row, "is_snapshot", 5)),
            "created_at": _value(row, "created_at", 6),
        } for row in run_rows]

        by_observation: dict[str, dict[str, Any]] = {}
        if include_research:
            joined_rows, error = _query(
                conn,
                "SELECT o.observation_id, o.record, oc.horizon, oc.record AS outcome_record "
                "FROM hsf_observations o LEFT JOIN hsf_observation_outcomes oc "
                "ON oc.observation_id = o.observation_id "
                "WHERE o.timestamp >= %s ORDER BY o.timestamp DESC LIMIT %s",
                (since, min(MAX_RESEARCH_ROWS, max(1, int(research_limit)))),
            )
            if error:
                errors["research"] = error
            for row in joined_rows:
                oid = str(_value(row, "observation_id", 0) or "")
                if not oid:
                    continue
                observation = by_observation.setdefault(oid, _loads(_value(row, "record", 1)))
                observation.setdefault("observation_id", oid)
                horizon = _value(row, "horizon", 2)
                outcome = _loads(_value(row, "outcome_record", 3))
                if horizon is not None and outcome:
                    observation.setdefault("outcomes", {})[str(horizon)] = outcome

        health_rows, error = _query(
            conn,
            "SELECT record FROM hsf_system_health ORDER BY generated_at DESC, id DESC LIMIT 1",
        )
        if error:
            errors["system_health"] = error
        health = _loads(_value(health_rows[0], "record", 0)) if health_rows else None

        recovery_rows, error = _query(
            conn,
            "SELECT record FROM hsf_recovery_events ORDER BY started_at DESC, id DESC LIMIT 100",
        )
        if error:
            errors["recovery"] = error
        recovery = [_loads(_value(row, "record", 0)) for row in recovery_rows]

        return {
            "available": True,
            "loaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "days": int(days),
            "users": users,
            "events": events,
            "runs": runs,
            "observations": list(by_observation.values()),
            "health": health,
            "recovery": [row for row in recovery if row],
            "errors": errors,
            "limits": {"users": MAX_USERS, "events": MAX_EVENTS, "runs": MAX_RUNS,
                       "research_rows": int(research_limit) if include_research else 0},
        }
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_signup_funnel(days: int = 30) -> dict[str, Any]:
    """Accounts created in the last `days` and how far each got: verified its email,
    saved a watchlist stock or an alert, pays for Pro/Premium; plus its sign-up source.
    Read-only; each step that fails to load is listed in `errors` and counts as empty."""
    conn = get_neon_conn()
    if conn is None:
        return {"available": False, "errors": {"database": "unavailable"}}
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max(1, int(days)))
    errors: dict[str, str] = {}

    def users_in(sql: str, names: list[str], key: str) -> set[str]:
        if not names:
            return set()
        rows, error = _query(conn, sql, (names,))
        if error:
            errors[key] = error
        return {str(_value(r, "u", 0) or "").strip().lower() for r in rows}

    try:
        rows, error = _query(
            conn,
            "SELECT lower(username) AS u, tier, stripe_subscription_id, email_verified, is_admin "
            "FROM users WHERE created_at >= %s ORDER BY created_at DESC LIMIT %s",
            (since, MAX_USERS),
        )
        if error:
            return {"available": False, "errors": {"users": error}}
        accounts = []
        for row in rows:
            name = str(_value(row, "u", 0) or "")
            if "@" not in name or bool(_value(row, "is_admin", 4)):
                continue
            accounts.append({"user": name, "tier": _value(row, "tier", 1),
                             "has_subscription": bool(_value(row, "stripe_subscription_id", 2)),
                             "verified": bool(_value(row, "email_verified", 3))})
        names = [a["user"] for a in accounts]
        engaged = (
            users_in("SELECT DISTINCT lower(w.user_id) AS u FROM watchlists w "
                     "JOIN watchlist_items i ON i.watchlist_id = w.id WHERE lower(w.user_id) = ANY(%s)",
                     names, "watchlists")
            | users_in("SELECT DISTINCT lower(user_id) AS u FROM user_alerts WHERE lower(user_id) = ANY(%s)",
                       names, "alerts")
            | users_in("SELECT DISTINCT lower(user_id) AS u FROM hsf_alert_rules WHERE lower(user_id) = ANY(%s)",
                       names, "alert_rules")
        )
        trials = users_in("SELECT DISTINCT lower(user_id) AS u FROM hsf_plan_grants "
                          "WHERE tier = 'pro' AND lower(user_id) = ANY(%s)", names, "trials")
        src_rows, error = _query(
            conn,
            "SELECT user_hash, source FROM acquisition_events "
            "WHERE event_name = 'signup_completed' AND occurred_at >= %s LIMIT %s",
            (since, MAX_EVENTS),
        )
        if error:
            errors["sources"] = error
        sources = {str(_value(r, "user_hash", 0)): str(_value(r, "source", 1) or "")
                   for r in src_rows if _value(r, "user_hash", 0)}
        for a in accounts:
            a["engaged"] = a["user"] in engaged
            a["trial"] = a["user"] in trials
        return {"available": True, "accounts": accounts, "sources": sources, "errors": errors}
    finally:
        try:
            conn.close()
        except Exception:
            pass
