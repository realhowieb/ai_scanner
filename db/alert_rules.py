"""Storage for user-owned HSF alert rules, their per-ticker state and their events.

Three tables, all keyed by the account's user_id like user_alerts:

* hsf_alert_rules: what the user asked for (rule type, threshold, scope = one
  ticker or one watchlist, channels, cooldown).
* hsf_alert_rule_state: per (rule, ticker), the last value the evaluator saw and
  which market observation it came from. This is what lets "crosses above 80"
  fire on 77 -> 82 but not on 83 -> 84, and makes re-evaluating the same
  observation a no-op.
* hsf_alert_rule_events: one row per trigger. UNIQUE (rule_id, ticker,
  observation_id) is the hard idempotency guarantee: a second process or a
  retried pass can't insert the same trigger twice.

Postgres (Neon) in production. Tests pass a SQLite connection through
set_connection_factory; the SQL is written once with %s placeholders and
timestamps are ISO strings passed from Python, so both backends run the same
statements. Every write that must stay consistent (event + state) happens in
one transaction (save_evaluation).
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from db.engine import get_neon_conn, schema_once

_conn_factory: Optional[Callable[[], Any]] = None


def set_connection_factory(factory: Optional[Callable[[], Any]]) -> None:
    """Tests: route every call to `factory()` (e.g. one shared SQLite connection)."""
    global _conn_factory
    _conn_factory = factory


class RuleLimitReached(ValueError):
    """The user already has as many active alerts as the plan allows."""


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def to_iso(value: Any) -> Optional[str]:
    """Timestamps out of either backend as ISO 8601 UTC strings."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        v = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        return v.astimezone(dt.timezone.utc).isoformat()
    s = str(value)
    try:
        return to_iso(dt.datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        return s


def _connect() -> Tuple[Any, bool]:
    conn = _conn_factory() if _conn_factory is not None else get_neon_conn()
    if conn is None:
        raise RuntimeError("Neon is not available (missing URL or connection failed).")
    is_sqlite = isinstance(conn, sqlite3.Connection)
    _ensure_schema(conn, is_sqlite)
    return conn, is_sqlite


def _sql(query: str, is_sqlite: bool) -> str:
    return query.replace("%s", "?") if is_sqlite else query


def _rows(cur) -> List[Dict[str, Any]]:
    rows = cur.fetchall() or []
    cols = [d[0] for d in cur.description] if cur.description else []
    return [dict(r) if isinstance(r, dict) else dict(zip(cols, tuple(r))) for r in rows]


def _close(conn, is_sqlite: bool) -> None:
    if is_sqlite and _conn_factory is not None:
        return  # the test's shared connection
    try:
        conn.close()
    except Exception:
        pass


@schema_once
def _ensure_schema(conn, is_sqlite: bool) -> None:
    pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if is_sqlite else "BIGSERIAL PRIMARY KEY"
    ts = "TEXT" if is_sqlite else "TIMESTAMPTZ"
    ref = "INTEGER" if is_sqlite else "BIGINT"
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS hsf_alert_rules (
            id {pk},
            user_id TEXT NOT NULL,
            watchlist_id {ref},
            ticker TEXT,
            rule_type TEXT NOT NULL,
            operator TEXT NOT NULL,
            threshold DOUBLE PRECISION,
            value TEXT,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            delivery_channels TEXT NOT NULL DEFAULT '["in_app"]',
            cooldown_seconds INTEGER NOT NULL DEFAULT 3600,
            created_at {ts} NOT NULL,
            updated_at {ts} NOT NULL,
            last_evaluated_at {ts},
            last_triggered_at {ts}
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_alert_rules_user ON hsf_alert_rules (user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_alert_rules_enabled ON hsf_alert_rules (enabled)")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS hsf_alert_rule_state (
            rule_id {ref} NOT NULL,
            ticker TEXT NOT NULL,
            last_value DOUBLE PRECISION,
            last_flag BOOLEAN,
            last_text TEXT,
            observation_id TEXT,
            last_triggered_at {ts},
            updated_at {ts} NOT NULL,
            PRIMARY KEY (rule_id, ticker)
        )""")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS hsf_alert_rule_events (
            id {pk},
            user_id TEXT NOT NULL,
            rule_id {ref} NOT NULL,
            watchlist_id {ref},
            ticker TEXT NOT NULL,
            rule_type TEXT NOT NULL,
            operator TEXT NOT NULL,
            threshold DOUBLE PRECISION,
            trigger_value DOUBLE PRECISION,
            previous_value DOUBLE PRECISION,
            hsf_score DOUBLE PRECISION,
            setup TEXT,
            message TEXT NOT NULL,
            observation_id TEXT NOT NULL,
            market_data_as_of {ts},
            triggered_at {ts} NOT NULL,
            delivery TEXT NOT NULL DEFAULT '{{}}',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            last_delivery_error TEXT,
            UNIQUE (rule_id, ticker, observation_id)
        )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hsf_rule_events_user_time "
                "ON hsf_alert_rule_events (user_id, triggered_at DESC)")
    conn.commit()
    cur.close()


# ---- rules ----------------------------------------------------------------------------------------
RULE_COLUMNS = ("id", "user_id", "watchlist_id", "ticker", "rule_type", "operator", "threshold", "value",
                "enabled", "delivery_channels", "cooldown_seconds", "created_at", "updated_at",
                "last_evaluated_at", "last_triggered_at")


def _rule_out(r: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: r.get(k) for k in RULE_COLUMNS}
    out["id"] = int(out["id"])
    out["watchlist_id"] = int(out["watchlist_id"]) if out["watchlist_id"] is not None else None
    out["enabled"] = bool(out["enabled"])
    out["threshold"] = float(out["threshold"]) if out["threshold"] is not None else None
    out["cooldown_seconds"] = int(out["cooldown_seconds"] or 0)
    try:
        channels = json.loads(out["delivery_channels"] or "[]")
    except (TypeError, ValueError):
        channels = []
    out["delivery_channels"] = [str(c) for c in channels] if isinstance(channels, list) else []
    for k in ("created_at", "updated_at", "last_evaluated_at", "last_triggered_at"):
        out[k] = to_iso(out[k])
    return out


def _lock_user(cur, user_id: str, is_sqlite: bool) -> None:
    """Per-user advisory lock so the count check and insert can't interleave
    (same pattern as db.alerts.create_alert)."""
    if not is_sqlite:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("alerts:" + str(user_id),))


def count_active(user_id: str, *, conn=None) -> int:
    """Enabled rules plus enabled legacy alerts: one plan limit covers both."""
    own = conn is None
    conn, is_sqlite = (conn, isinstance(conn, sqlite3.Connection)) if conn is not None else _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT COUNT(*) FROM hsf_alert_rules WHERE user_id = %s AND enabled", is_sqlite),
                    (user_id,))
        row = cur.fetchone()
        n = int((list(row.values())[0] if isinstance(row, dict) else row[0]) or 0)
        n += _legacy_enabled_count(cur, user_id, is_sqlite)
        cur.close()
        return n
    finally:
        if own:
            _close(conn, is_sqlite)


def _legacy_enabled_count(cur, user_id: str, is_sqlite: bool) -> int:
    if is_sqlite:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_alerts'")
        if cur.fetchone() is None:
            return 0
    else:
        cur.execute("SELECT to_regclass('public.user_alerts') IS NOT NULL AS ok")
        row = cur.fetchone()
        if not (row["ok"] if isinstance(row, dict) else row[0]):
            return 0
    cur.execute(_sql("SELECT COUNT(*) FROM user_alerts WHERE user_id = %s AND enabled", is_sqlite), (user_id,))
    row = cur.fetchone()
    return int((list(row.values())[0] if isinstance(row, dict) else row[0]) or 0)


def create_rule(user_id: str, *, rule_type: str, operator: str, threshold: Optional[float],
                value: Optional[str], ticker: Optional[str], watchlist_id: Optional[int],
                delivery_channels: Sequence[str], cooldown_seconds: int, enabled: bool = True,
                max_active: Optional[int] = None) -> Dict[str, Any]:
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        _lock_user(cur, user_id, is_sqlite)
        cur.execute(_sql("SELECT 1 FROM hsf_alert_rules WHERE user_id = %s AND rule_type = %s "
                         "AND COALESCE(ticker, '') = %s AND COALESCE(watchlist_id, 0) = %s "
                         "AND COALESCE(threshold, -1e300) = %s AND COALESCE(value, '') = %s", is_sqlite),
                    (user_id, rule_type, ticker or "", int(watchlist_id or 0),
                     float(threshold) if threshold is not None else -1e300, value or ""))
        if cur.fetchone() is not None:
            conn.rollback()
            raise ValueError("You already have this alert rule.")
        if enabled and max_active is not None:
            if count_active(user_id, conn=conn) >= int(max_active):
                conn.rollback()
                noun = "alert" if int(max_active) == 1 else "alerts"
                raise RuleLimitReached(f"You've reached the maximum of {int(max_active)} active {noun} on your plan.")
        now = now_iso()
        cur.execute(_sql(
            "INSERT INTO hsf_alert_rules (user_id, watchlist_id, ticker, rule_type, operator, threshold, value, "
            "enabled, delivery_channels, cooldown_seconds, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id", is_sqlite),
            (user_id, watchlist_id, ticker, rule_type, operator, threshold, value, bool(enabled),
             json.dumps(list(delivery_channels)), int(cooldown_seconds), now, now))
        row = cur.fetchone()
        new_id = int(row["id"] if isinstance(row, dict) else row[0])
        conn.commit()
        cur.close()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _close(conn, is_sqlite)
    rule = get_rule(user_id, new_id)
    assert rule is not None
    return rule


def list_rules(user_id: str) -> List[Dict[str, Any]]:
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql(f"SELECT {', '.join(RULE_COLUMNS)} FROM hsf_alert_rules WHERE user_id = %s "
                         "ORDER BY created_at DESC, id DESC", is_sqlite), (user_id,))
        rows = _rows(cur)
        cur.close()
        conn.commit()
        return [_rule_out(r) for r in rows]
    finally:
        _close(conn, is_sqlite)


def get_rule(user_id: str, rule_id: int) -> Optional[Dict[str, Any]]:
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql(f"SELECT {', '.join(RULE_COLUMNS)} FROM hsf_alert_rules WHERE id = %s AND user_id = %s",
                         is_sqlite), (int(rule_id), user_id))
        rows = _rows(cur)
        cur.close()
        conn.commit()
        return _rule_out(rows[0]) if rows else None
    finally:
        _close(conn, is_sqlite)


UPDATABLE = ("threshold", "value", "enabled", "delivery_channels", "cooldown_seconds")


def update_rule(user_id: str, rule_id: int, changes: Dict[str, Any],
                max_active: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Apply `changes` (keys in UPDATABLE). Changing the threshold or value resets the
    rule's per-ticker state, so the new condition starts from a fresh baseline."""
    sets = {k: v for k, v in changes.items() if k in UPDATABLE}
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        _lock_user(cur, user_id, is_sqlite)
        cur.execute(_sql("SELECT enabled FROM hsf_alert_rules WHERE id = %s AND user_id = %s", is_sqlite),
                    (int(rule_id), user_id))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            return None
        was_enabled = bool(row["enabled"] if isinstance(row, dict) else row[0])
        if sets.get("enabled") and not was_enabled and max_active is not None:
            if count_active(user_id, conn=conn) >= int(max_active):
                conn.rollback()
                noun = "alert" if int(max_active) == 1 else "alerts"
                raise RuleLimitReached(f"You've reached the maximum of {int(max_active)} active {noun} on your plan.")
        if sets:
            values = [json.dumps(list(v)) if k == "delivery_channels" else (bool(v) if k == "enabled" else v)
                      for k, v in sets.items()]
            assign = ", ".join(f"{k} = %s" for k in sets)
            cur.execute(_sql(f"UPDATE hsf_alert_rules SET {assign}, updated_at = %s WHERE id = %s AND user_id = %s",
                             is_sqlite), (*values, now_iso(), int(rule_id), user_id))
            if "threshold" in sets or "value" in sets:
                cur.execute(_sql("DELETE FROM hsf_alert_rule_state WHERE rule_id = %s", is_sqlite), (int(rule_id),))
        conn.commit()
        cur.close()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _close(conn, is_sqlite)
    return get_rule(user_id, rule_id)


def delete_rule(user_id: str, rule_id: int) -> bool:
    """Deletes the rule and its state. Its events stay as history (rule_id kept)."""
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("DELETE FROM hsf_alert_rules WHERE id = %s AND user_id = %s", is_sqlite),
                    (int(rule_id), user_id))
        gone = (cur.rowcount or 0) > 0
        if gone:
            cur.execute(_sql("DELETE FROM hsf_alert_rule_state WHERE rule_id = %s", is_sqlite), (int(rule_id),))
        conn.commit()
        cur.close()
        return gone
    finally:
        _close(conn, is_sqlite)


def disable_rules_for_watchlist(user_id: str, watchlist_id: int) -> int:
    """A deleted watchlist's rules are switched off (kept, so the user sees why)."""
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("UPDATE hsf_alert_rules SET enabled = %s, updated_at = %s "
                         "WHERE user_id = %s AND watchlist_id = %s AND enabled", is_sqlite),
                    (False, now_iso(), user_id, int(watchlist_id)))
        n = cur.rowcount or 0
        conn.commit()
        cur.close()
        return n
    finally:
        _close(conn, is_sqlite)


# ---- evaluator reads --------------------------------------------------------------------------------
def load_enabled_rules() -> List[Dict[str, Any]]:
    """Every enabled rule with its owner's plan fields, newest first per user."""
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT r.id, r.user_id, r.watchlist_id, r.ticker, r.rule_type, r.operator, r.threshold, "
                    "r.value, r.enabled, r.delivery_channels, r.cooldown_seconds, r.created_at, r.updated_at, "
                    "r.last_evaluated_at, r.last_triggered_at, u.tier AS owner_tier, u.is_admin AS owner_is_admin, "
                    "u.email_verified AS owner_email_verified, u.is_active AS owner_is_active "
                    "FROM hsf_alert_rules r LEFT JOIN users u ON lower(u.username) = lower(r.user_id) "
                    "WHERE r.enabled ORDER BY r.user_id, r.created_at DESC, r.id DESC")
        rows = _rows(cur)
        legacy: Dict[str, int] = {}
        if _legacy_table(cur, is_sqlite):
            cur.execute("SELECT user_id, COUNT(*) AS n FROM user_alerts WHERE enabled GROUP BY user_id")
            legacy = {str(r["user_id"]): int(r["n"] or 0) for r in _rows(cur)}
        cur.close()
        conn.commit()
    finally:
        _close(conn, is_sqlite)
    out = []
    for r in rows:
        rule = _rule_out(r)
        rule.update(owner_tier=str(r.get("owner_tier") or "basic").lower(),
                    owner_is_admin=bool(r.get("owner_is_admin")),
                    owner_email_verified=bool(r.get("owner_email_verified")),
                    owner_is_active=r.get("owner_is_active") is not False and r.get("owner_is_active") != 0,
                    owner_legacy_alerts=legacy.get(str(r["user_id"]), 0))
        out.append(rule)
    return out


def _legacy_table(cur, is_sqlite: bool) -> bool:
    if is_sqlite:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_alerts'")
        return cur.fetchone() is not None
    cur.execute("SELECT to_regclass('public.user_alerts') IS NOT NULL AS ok")
    row = cur.fetchone()
    return bool(row["ok"] if isinstance(row, dict) else row[0])


def watchlist_tickers(pairs: Iterable[Tuple[str, int]]) -> Dict[int, List[str]]:
    """Tickers of each (owner, watchlist id) in one query; a watchlist that is gone
    or belongs to someone else is absent from the result."""
    wanted = {(str(u), int(w)) for u, w in pairs}
    if not wanted:
        return {}
    ids = sorted({w for _, w in wanted})
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        marks = ", ".join(["%s"] * len(ids))
        cur.execute(_sql(f"SELECT wl.id AS wid, wl.user_id AS owner, wi.ticker AS ticker FROM watchlists wl "
                         f"LEFT JOIN watchlist_items wi ON wi.watchlist_id = wl.id WHERE wl.id IN ({marks})",
                         is_sqlite), tuple(ids))
        rows = _rows(cur)
        cur.close()
        conn.commit()
    finally:
        _close(conn, is_sqlite)
    out: Dict[int, List[str]] = {}
    for r in rows:
        wid, owner = int(r["wid"]), str(r["owner"])
        if (owner, wid) not in wanted:
            continue
        out.setdefault(wid, [])
        if r.get("ticker"):
            out[wid].append(str(r["ticker"]).upper())
    return out


def load_states(rule_ids: Sequence[int]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    if not rule_ids:
        return {}
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        marks = ", ".join(["%s"] * len(rule_ids))
        cur.execute(_sql("SELECT rule_id, ticker, last_value, last_flag, last_text, observation_id, "
                         f"last_triggered_at FROM hsf_alert_rule_state WHERE rule_id IN ({marks})", is_sqlite),
                    tuple(int(i) for i in rule_ids))
        rows = _rows(cur)
        cur.close()
        conn.commit()
    finally:
        _close(conn, is_sqlite)
    out = {}
    for r in rows:
        flag = r.get("last_flag")
        out[(int(r["rule_id"]), str(r["ticker"]))] = {
            "last_value": float(r["last_value"]) if r.get("last_value") is not None else None,
            "last_flag": None if flag is None else bool(flag),
            "last_text": r.get("last_text"),
            "observation_id": r.get("observation_id"),
            "last_triggered_at": to_iso(r.get("last_triggered_at")),
        }
    return out


# ---- evaluator writes -------------------------------------------------------------------------------
EVENT_FIELDS = ("user_id", "rule_id", "watchlist_id", "ticker", "rule_type", "operator", "threshold",
                "trigger_value", "previous_value", "hsf_score", "setup", "message", "observation_id",
                "market_data_as_of", "triggered_at", "delivery")


def save_evaluation(*, states: List[Dict[str, Any]], events: List[Dict[str, Any]],
                    evaluated_rule_ids: Sequence[int], triggered_rule_ids: Sequence[int],
                    now: str) -> List[Dict[str, Any]]:
    """One transaction: insert new events (duplicates by (rule, ticker, observation)
    are skipped), upsert per-ticker state, stamp the rules. Returns the events that
    were actually inserted, with their ids. Any failure rolls everything back, so a
    pass either fully happened or can be retried."""
    conn, is_sqlite = _connect()
    inserted: List[Dict[str, Any]] = []
    try:
        cur = conn.cursor()
        for ev in events:
            row = {k: ev.get(k) for k in EVENT_FIELDS}
            row["delivery"] = json.dumps(row.get("delivery") or {})
            cols = ", ".join(EVENT_FIELDS)
            marks = ", ".join(["%s"] * len(EVENT_FIELDS))
            cur.execute(_sql(f"INSERT INTO hsf_alert_rule_events ({cols}) VALUES ({marks}) "
                             "ON CONFLICT (rule_id, ticker, observation_id) DO NOTHING RETURNING id", is_sqlite),
                        tuple(row[k] for k in EVENT_FIELDS))
            got = cur.fetchone()
            if got is not None:
                inserted.append({**ev, "id": int(got["id"] if isinstance(got, dict) else got[0])})
        for st in states:
            cur.execute(_sql(
                "INSERT INTO hsf_alert_rule_state (rule_id, ticker, last_value, last_flag, last_text, observation_id, "
                "last_triggered_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (rule_id, ticker) DO UPDATE SET last_value = excluded.last_value, "
                "last_flag = excluded.last_flag, last_text = excluded.last_text, "
                "observation_id = excluded.observation_id, last_triggered_at = excluded.last_triggered_at, "
                "updated_at = excluded.updated_at", is_sqlite),
                (int(st["rule_id"]), st["ticker"], st.get("last_value"),
                 None if st.get("last_flag") is None else bool(st["last_flag"]), st.get("last_text"),
                 st.get("observation_id"), st.get("last_triggered_at"), now))
        ids = sorted({int(i) for i in evaluated_rule_ids})
        if ids:
            marks = ", ".join(["%s"] * len(ids))
            cur.execute(_sql(f"UPDATE hsf_alert_rules SET last_evaluated_at = %s WHERE id IN ({marks})", is_sqlite),
                        (now, *ids))
        fired = sorted({int(e["rule_id"]) for e in inserted} & {int(i) for i in triggered_rule_ids})
        if fired:
            marks = ", ".join(["%s"] * len(fired))
            cur.execute(_sql(f"UPDATE hsf_alert_rules SET last_triggered_at = %s WHERE id IN ({marks})", is_sqlite),
                        (now, *fired))
        conn.commit()
        cur.close()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        _close(conn, is_sqlite)
    return inserted


def set_delivery(event_id: int, delivery: Dict[str, str], *, attempts_inc: int = 0,
                 error: Optional[str] = None) -> None:
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("UPDATE hsf_alert_rule_events SET delivery = %s, "
                         "delivery_attempts = delivery_attempts + %s, last_delivery_error = %s WHERE id = %s",
                         is_sqlite), (json.dumps(delivery), int(attempts_inc), (error or None), int(event_id)))
        conn.commit()
        cur.close()
    finally:
        _close(conn, is_sqlite)


def events_to_retry(*, since_iso: str, before_iso: str, max_attempts: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Events with a failed email delivery, triggered in [since, before) and under the attempt cap."""
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql("SELECT * FROM hsf_alert_rule_events WHERE triggered_at >= %s AND triggered_at < %s "
                         "AND delivery_attempts < %s AND delivery LIKE %s ORDER BY triggered_at ASC LIMIT %s",
                         is_sqlite), (since_iso, before_iso, int(max_attempts), '%"failed"%', int(limit)))
        rows = _rows(cur)
        cur.close()
        conn.commit()
    finally:
        _close(conn, is_sqlite)
    return [_event_out(r) for r in rows]


# ---- events (API reads) -----------------------------------------------------------------------------
def _event_out(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)
    for k in ("id", "rule_id", "watchlist_id", "delivery_attempts"):
        if out.get(k) is not None:
            out[k] = int(out[k])
    for k in ("threshold", "trigger_value", "previous_value", "hsf_score"):
        if out.get(k) is not None:
            out[k] = float(out[k])
    for k in ("triggered_at", "market_data_as_of"):
        out[k] = to_iso(out.get(k))
    try:
        d = json.loads(out.get("delivery") or "{}")
    except (TypeError, ValueError):
        d = {}
    out["delivery"] = d if isinstance(d, dict) else {}
    return out


def list_events(user_id: str, *, limit: int, ticker: Optional[str] = None, rule_id: Optional[int] = None,
                watchlist_id: Optional[int] = None, after: Optional[str] = None,
                before: Optional[str] = None) -> List[Dict[str, Any]]:
    """The user's rule events, newest first. `before` is inclusive (the caller
    drops the cursor row itself, so equal timestamps aren't skipped)."""
    where, args = ["user_id = %s"], [user_id]
    for col, val in (("ticker", ticker), ("rule_id", rule_id), ("watchlist_id", watchlist_id)):
        if val is not None:
            where.append(f"{col} = %s")
            args.append(val)
    if after:
        where.append("triggered_at >= %s")
        args.append(after)
    if before:
        where.append("triggered_at <= %s")
        args.append(before)
    conn, is_sqlite = _connect()
    try:
        cur = conn.cursor()
        cur.execute(_sql(f"SELECT * FROM hsf_alert_rule_events WHERE {' AND '.join(where)} "
                         "ORDER BY triggered_at DESC, id DESC LIMIT %s", is_sqlite), (*args, int(limit)))
        rows = _rows(cur)
        cur.close()
        conn.commit()
    finally:
        _close(conn, is_sqlite)
    return [_event_out(r) for r in rows]
