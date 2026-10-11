"""GET /v1/me/export: everything HSF keeps about your account, as one JSON file.

Built from the same per-account reads the API already serves, so it shows exactly
what the app shows you. Never includes passwords, tokens, push tokens, paper-trading
keys or anyone else's data. A section that can't be read right now is null and named
in `unavailable`, so a partial export never looks complete.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, List

EXPORT_VERSION = 1
JOURNAL_LIMIT = 5000
EVENT_LIMIT = 500
RUN_LIMIT = 500


def build_export(user: str, me: Dict[str, Any]) -> Dict[str, Any]:
    from api import account as acct
    from api import alert_rules, devices, history, user_data

    def watchlists() -> List[Dict[str, Any]]:
        return [user_data.get_watchlist(user, int(w["id"])) for w in user_data.list_watchlists(user)]

    def journal() -> List[Dict[str, Any]]:
        from db.trades import list_trades

        return list_trades(user, limit=JOURNAL_LIMIT)

    sections: Dict[str, Callable[[], Any]] = {
        "watchlists": watchlists,
        "price_alerts": lambda: user_data.list_alerts(user),
        "alert_rules": lambda: alert_rules.list_rules(user),
        "alert_events": lambda: alert_rules.list_events(user, limit=EVENT_LIMIT),
        "journal": journal,
        "email_preferences": lambda: acct.get_email_prefs(user),
        "devices": lambda: devices.list_devices(user),
        "saved_scans": lambda: history.saved_runs(user, RUN_LIMIT, include_snapshots=False),
    }
    out: Dict[str, Any] = {
        "format": "hsf-account-export", "version": EXPORT_VERSION,
        "exported_at": dt.datetime.now(dt.timezone.utc),
        "account": {k: me.get(k) for k in ("email", "name", "plan", "plan_label", "email_verified", "alert_limit")},
    }
    unavailable: List[str] = []
    for name, read in sections.items():
        try:
            out[name] = read()
        except Exception:
            out[name] = None
            unavailable.append(name)
    out["unavailable"] = unavailable
    return out
