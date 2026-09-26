"""Run 62 — product trust / freshness banner.

A compact, operational-only line that tells a user whether the market data
they are looking at is fresh, e.g.

    FULL U.S. MARKET · Last scan Fri Sep 25, 3:35 PM ET (2 h ago)
    Universe 11,533 tradable stocks · Market closed · ● Operational

Sources (read-only, both already persisted by production):
- the latest scheduled full-market run in the `runs` table
  (username "cron", label "US_MARKET") for the scan time and result count;
- the latest persisted system-health snapshot (Run 59) for the universe size
  and a user-facing status.

Only values that exist are shown; nothing is fabricated. Coverage % is not
persisted outside CI artifacts, so it is not shown. The status is derived only
from the user-relevant subsystems (universe, scanner, market data, database);
research, maturation, recovery and forward-experiment state never reach it.

The pure functions (`user_status`, `latest_market_scan`, `build_trust_info`)
take plain data so they are testable without Streamlit or a database.
"""
from __future__ import annotations

import datetime as _dt
import html
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

from analytics import market_calendar as mc

UTC = _dt.timezone.utc
MARKET_SCAN_USER = "cron"
MARKET_SCAN_LABEL = "US_MARKET"
UNAVAILABLE_TEXT = "Latest market scan unavailable"

# Health subsystems that decide whether market results are trustworthy.
USER_FACING_SUBSYSTEMS = ("universe", "scanner", "market_data", "database")
# A health snapshot older than this is too old to vouch for current results.
HEALTH_MAX_AGE = _dt.timedelta(hours=36)
# The universe count comes from a weekly-refreshed list; allow a week of age.
UNIVERSE_MAX_AGE = _dt.timedelta(days=8)

# level -> (label, shape). Shape + words carry the meaning, not colour alone.
STATUS_LABELS = {
    "ok": ("Operational", "●"),
    "limited": ("Limited", "▲"),
    "issue": ("Service issue", "■"),
    "unknown": ("Status unavailable", "○"),
}


def _parse_ts(value: Any) -> Optional[_dt.datetime]:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        dt = value
    else:
        try:
            dt = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def health_is_fresh(generated: Optional[_dt.datetime], now: _dt.datetime) -> bool:
    """A snapshot vouches for current results when it is < 36 h old, or was
    taken after the most recent completed session closed (so Friday night's
    snapshot still counts over a weekend or holiday)."""
    if generated is None:
        return False
    if now - generated <= HEALTH_MAX_AGE:
        return True
    today = now.astimezone(mc.ET).date()
    if mc.is_trading_day(today) and now >= mc.session_bounds_utc(today)[1]:
        last_day = today
    else:
        last_day = mc.previous_trading_day(today)
    last_close = mc.session_bounds_utc(last_day)[1]
    return generated >= last_close - _dt.timedelta(hours=1)


def user_status(health: Optional[Mapping[str, Any]], now: _dt.datetime) -> Dict[str, str]:
    """Translate the internal health snapshot into a user-facing status.

    ACTION_REQUIRED on any user-facing subsystem → "Service issue";
    DEGRADED → "Limited"; otherwise "Operational". A missing or stale
    snapshot → "Status unavailable". Internal-only subsystems (maturation,
    research capture, cohort parity, forward evidence, workflows, artifacts)
    are ignored, so research incidents never reach normal users.
    """
    if not health:
        return {"level": "unknown", "label": STATUS_LABELS["unknown"][0]}
    generated = _parse_ts(health.get("generated_at"))
    if not health_is_fresh(generated, now):
        return {"level": "unknown", "label": STATUS_LABELS["unknown"][0]}
    subs = health.get("subsystems") or {}
    states = [str((subs.get(name) or {}).get("status") or "UNKNOWN").upper() for name in USER_FACING_SUBSYSTEMS]
    if "ACTION_REQUIRED" in states:
        level = "issue"
    elif "DEGRADED" in states:
        level = "limited"
    elif all(s == "UNKNOWN" for s in states):
        level = "unknown"
    else:
        level = "ok"
    return {"level": level, "label": STATUS_LABELS[level][0]}


def latest_market_scan(runs: Optional[Sequence[Mapping[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The newest scheduled full-market run (cron · US_MARKET), or None."""
    best: Optional[Dict[str, Any]] = None
    best_ts: Optional[_dt.datetime] = None
    for r in runs or []:
        if str(r.get("username") or "").lower() != MARKET_SCAN_USER:
            continue
        if str(r.get("label") or "").upper() != MARKET_SCAN_LABEL:
            continue
        ts = _parse_ts(r.get("created_at"))
        if ts is None:
            continue
        if best_ts is None or ts > best_ts:
            best, best_ts = dict(r), ts
    if best is not None:
        best["created_at"] = best_ts
    return best


def market_session_label(now: _dt.datetime) -> str:
    """Plain-English U.S. market session (holiday-aware)."""
    et = now.astimezone(mc.ET)
    if not mc.is_trading_day(et.date()):
        return "Market closed"
    if mc.is_market_open(now):
        return "Market open"
    open_utc, close_utc = mc.session_bounds_utc(et.date())
    if now < open_utc and et.hour >= 4:
        return "Pre-market"
    if now >= close_utc and et.hour < 20:
        return "After hours"
    return "Market closed"


def _fmt_et(ts: _dt.datetime, now: _dt.datetime) -> str:
    et, now_et = ts.astimezone(mc.ET), now.astimezone(mc.ET)
    clock = et.strftime("%I:%M %p").lstrip("0")
    if et.date() == now_et.date():
        return f"{clock} ET"
    return f"{et.strftime('%a %b')} {et.day}, {clock} ET"


def _fmt_age(ts: _dt.datetime, now: _dt.datetime) -> str:
    mins = max(0, int((now - ts).total_seconds() // 60))
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    hours = mins // 60
    if hours < 48:
        return f"{hours} h ago"
    return f"{hours // 24} days ago"


def build_trust_info(runs: Optional[Sequence[Mapping[str, Any]]], health: Optional[Mapping[str, Any]],
                     now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """Everything the banner shows, as plain data (no fabrication)."""
    now = now or _dt.datetime.now(UTC)
    scan = latest_market_scan(runs)
    info: Dict[str, Any] = {
        "available": scan is not None,
        "universe_label": "Full U.S. market",
        "session": market_session_label(now),
        "status": user_status(health, now),
        "last_scan_at": None, "last_scan_text": None, "last_scan_age": None,
        "result_count": None, "universe_symbols": None,
    }
    if scan is not None:
        ts = scan["created_at"]
        info["last_scan_at"] = ts
        info["last_scan_text"] = _fmt_et(ts, now)
        info["last_scan_age"] = _fmt_age(ts, now)
        rc = scan.get("row_count")
        info["result_count"] = int(rc) if isinstance(rc, (int, float)) and rc > 0 else None
    if health:
        generated = _parse_ts(health.get("generated_at"))
        uni = ((health.get("subsystems") or {}).get("universe") or {}).get("metrics") or {}
        count = uni.get("symbol_count")
        if generated is not None and now - generated <= UNIVERSE_MAX_AGE and isinstance(count, int) and count > 0:
            info["universe_symbols"] = count
    return info


def banner_parts(info: Mapping[str, Any]) -> list:
    """The banner's text segments, in display order."""
    parts = []
    if info.get("available"):
        parts.append(f"Last scan {info['last_scan_text']} ({info['last_scan_age']})")
    else:
        parts.append(UNAVAILABLE_TEXT)
    if info.get("universe_symbols"):
        parts.append(f"Universe {info['universe_symbols']:,} tradable stocks")
    if info.get("result_count"):
        parts.append(f"{info['result_count']} ranked setups")
    parts.append(info.get("session") or "")
    return [p for p in parts if p]


# ---- Streamlit rendering ----------------------------------------------------------------------
_CSS = """
<style>
.hsf-trust{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;
  padding:8px 12px;margin:4px 0 10px;border:1px solid rgba(128,128,128,.28);
  border-radius:8px;font-size:.86rem;line-height:1.4}
.hsf-trust .hsf-trust-scope{font-weight:700;letter-spacing:.06em;font-size:.74rem;text-transform:uppercase;opacity:.85}
.hsf-trust .hsf-trust-part{white-space:normal}
.hsf-trust .hsf-trust-status{font-weight:600;margin-left:auto;white-space:nowrap}
.hsf-trust .hsf-lvl-ok{color:#3aa76d}.hsf-trust .hsf-lvl-limited{color:#d19a2a}
.hsf-trust .hsf-lvl-issue{color:#e0625c}.hsf-trust .hsf-lvl-unknown{opacity:.75}
@media (max-width:640px){.hsf-trust .hsf-trust-status{margin-left:0}}
</style>
"""


def banner_html(info: Mapping[str, Any]) -> str:
    level = (info.get("status") or {}).get("level", "unknown")
    label, shape = STATUS_LABELS.get(level, STATUS_LABELS["unknown"])
    segs = "".join(f'<span class="hsf-trust-part">{html.escape(p)}</span>' for p in banner_parts(info))
    return (
        f'{_CSS}<div class="hsf-trust" role="status" aria-label="Market data status">'
        f'<span class="hsf-trust-scope">{html.escape(str(info.get("universe_label") or ""))}</span>{segs}'
        f'<span class="hsf-trust-status hsf-lvl-{level}"><span aria-hidden="true">{shape}</span> '
        f'System: {html.escape(label)}</span></div>'
    )


if st is not None:
    @st.cache_data(ttl=120, show_spinner=False)
    def _load_runs() -> list:
        from db.runs import list_runs

        return list_runs(limit=25, include_snapshots=True, username=MARKET_SCAN_USER) or []

    @st.cache_data(ttl=300, show_spinner=False)
    def _load_health() -> Optional[dict]:
        from db.system_health import load_latest

        return load_latest()
else:  # pragma: no cover
    def _load_runs() -> list:
        return []

    def _load_health() -> Optional[dict]:
        return None


def render_trust_banner(*, show_methodology_link: bool = True) -> None:
    """Render the banner. Never raises; missing data degrades to the fallback text."""
    if st is None:
        return
    try:
        runs = _load_runs()
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("trust banner scan history", exc)
        runs = []
    try:
        health = _load_health()
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("trust banner health snapshot", exc)
        health = None
    try:
        st.markdown(banner_html(build_trust_info(runs, health)), unsafe_allow_html=True)
        if show_methodology_link:
            st.page_link("pages/methodology.py", label="How HSF works")
    except Exception as exc:
        from ui.safe_errors import report_error

        report_error("trust banner", exc)
