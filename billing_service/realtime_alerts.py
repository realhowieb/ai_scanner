"""Real-time price-alert worker — runs inside the always-on Render billing service.

The scheduled cron checks alerts a few times a day; day traders need price
alerts in seconds-to-minutes. This worker polls Alpaca snapshots for all
enabled price alerts every REALTIME_POLL_SECONDS (default 60) during extended
market hours and fires immediately: in-app event + email (verified Pro+ only).

Deliberately self-contained (psycopg2 + httpx + stdlib): the billing service's
environment has no streamlit/pandas, so nothing here imports the main app.
Coordination with the cron is via user_alerts.last_fired_at — both paths mark
it, so the shared per-alert throttle prevents double-firing.

Enable with REALTIME_ALERTS_ENABLED=1 plus DATABASE_URL, ALPACA_API_KEY_ID,
ALPACA_API_SECRET_KEY, and SMTP_* in the Render environment (a separate secret
store from Streamlit Cloud and GitHub Actions).
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

POLL_SECONDS = int(os.getenv("REALTIME_POLL_SECONDS", "60") or "60")
THROTTLE_HOURS = float(os.getenv("ALERT_THROTTLE_HOURS", "12") or "12")
EMAIL_TIERS = ("pro", "premium", "admin")


def _log(msg: str) -> None:
    print(f"[realtime_alerts] {msg}", flush=True)


def market_session_open(now_utc: Optional[datetime] = None) -> bool:
    """True during extended US market hours (4:00-20:00 ET, Mon-Fri)."""
    now = now_utc or datetime.now(timezone.utc)
    et = now.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    return 4 * 60 <= minutes < 20 * 60


def crossed(direction: str, last: float, threshold: float) -> bool:
    """Whether a price condition is met."""
    if direction == "below":
        return last <= threshold
    return last >= threshold  # default/above


# --------------------------- data access (psycopg2) ---------------------------


def _conn():
    import psycopg2

    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return None
    return psycopg2.connect(url)


def _due_price_alerts(conn) -> List[Dict[str, Any]]:
    """Enabled price alerts past their throttle, with the owner's email gates."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.id, a.user_id, a.ticker, a.threshold, a.direction, a.alert_type,
               COALESCE(u.email_verified, FALSE), COALESCE(u.tier, 'basic')
        FROM user_alerts a
        LEFT JOIN users u ON u.username = a.user_id
        WHERE a.enabled
          AND a.alert_type IN ('price', 'move', 'rvol')
          AND a.ticker IS NOT NULL
          AND a.threshold IS NOT NULL
          AND (a.last_fired_at IS NULL
               OR a.last_fired_at < NOW() - make_interval(hours => %s))
        """,
        (THROTTLE_HOURS,),
    )
    rows = cur.fetchall()
    cur.close()
    return [
        {
            "id": r[0],
            "user_id": r[1],
            "ticker": str(r[2]).upper(),
            "threshold": float(r[3]),
            "direction": (r[4] or "above").lower(),
            "alert_type": str(r[5] or "price").lower(),
            "email_verified": bool(r[6]),
            "tier": str(r[7]).lower(),
        }
        for r in rows
    ]


def _record_fire(conn, alert: Dict[str, Any], message: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alert_events (user_id, alert_id, ticker, message) VALUES (%s, %s, %s, %s)",
        (alert["user_id"], alert["id"], alert["ticker"], message),
    )
    cur.execute("UPDATE user_alerts SET last_fired_at = NOW() WHERE id = %s", (alert["id"],))
    conn.commit()
    cur.close()


# ------------------------------ quotes (httpx) --------------------------------


def _latest_snapshots(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Per ticker: {'last', 'prev_close', 'volume'} from Alpaca snapshots."""
    key = os.getenv("ALPACA_API_KEY_ID", "").strip()
    secret = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    if not key or not secret or not tickers:
        return {}
    base = (os.getenv("ALPACA_DATA_URL", "").strip() or "https://data.alpaca.markets").rstrip("/")

    import httpx

    try:
        resp = httpx.get(
            f"{base}/v2/stocks/snapshots",
            params={"symbols": ",".join(sorted(set(tickers)))},
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        _log(f"quote fetch failed: {type(e).__name__}: {e}")
        return {}
    def _f(val):
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    out: Dict[str, Dict[str, Optional[float]]] = {}
    if isinstance(data, dict):
        for sym, snap in data.items():
            if not isinstance(snap, dict):
                continue
            trade = snap.get("latestTrade") or {}
            minute = snap.get("minuteBar") or {}
            daily = snap.get("dailyBar") or {}
            prev = snap.get("prevDailyBar") or {}
            last = _f(trade.get("p")) or _f(minute.get("c")) or _f(daily.get("c"))
            if last is None:
                continue
            out[str(sym).upper()] = {
                "last": last,
                "prev_close": _f(prev.get("c")),
                "volume": _f(daily.get("v")) or _f(minute.get("v")),
            }
    return out


# ------------------------------- email (smtp) ---------------------------------


def _send_email(to_address: str, subject: str, body: str) -> bool:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    host = os.getenv("SMTP_HOST", "").strip()
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASS", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip() or user
    port = int(os.getenv("SMTP_PORT", "587") or "587")
    if not host or not user or not password:
        return False
    disclaimer = (
        "Informational and educational purposes only — not financial advice. "
        "Trading involves risk of loss."
    )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"HSFinest.AI — {subject}"
    msg["From"] = sender
    msg["To"] = to_address
    msg.attach(MIMEText(f"HSFinest.AI alert\n\n{body}\n\n{disclaimer}", "plain"))
    msg.attach(
        MIMEText(
            f"<p><strong>⚡ HSFinest.AI</strong></p><pre>{body}</pre>"
            f"<p style='color:#aaa;font-size:11px'>{disclaimer}</p>",
            "html",
        )
    )
    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(user, password)
            server.sendmail(sender, [to_address], msg.as_string())
        return True
    except Exception as e:
        _log(f"email to {to_address} failed: {type(e).__name__}: {e}")
        return False


# Daily-cached 20d average volume per ticker (for RVOL alerts). One bars call
# per ticker per day; the denominator moves slowly.
_AVG_VOL: Dict[str, tuple] = {}


def _avg_volume(ticker: str) -> Optional[float]:
    today = datetime.now(timezone.utc).date()
    cached = _AVG_VOL.get(ticker)
    if cached and cached[0] == today:
        return cached[1]
    key = os.getenv("ALPACA_API_KEY_ID", "").strip()
    secret = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    if not key or not secret:
        return None
    base = (os.getenv("ALPACA_DATA_URL", "").strip() or "https://data.alpaca.markets").rstrip("/")

    import httpx

    try:
        resp = httpx.get(
            f"{base}/v2/stocks/bars",
            params={
                "symbols": ticker,
                "timeframe": "1Day",
                "start": (datetime.now(timezone.utc) - timedelta(days=45)).strftime("%Y-%m-%d"),
                "limit": 30,
                "adjustment": "raw",
                "feed": "iex",
            },
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=10.0,
        )
        resp.raise_for_status()
        bars = ((resp.json() or {}).get("bars") or {}).get(ticker) or []
        vols = [float(b.get("v") or 0) for b in bars[-21:-1]] or [
            float(b.get("v") or 0) for b in bars
        ]
        avg = sum(vols) / len(vols) if vols else None
        if avg and avg > 0:
            _AVG_VOL[ticker] = (today, avg)
            return avg
    except Exception as e:
        _log(f"avg volume fetch failed for {ticker}: {type(e).__name__}: {e}")
    return None


def evaluate_alert(alert: Dict[str, Any], snap: Dict[str, Optional[float]]) -> Optional[str]:
    """Message when the alert condition is met against a snapshot, else None."""
    last = snap.get("last")
    if last is None:
        return None
    atype = alert.get("alert_type", "price")
    threshold = alert["threshold"]
    if atype == "price":
        if crossed(alert["direction"], last, threshold):
            return (
                f"Price alert: {alert['ticker']} {last:,.2f} is "
                f"{alert['direction']} your {threshold:,.2f} target (live)"
            )
        return None
    if atype == "move":
        prev = snap.get("prev_close")
        if not prev:
            return None
        move = (last - prev) / prev * 100.0
        if abs(move) >= threshold:
            return (
                f"Move alert: {alert['ticker']} {move:+.1f}% today "
                f"(last {last:,.2f}, threshold ±{threshold:g}%)"
            )
        return None
    if atype == "rvol":
        volume = snap.get("volume")
        if not volume:
            return None
        avg = _avg_volume(alert["ticker"])
        if not avg:
            return None
        rvol = volume / avg
        if rvol >= threshold:
            return (
                f"RVOL alert: {alert['ticker']} trading {rvol:.1f}x its 20d "
                f"average volume (threshold {threshold:g}x)"
            )
        return None
    return None


# --------------------------------- the loop -----------------------------------


def check_once() -> int:
    """One evaluation pass. Returns number of alerts fired."""
    conn = _conn()
    if conn is None:
        return 0
    try:
        alerts = _due_price_alerts(conn)
        if not alerts:
            return 0
        snaps = _latest_snapshots([a["ticker"] for a in alerts])
        fired = 0
        for alert in alerts:
            snap = snaps.get(alert["ticker"])
            if not snap:
                continue
            message = evaluate_alert(alert, snap)
            if message is None:
                continue
            _record_fire(conn, alert, message)
            fired += 1
            if (
                "@" in alert["user_id"]
                and alert["email_verified"]
                and alert["tier"] in EMAIL_TIERS
            ):
                _send_email(alert["user_id"], "⚡ Live alert triggered", message)
        if fired:
            _log(f"fired {fired} live alert(s)")
        return fired
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_loop() -> None:
    """Poll forever (daemon thread). Errors are logged, never fatal."""
    _log(f"worker started (poll={POLL_SECONDS}s, throttle={THROTTLE_HOURS}h)")
    while True:
        try:
            if market_session_open():
                check_once()
        except Exception as e:
            _log(f"pass failed: {type(e).__name__}: {e}")
        time.sleep(POLL_SECONDS)


_worker_thread: Optional[threading.Thread] = None


def worker_status() -> Dict[str, Any]:
    """Introspection for /debug/status: is the worker enabled/alive?"""
    return {
        "enabled": os.getenv("REALTIME_ALERTS_ENABLED", "0").strip() == "1",
        "alive": bool(_worker_thread and _worker_thread.is_alive()),
        "poll_seconds": POLL_SECONDS,
        "throttle_hours": THROTTLE_HOURS,
        "market_open_now": market_session_open(),
        "alpaca_env": bool(os.getenv("ALPACA_API_KEY_ID", "").strip()),
        "smtp_env": bool(os.getenv("SMTP_HOST", "").strip()),
    }


def start_background_worker() -> bool:
    """Start the polling thread when enabled; returns whether it started."""
    global _worker_thread
    if os.getenv("REALTIME_ALERTS_ENABLED", "0").strip() != "1":
        _log("disabled (set REALTIME_ALERTS_ENABLED=1 to enable)")
        return False
    _worker_thread = threading.Thread(target=run_loop, name="realtime-alerts", daemon=True)
    _worker_thread.start()
    return True
