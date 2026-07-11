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
from datetime import datetime, timezone
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
        SELECT a.id, a.user_id, a.ticker, a.threshold, a.direction,
               COALESCE(u.email_verified, FALSE), COALESCE(u.tier, 'basic')
        FROM user_alerts a
        LEFT JOIN users u ON u.username = a.user_id
        WHERE a.enabled
          AND a.alert_type = 'price'
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
            "email_verified": bool(r[5]),
            "tier": str(r[6]).lower(),
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


def _latest_prices(tickers: List[str]) -> Dict[str, float]:
    """Latest trade price per ticker from Alpaca snapshots (extended hours)."""
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
    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for sym, snap in data.items():
            if not isinstance(snap, dict):
                continue
            for src in ("latestTrade", "minuteBar", "dailyBar"):
                node = snap.get(src) or {}
                px = node.get("p") if src == "latestTrade" else node.get("c")
                try:
                    if px is not None:
                        out[str(sym).upper()] = float(px)
                        break
                except (TypeError, ValueError):
                    continue
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
        prices = _latest_prices([a["ticker"] for a in alerts])
        fired = 0
        for alert in alerts:
            last = prices.get(alert["ticker"])
            if last is None:
                continue
            if not crossed(alert["direction"], last, alert["threshold"]):
                continue
            message = (
                f"Price alert: {alert['ticker']} {last:,.2f} is "
                f"{alert['direction']} your {alert['threshold']:,.2f} target (live)"
            )
            _record_fire(conn, alert, message)
            fired += 1
            if (
                "@" in alert["user_id"]
                and alert["email_verified"]
                and alert["tier"] in EMAIL_TIERS
            ):
                _send_email(alert["user_id"], "⚡ Price alert triggered", message)
        if fired:
            _log(f"fired {fired} price alert(s)")
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


def start_background_worker() -> bool:
    """Start the polling thread when enabled; returns whether it started."""
    if os.getenv("REALTIME_ALERTS_ENABLED", "0").strip() != "1":
        _log("disabled (set REALTIME_ALERTS_ENABLED=1 to enable)")
        return False
    thread = threading.Thread(target=run_loop, name="realtime-alerts", daemon=True)
    thread.start()
    return True
