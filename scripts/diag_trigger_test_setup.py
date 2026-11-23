"""Live test of breakout trigger alerts (Howard approved 2026-11-22): signs in as a Live
acceptance test account through the public API and creates up to 3 'trigger' alerts on
scan names trading just under their trade-plan trigger (the 20-day high). Run through the
Diagnostics workflow; logs are public, so no emails, tokens or ids beyond alert ids print.

TRIGGER_TEST_MODE=cleanup deletes this test's trigger alerts again (all trigger alerts on
the test account).
"""
from __future__ import annotations

import os
import sys

import requests

API = (os.environ.get("API_BASE_URL") or "https://api.hsfinest.com").rstrip("/")
MAX_NEW = 3
BAND = (0.003, 0.025)   # 0.3% to 2.5% below the trigger


def login() -> dict:
    for role in ("ACC_PRO2", "ACC_PRO"):
        email, pw = os.environ.get(f"{role}_EMAIL"), os.environ.get(f"{role}_PASSWORD")
        if not email or not pw:
            continue
        r = requests.post(f"{API}/v1/auth/login", json={"email": email, "password": pw, "client": "acceptance"}, timeout=30)
        if r.status_code == 200:
            print(f"signed in as {role}")
            return {"Authorization": f"Bearer {r.json()['access_token']}"}
        print(f"{role} login HTTP {r.status_code}")
    sys.exit("no test account could sign in")


def main() -> None:
    h = login()
    alerts = requests.get(f"{API}/v1/alerts", headers=h, timeout=30).json()
    mine = [a for a in alerts.get("alerts", []) if a.get("type") == "trigger"]
    print(f"alerts used {alerts.get('used')}/{alerts.get('limit')}; trigger alerts now: "
          f"{[(a.get('id'), a.get('ticker'), a.get('threshold')) for a in mine]}")

    if os.environ.get("TRIGGER_TEST_MODE") == "cleanup":
        for a in mine:
            r = requests.delete(f"{API}/v1/alerts/{a['id']}", headers=h, timeout=30)
            print(f"deleted {a['id']} {a.get('ticker')}: HTTP {r.status_code}")
        return

    room = min(MAX_NEW - len(mine), int(alerts.get("limit") or 0) - int(alerts.get("used") or 0))
    if room <= 0:
        print("no room for new trigger alerts; nothing created")
        return
    scan = requests.get(f"{API}/v1/scans/latest", params={"limit": 100}, headers=h, timeout=60).json()
    print(f"scan_at {scan.get('scan_at')}, {len(scan.get('setups') or [])} setups")
    picks = []
    for s in (scan.get("setups") or [])[:60]:
        t = s.get("ticker")
        r = requests.get(f"{API}/v1/stocks/{t}/plan", headers=h, timeout=30)
        if r.status_code != 200:
            continue
        p = r.json()
        trig, live = p.get("trigger"), p.get("live_price") or p.get("scan_price")
        if not trig or not live:
            continue
        gap = (trig - live) / trig
        if BAND[0] <= gap <= BAND[1]:
            picks.append((gap, t, trig, live, (p.get("liquidity") or {}).get("dollar_volume_20d")))
    picks.sort()
    print("candidates (gap, ticker, trigger, live, $vol20d):", [(round(g * 100, 2), t, tr, lv, dv) for g, t, tr, lv, dv in picks[:10]])
    have = {a.get("ticker") for a in mine}
    for _, t, trig, live, _ in [p for p in picks if p[1] not in have][:room]:
        r = requests.post(f"{API}/v1/alerts", headers=h, timeout=30,
                          json={"type": "trigger", "ticker": t, "threshold": trig, "direction": "above"})
        print(f"created trigger alert {t} >= {trig} (live {live}): HTTP {r.status_code} "
              f"{r.json().get('id') if r.status_code == 201 else r.text[:200]}")


if __name__ == "__main__":
    main()
