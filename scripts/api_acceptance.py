#!/usr/bin/env python3
"""HSF API acceptance journey (P1-59). Runs against any deployment.

    API_BASE_URL=https://... python scripts/api_acceptance.py --allow-writes \\
        --origin https://app.example.com --report acceptance.json

Test accounts come from the environment, never from arguments, and are never
printed: ACC_<ROLE>_EMAIL and ACC_<ROLE>_PASSWORD for ROLE in FREE, PRO, PRO2,
PREMIUM, ADMIN (PRO2 = a second Pro account for the isolation checks). Use
dedicated test accounts only. Roles without credentials are reported as BLOCKED.

Journey per account: login -> /me -> latest scan -> stock detail -> create
watchlist -> add tickers -> note -> custom scans (plan refusals, a one-ticker
scan and a watchlist scan, polled to completion) -> push device register /
list / refresh / remove -> create alert -> list -> disable -> delete -> delete
watchlist -> refresh rotation -> logout. Writes need --allow-writes;
everything the run creates is named "zz-acceptance-<run id>" and deleted at the
end (only by id, only records this run created). Alerts use a threshold that
cannot fire, so no email is sent.

Results: PASS / FAIL / BLOCKED per check, with evidence (status codes, counts,
timings), never tokens, passwords or response bodies with personal data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import httpx

ROLES = {"FREE": ("basic", 25, 1), "PRO": ("pro", 100, 5), "PRO2": ("pro", 100, 5),
         "PREMIUM": ("premium", 200, 25), "ADMIN": ("admin", 9999, 25)}
REQUIRED_PATHS = ["/healthz", "/v1/auth/login", "/v1/auth/refresh", "/v1/auth/logout", "/v1/me", "/v1/today",
                  "/v1/scans/latest", "/v1/stocks/{ticker}", "/v1/watchlists", "/v1/watchlists/{watchlist_id}",
                  "/v1/watchlists/{watchlist_id}/tickers", "/v1/watchlists/{watchlist_id}/tickers/{ticker}",
                  "/v1/alerts", "/v1/alerts/{alert_id}", "/v1/alerts/events",
                  "/readyz", "/v1/auth/signup", "/v1/auth/verify-email", "/v1/auth/password-reset",
                  "/v1/auth/password-reset/confirm", "/v1/me/password", "/v1/me/verify-email",
                  "/v1/me/email-preferences", "/v1/billing/checkout", "/v1/billing/portal",
                  "/v1/me/devices", "/v1/me/devices/{device_id}", "/v1/scans", "/v1/scans/{scan_id}",
                  "/v1/runs", "/v1/runs/{run_id}", "/v1/track-record", "/v1/track-record/daily", "/v1/earnings",
                  "/v1/brief", "/v1/day-trader", "/v1/day-trader/stair-steppers", "/v1/ai/summary", "/v1/ai/chat",
                  "/v1/ai/notes/{ticker}", "/v1/ai/brief-narrative", "/v1/journal", "/v1/journal/{trade_id}",
                  "/v1/journal/{trade_id}/close", "/v1/stocks/{ticker}/plan", "/v1/paper/account",
                  "/v1/paper/activity", "/v1/paper/orders", "/v1/alerts/types"]
# Operations Web v2 needs that share a path with an older one (a path check can't see them).
REQUIRED_OPERATIONS = [("delete", "/v1/scans/{scan_id}"), ("get", "/v1/alerts/types")]
# Plan floor for each paid feature: (method, path, body, lowest plan that gets 2xx). AI is checked only
# for refusals below Premium (a Premium call would spend money); paper only for status/activity.
PAID_CHECKS = [("GET", "/v1/runs", None, "pro"), ("GET", "/v1/track-record", None, "pro"),
               ("GET", "/v1/earnings?days=7", None, "pro"), ("GET", "/v1/day-trader?source=megacaps", None, "pro"),
               ("GET", "/v1/journal", None, "basic"), ("GET", "/v1/brief", None, "basic"),
               ("POST", "/v1/ai/summary", {}, "premium"), ("GET", "/v1/paper/account", None, "premium"),
               ("GET", "/v1/paper/activity", None, "premium")]
PLAN_ORDER = {"basic": 0, "pro": 1, "premium": 2, "admin": 3}
SCAN_TIMEOUT_S = float(os.environ.get("API_SCAN_TIMEOUT_S", "300"))
NEVER_FIRES = 999_999.0  # % move threshold no stock reaches


def _strict_json(text: str) -> Any:
    def bad(c):
        raise ValueError(f"non-standard JSON constant {c}")
    return json.loads(text, parse_constant=bad)


class Run:
    def __init__(self, base: str, timeout: float = 60.0):
        self.base = base.rstrip("/")
        self.http = httpx.Client(timeout=timeout)
        self.results: List[Dict[str, Any]] = []
        self.timings: Dict[str, List[float]] = {}
        self.run_id = uuid.uuid4().hex[:8]
        self.created: List[tuple] = []  # (headers, kind, id) for cleanup

    def req(self, method: str, path: str, *, label: Optional[str] = None, **kw) -> httpx.Response:
        t0 = time.perf_counter()
        r = self.http.request(method, self.base + path, **kw)
        self.timings.setdefault(label or f"{method} {path.split('?')[0]}", []).append(time.perf_counter() - t0)
        return r

    def check(self, area: str, name: str, ok: Optional[bool], evidence: str) -> bool:
        status = "BLOCKED" if ok is None else ("PASS" if ok else "FAIL")
        self.results.append({"area": area, "check": name, "status": status, "evidence": evidence})
        print(f"[{status:7}] {area} · {name} — {evidence}")
        return bool(ok)

    def guard(self, area: str, name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as e:  # a crash in one check is a FAIL, not an abort
            self.check(area, name, False, f"{type(e).__name__}: {str(e)[:160]}")


def creds(role: str):
    e, p = os.environ.get(f"ACC_{role}_EMAIL"), os.environ.get(f"ACC_{role}_PASSWORD")
    return (e, p) if e and p else None


def login(run: Run, role: str) -> Optional[Dict[str, Any]]:
    c = creds(role)
    if not c:
        return None
    r = run.req("POST", "/v1/auth/login", json={"email": c[0], "password": c[1], "client": "acceptance"},
                label="POST /v1/auth/login")
    if r.status_code != 200:
        run.check("auth", f"{role} login", False, f"HTTP {r.status_code}")
        return None
    pair = r.json()
    return {"role": role, "pair": pair, "h": {"Authorization": f"Bearer {pair['access_token']}"}}


# ---- unauthenticated -----------------------------------------------------------------------------
def deployment_checks(run: Run, origin: Optional[str]) -> None:
    A = "deployment"
    r = run.req("GET", "/healthz", label="cold GET /healthz")
    run.check(A, "/healthz liveness", r.status_code == 200 and r.json() == {"ok": True},
              f"HTTP {r.status_code}, first request {run.timings['cold GET /healthz'][0]*1000:.0f} ms")
    for _ in range(5):
        run.req("GET", "/healthz", label="warm GET /healthz")
    r = run.req("GET", "/readyz")
    run.check(A, "/readyz database readiness", r.status_code == 200 and r.json().get("database") == "ok"
              if r.status_code != 404 else False, f"HTTP {r.status_code} {r.text[:80]}")
    r = run.req("GET", "/docs")
    run.check(A, "/docs served", r.status_code == 200 and "swagger" in r.text.lower(), f"HTTP {r.status_code}")
    r = run.req("GET", "/openapi.json")
    spec_paths = (r.json() or {}).get("paths", {}) if r.status_code == 200 else {}
    paths = set(spec_paths)
    missing = [p for p in REQUIRED_PATHS if p not in paths]
    missing += [f"{m.upper()} {p}" for m, p in REQUIRED_OPERATIONS if m not in (spec_paths.get(p) or {})]
    run.check(A, "OpenAPI has every current endpoint", r.status_code == 200 and not missing,
              f"HTTP {r.status_code}, {len(paths)} paths, missing: {missing or 'none'}")
    if origin:
        r = run.req("OPTIONS", "/v1/me", headers={"Origin": origin, "Access-Control-Request-Method": "GET",
                                                 "Access-Control-Request-Headers": "authorization"})
        allowed = r.headers.get("access-control-allow-origin")
        run.check(A, f"CORS preflight from {origin}", r.status_code == 200 and allowed == origin,
                  f"HTTP {r.status_code}, allow-origin={allowed!r}")
        r = run.req("OPTIONS", "/v1/me", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
        run.check(A, "CORS refuses an unlisted origin", r.headers.get("access-control-allow-origin") is None,
                  f"HTTP {r.status_code}, allow-origin={r.headers.get('access-control-allow-origin')!r}")
    else:
        run.check(A, "CORS preflight for the frontend origin", None, "no --origin given")


def auth_rejections(run: Run, expired_token: Optional[str]) -> None:
    A = "auth"
    routes = [("GET", "/v1/me"), ("GET", "/v1/scans/latest"), ("GET", "/v1/stocks/AAPL"), ("GET", "/v1/watchlists"),
              ("GET", "/v1/alerts"), ("POST", "/v1/alerts")]
    codes = [run.req(m, p).status_code for m, p in routes]
    run.check(A, "protected routes reject a missing token", all(c == 401 for c in codes), f"codes {codes}")
    codes = [run.req(m, p, headers={"Authorization": "Bearer not.a.token"}).status_code for m, p in routes]
    run.check(A, "protected routes reject an invalid token", all(c == 401 for c in codes), f"codes {codes}")
    if expired_token:
        c = run.req("GET", "/v1/me", headers={"Authorization": f"Bearer {expired_token}"}).status_code
        run.check(A, "expired access token rejected", c == 401, f"HTTP {c}")
    else:
        run.check(A, "expired access token rejected", None, "needs a token minted with the server secret (isolated runs only)")
    r = run.req("POST", "/v1/auth/login", json={"email": f"nobody-{run.run_id}@example.invalid", "password": "wrong"})
    run.check(A, "invalid login rejected", r.status_code == 401, f"HTTP {r.status_code}")


# ---- per-account journey -----------------------------------------------------------------------
def journey(run: Run, s: Dict[str, Any], allow_writes: bool) -> None:
    role = s["role"]
    tier, cap, alert_limit = ROLES[role]
    h = s["h"]
    A = f"journey:{role}"

    me = run.req("GET", "/v1/me", headers=h).json()
    run.check(A, "/me plan", me.get("plan") == tier and me.get("alert_limit") == alert_limit,
              f"plan={me.get('plan')} alert_limit={me.get('alert_limit')}")

    r = run.req("GET", "/v1/scans/latest?limit=200", headers=h)
    scan = _strict_json(r.text)
    setups = scan.get("setups", [])
    run.check(A, "scan: strict JSON, cap", r.status_code == 200 and scan.get("max_results") == min(cap, 9999)
              and len(setups) <= cap, f"HTTP {r.status_code}, max_results={scan.get('max_results')}, rows={len(setups)}, "
              f"total={scan.get('total')}, limited={scan.get('limited')}, scan_at={scan.get('scan_at')}")
    scores = [x["score"] for x in setups]
    run.check(A, "scan: ranked by score", scores == sorted(scores, reverse=True), f"{len(scores)} rows")
    premium = tier in ("premium", "admin")
    hidden = all(x.get("prob") is None and "prebreakout" not in x.get("signals", []) for x in setups)
    run.check(A, "scan: PreBreakout visibility", hidden != premium or not setups,
              f"{'visible' if not hidden else 'hidden'} for {tier}")
    c = run.req("GET", "/v1/scans/latest?signal=prebreakout", headers=h).status_code
    run.check(A, "scan: signal=prebreakout gated", c == (200 if premium else 403), f"HTTP {c}")
    p1 = run.req("GET", "/v1/scans/latest?limit=5&offset=0", headers=h).json()["setups"]
    p2 = run.req("GET", "/v1/scans/latest?limit=5&offset=5", headers=h).json()["setups"]
    run.check(A, "scan: pagination consistent", [x["ticker"] for x in p1 + p2] == [x["ticker"] for x in setups[:10]],
              f"pages {len(p1)}+{len(p2)}")
    beyond = run.req("GET", f"/v1/scans/latest?limit=200&offset={cap}", headers=h).json()["setups"]
    run.check(A, "scan: offset can't pass the plan cap", not beyond, f"offset={cap} rows={len(beyond)}")
    hi = run.req("GET", "/v1/scans/latest?min_score=70&limit=200", headers=h).json()
    run.check(A, "scan: min_score filter", all(x["score"] >= 70 for x in hi["setups"]) and len(hi["setups"]) <= cap,
              f"rows={len(hi['setups'])} total={hi['total']}")

    ticker = setups[0]["ticker"] if setups else "AAPL"
    r = run.req("GET", f"/v1/stocks/{ticker}", headers=h)
    d = _strict_json(r.text)
    run.check(A, "stock detail", r.status_code == 200 and d.get("ticker") == ticker
              and (premium or d.get("prob") is None),
              f"HTTP {r.status_code}, score={d.get('hsf_score')}, bars={len(d.get('bars', []))}")
    c = run.req("GET", "/v1/stocks/ZZZZQ", headers=h)
    run.check(A, "unknown ticker", c.status_code == 200 and not c.json().get("in_latest_scan"), f"HTTP {c.status_code}")
    c = run.req("GET", "/v1/stocks/bad%20ticker", headers=h).status_code
    run.check(A, "invalid ticker 422", c == 422, f"HTTP {c}")

    for method, path, body, floor in PAID_CHECKS:
        allowed = PLAN_ORDER[tier] >= PLAN_ORDER[floor]
        if allowed and path.startswith("/v1/ai/"):
            continue                       # don't spend AI calls in acceptance runs
        r = run.req(method, path, headers=h, **({"json": body} if body is not None else {}))
        ok = (200 <= r.status_code < 300 or (r.status_code == 503 and path.startswith("/v1/paper"))) if allowed \
            else r.status_code == 403
        run.check(A, f"plan gate {path.split('?')[0]} ({floor}+)", ok, f"HTTP {r.status_code} for {tier}")

    if not allow_writes:
        run.check(A, "watchlist/alert writes", None, "run without --allow-writes")
        return
    name = f"zz-acceptance-{run.run_id}-{role.lower()}"
    r = run.req("POST", "/v1/watchlists", headers=h, json={"name": name})
    ok = run.check(A, "create watchlist", r.status_code == 201, f"HTTP {r.status_code}")
    if not ok:
        return
    wid = r.json()["id"]
    run.created.append((h, "watchlist", wid))
    c = run.req("POST", "/v1/watchlists", headers=h, json={"name": name.upper()}).status_code
    run.check(A, "duplicate watchlist name 409", c == 409, f"HTTP {c}")
    r = run.req("POST", f"/v1/watchlists/{wid}/tickers", headers=h, json={"tickers": [ticker, ticker.lower(), "MSFT", "bad!"]})
    body = r.json()
    run.check(A, "bulk add tickers", r.status_code == 200 and "BAD!" in body.get("invalid", [])
              and ticker in body.get("added", []), f"added={len(body.get('added', []))} invalid={len(body.get('invalid', []))}")
    c = run.req("PATCH", f"/v1/watchlists/{wid}/tickers/{ticker}", headers=h, json={"note": "acceptance note"}).status_code
    items = run.req("GET", f"/v1/watchlists/{wid}", headers=h).json().get("items", [])
    run.check(A, "save note", c == 204 and any(i["ticker"] == ticker and i["note"] == "acceptance note" for i in items),
              f"HTTP {c}, items={len(items)}")
    c = run.req("DELETE", f"/v1/watchlists/{wid}/tickers/MSFT", headers=h).status_code
    run.check(A, "remove ticker", c == 204, f"HTTP {c}")

    custom_scans(run, s, A, tier, ticker, wid)
    devices(run, s, A)

    before = run.req("GET", "/v1/alerts", headers=h).json()
    r = run.req("POST", "/v1/alerts", headers=h, json={"type": "move", "ticker": ticker, "threshold": NEVER_FIRES})
    if before["used"] >= before["limit"]:
        run.check(A, "alert limit enforced", r.status_code == 403, f"used={before['used']}/{before['limit']} HTTP {r.status_code}")
    else:
        ok = run.check(A, "create alert", r.status_code == 201, f"HTTP {r.status_code}")
        if ok:
            aid = r.json()["id"]
            run.created.append((h, "alert", aid))
            listed = run.req("GET", "/v1/alerts", headers=h).json()
            run.check(A, "view alert", any(a["id"] == aid for a in listed["alerts"]) and listed["used"] == before["used"] + 1,
                      f"used={listed['used']}/{listed['limit']}")
            c = run.req("POST", "/v1/alerts", headers=h, json={"type": "move", "ticker": ticker, "threshold": NEVER_FIRES}).status_code
            run.check(A, "duplicate alert rejected", c in (409, 403), f"HTTP {c}")
            r = run.req("PATCH", f"/v1/alerts/{aid}", headers=h, json={"enabled": False})
            run.check(A, "disable alert", r.status_code == 200 and r.json()["enabled"] is False, f"HTTP {r.status_code}")
            c = run.req("DELETE", f"/v1/alerts/{aid}", headers=h).status_code
            run.check(A, "delete alert", c == 204, f"HTTP {c}")
            run.created.remove((h, "alert", aid))
    c = run.req("POST", "/v1/alerts", headers=h, json={"type": "price", "ticker": ticker, "threshold": 0, "direction": "above"}).status_code
    run.check(A, "invalid alert input 422", c == 422, f"HTTP {c}")
    c = run.req("DELETE", f"/v1/watchlists/{wid}", headers=h).status_code
    run.check(A, "remove test watchlist", c == 204, f"HTTP {c}")
    run.created.remove((h, "watchlist", wid))


def _poll_scan(run: Run, h: Dict[str, str], scan_id: str) -> Dict[str, Any]:
    deadline = time.time() + SCAN_TIMEOUT_S
    job: Dict[str, Any] = {}
    while time.time() < deadline:
        job = run.req("GET", f"/v1/scans/{scan_id}", headers=h, label="GET /v1/scans/{scan_id}").json()
        if job.get("status") in ("complete", "failed"):
            return job
        time.sleep(2)
    return job


def custom_scans(run: Run, s: Dict[str, Any], A: str, tier: str, ticker: str, wid: int) -> None:
    """Custom Scan through the API: plan rules, a one-ticker scan and a watchlist scan."""
    h = s["h"]
    expect = {"basic": {"nasdaq": 403, "us_market": 403}, "pro": {"us_market": 403}}.get(tier, {})
    for universe, code in expect.items():
        c = run.req("POST", "/v1/scans", headers=h, json={"universe": universe}).status_code
        run.check(A, f"scan: {universe} refused for {tier}", c == code, f"HTTP {c}")
    cap = ROLES[s["role"]][1]
    c = run.req("POST", "/v1/scans", headers=h, json={"universe": "sp500", "filters": {"top_n": cap + 5}}).status_code
    run.check(A, "scan: rows above the plan cap refused", c == 403 if cap < 9999 else c in (202, 409, 422), f"HTTP {c}")
    for name, body in (("ticker scan", {"universe": "ticker", "ticker": ticker}),
                       ("watchlist scan", {"universe": "watchlist", "watchlist_id": wid, "score_all": True})):
        r = run.req("POST", "/v1/scans", headers=h, json=body)
        if not run.check(A, f"scan: {name} queued", r.status_code == 202, f"HTTP {r.status_code} {r.text[:120]}"):
            continue
        t0 = time.perf_counter()
        job = _poll_scan(run, h, r.json()["scan_id"])
        res = job.get("result") or {}
        run.check(A, f"scan: {name} completes", job.get("status") == "complete" and isinstance(res.get("setups"), list),
                  f"status={job.get('status')} error={job.get('error')} symbols={res.get('symbols_scanned')} "
                  f"setups={res.get('total')} in {time.perf_counter() - t0:.0f}s")
    listed = run.req("GET", "/v1/scans", headers=h).json()
    run.check(A, "scan: history lists them", isinstance(listed, list) and len(listed) >= 2, f"{len(listed)} jobs")


def devices(run: Run, s: Dict[str, Any], A: str) -> None:
    """Push device register -> list -> refresh token -> remove (a fake Expo token; nothing is sent)."""
    h = s["h"]
    token = f"ExponentPushToken[zzacceptance{run.run_id}{s['role'].lower()}]"
    r = run.req("POST", "/v1/me/devices", headers=h, json={"push_token": token, "platform": "ios", "device_name": "acceptance"})
    if not run.check(A, "device: register", r.status_code == 200 and r.json().get("provider") == "expo"
                     and "push_token" not in r.json(), f"HTTP {r.status_code}"):
        return
    did = r.json()["id"]
    run.created.append((h, "device", did))
    again = run.req("POST", "/v1/me/devices", headers=h, json={"push_token": token, "platform": "ios"}).json()
    run.check(A, "device: re-register is idempotent", again.get("id") == did, f"id {again.get('id')} vs {did}")
    listed = run.req("GET", "/v1/me/devices", headers=h).json()
    run.check(A, "device: listed", any(d["id"] == did for d in listed), f"{len(listed)} devices")
    rt = s["pair"]["refresh_token"]
    r = run.req("POST", "/v1/auth/refresh", json={"refresh_token": rt})
    if r.status_code == 200:   # keep the journey's later session checks on a fresh pair
        s["pair"] = r.json()
        s["h"] = {"Authorization": f"Bearer {s['pair']['access_token']}"}
    run.check(A, "device: token refresh keeps the device", r.status_code == 200
              and any(d["id"] == did for d in run.req("GET", "/v1/me/devices", headers=s["h"]).json()), f"HTTP {r.status_code}")
    c = run.req("DELETE", f"/v1/me/devices/{did}", headers=s["h"]).status_code
    run.check(A, "device: remove", c == 204, f"HTTP {c}")
    if c == 204:
        run.created.remove((h, "device", did))


def plan_limit(run: Run, s: Dict[str, Any]) -> None:
    """Fill the account up to its alert limit, expect 403 after, then delete what we made."""
    h, A = s["h"], f"limits:{s['role']}"
    info = run.req("GET", "/v1/alerts", headers=h).json()
    made = []
    try:
        for i in range(info["limit"] - info["used"]):
            r = run.req("POST", "/v1/alerts", headers=h, json={"type": "move", "ticker": f"ZQ{i:02d}", "threshold": NEVER_FIRES})
            if r.status_code == 201:
                made.append(r.json()["id"])
        r = run.req("POST", "/v1/alerts", headers=h, json={"type": "move", "ticker": "ZQXX", "threshold": NEVER_FIRES})
        used = run.req("GET", "/v1/alerts", headers=h).json()["used"]
        run.check(A, "alert plan limit", r.status_code == 403 and used == info["limit"],
                  f"limit={info['limit']} used={used} next=HTTP {r.status_code}")
    finally:
        for aid in made:
            run.req("DELETE", f"/v1/alerts/{aid}", headers=h)


def isolation(run: Run, a: Dict[str, Any], b: Dict[str, Any]) -> None:
    A = "isolation"
    r = run.req("POST", "/v1/watchlists", headers=a["h"], json={"name": f"zz-acceptance-{run.run_id}-iso"})
    wid = r.json()["id"]
    run.created.append((a["h"], "watchlist", wid))
    run.req("POST", f"/v1/watchlists/{wid}/tickers", headers=a["h"], json={"tickers": ["AAPL"]})
    run.req("PATCH", f"/v1/watchlists/{wid}/tickers/AAPL", headers=a["h"], json={"note": "private"})
    r = run.req("POST", "/v1/alerts", headers=a["h"], json={"type": "move", "ticker": "AAPL", "threshold": NEVER_FIRES})
    aid = r.json()["id"] if r.status_code == 201 else None
    if aid:
        run.created.append((a["h"], "alert", aid))
    attempts = [("GET", f"/v1/watchlists/{wid}", None), ("PATCH", f"/v1/watchlists/{wid}", {"name": "stolen"}),
                ("POST", f"/v1/watchlists/{wid}/tickers", {"tickers": ["TSLA"]}),
                ("PATCH", f"/v1/watchlists/{wid}/tickers/AAPL", {"note": "overwritten"}),
                ("DELETE", f"/v1/watchlists/{wid}/tickers/AAPL", None), ("DELETE", f"/v1/watchlists/{wid}", None)]
    if aid:
        attempts += [("PATCH", f"/v1/alerts/{aid}", {"enabled": False}), ("DELETE", f"/v1/alerts/{aid}", None)]
    codes = [run.req(m, p, headers=b["h"], **({"json": j} if j else {})).status_code for m, p, j in attempts]
    detail = run.req("GET", f"/v1/watchlists/{wid}", headers=a["h"]).json()
    intact = detail["name"].endswith("-iso") and [(i["ticker"], i["note"]) for i in detail["items"]] == [("AAPL", "private")]
    alert_ok = (not aid) or any(x["id"] == aid and x["enabled"] for x in run.req("GET", "/v1/alerts", headers=a["h"]).json()["alerts"])
    leak = any(w["id"] == wid for w in run.req("GET", "/v1/watchlists", headers=b["h"]).json())
    run.check(A, f"{b['role']} can't read or change {a['role']}'s watchlist, note or alert",
              all(c == 404 for c in codes) and intact and alert_ok and not leak, f"codes {codes}, intact={intact and alert_ok}, listed={leak}")


def session_checks(run: Run, s: Dict[str, Any]) -> None:
    A = "sessions"
    rt = s["pair"]["refresh_token"]
    r1 = run.req("POST", "/v1/auth/refresh", json={"refresh_token": rt})
    r2 = run.req("POST", "/v1/auth/refresh", json={"refresh_token": rt})
    run.check(A, "refresh rotation + retry within grace", r1.status_code == 200 and r2.status_code == 200
              and r1.json()["refresh_token"] != rt, f"first HTTP {r1.status_code}, immediate retry HTTP {r2.status_code}")
    new = r1.json()
    access = new["access_token"]
    c = run.req("POST", "/v1/auth/logout", json={"refresh_token": new["refresh_token"]}).status_code
    after = run.req("POST", "/v1/auth/refresh", json={"refresh_token": new["refresh_token"]}).status_code
    run.check(A, "logout revokes the refresh token", c == 204 and after == 401, f"logout HTTP {c}, refresh after HTTP {after}")
    still = run.req("GET", "/v1/me", headers={"Authorization": f"Bearer {access}"}).status_code
    run.check(A, "access token after logout (documented behaviour)", True,
              f"HTTP {still}: access tokens are stateless and stay valid until they expire (15 min)")


def cleanup(run: Run) -> None:
    for h, kind, rid in list(run.created):
        path = {"watchlist": f"/v1/watchlists/{rid}", "alert": f"/v1/alerts/{rid}",
                "device": f"/v1/me/devices/{rid}"}[kind]
        c = run.req("DELETE", path, headers=h).status_code
        run.check("cleanup", f"delete {kind} {rid}", c in (204, 404), f"HTTP {c}")
    run.created.clear()


def summary_timings(run: Run) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in sorted(run.timings.items()):
        ms = [x * 1000 for x in v]
        out[k] = {"n": len(ms), "first_ms": round(ms[0], 1), "median_ms": round(statistics.median(ms), 1),
                  "max_ms": round(max(ms), 1)}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-url", default=os.environ.get("API_BASE_URL"))
    p.add_argument("--origin", default=os.environ.get("API_TEST_ORIGIN"))
    p.add_argument("--allow-writes", action="store_true")
    p.add_argument("--report", default="")
    p.add_argument("--expired-token-env", default="", help="env var holding an expired token (isolated runs)")
    a = p.parse_args()
    if not a.base_url:
        print("Set API_BASE_URL or --base-url")
        return 2
    run = Run(a.base_url)
    try:
        deployment_checks(run, a.origin)
        auth_rejections(run, os.environ.get(a.expired_token_env) if a.expired_token_env else None)
        sessions = {role: login(run, role) for role in ROLES}
        for role, s in sessions.items():
            if s is None:
                run.check(f"journey:{role}", "full journey", None, f"no ACC_{role}_EMAIL / ACC_{role}_PASSWORD")
                continue
            run.check("auth", f"{role} login", True, "HTTP 200, token pair issued")
            run.guard(f"journey:{role}", "journey", lambda s=s: journey(run, s, a.allow_writes))
        if a.allow_writes and sessions.get("FREE"):
            run.guard("limits", "free alert limit", lambda: plan_limit(run, sessions["FREE"]))
        if a.allow_writes and sessions.get("PRO") and sessions.get("PRO2"):
            run.guard("isolation", "two accounts", lambda: isolation(run, sessions["PRO"], sessions["PRO2"]))
        else:
            run.check("isolation", "two-account isolation", None, "needs PRO and PRO2 test accounts and --allow-writes")
        if sessions.get("PRO"):
            run.guard("sessions", "refresh/logout", lambda: session_checks(run, sessions["PRO"]))
    finally:
        cleanup(run)
    counts = {s: sum(1 for r in run.results if r["status"] == s) for s in ("PASS", "FAIL", "BLOCKED")}
    timings = summary_timings(run)
    print(json.dumps({"counts": counts, "timings_ms": timings}, indent=2))
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump({"base_url": a.base_url, "run_id": run.run_id, "counts": counts, "results": run.results,
                       "timings_ms": timings}, fh, indent=2)
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
