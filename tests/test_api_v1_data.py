"""P1-59 steps 5-6: /v1/scans/latest, /v1/stocks/{ticker}, /v1/watchlists, /v1/alerts.

Watchlists and alerts run against in-memory fakes of db.watchlists / db.alerts
(same function signatures); scans use saved-run fixtures. No Postgres, no network.
"""
import datetime as dt
import importlib.util
import itertools
import json
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
UTC = dt.timezone.utc


class FakeWatchlists:
    """Enough of db.watchlists for the API, keyed by user like the real tables."""

    def __init__(self):
        self.ids = itertools.count(1)
        self.lists = {}  # id -> {user, name, is_default, items: {ticker: note}}

    def list_watchlists(self, user):
        return [{"id": i, "name": w["name"], "is_default": w["is_default"], "created_at": None,
                 "symbol_count": len(w["items"])} for i, w in self.lists.items() if w["user"] == user]

    def create_watchlist(self, user, name, *, make_default=False):
        name = " ".join(str(name).split())[:80]
        if not name:
            raise ValueError("Watchlist name is required.")
        if any(w["user"] == user and w["name"].lower() == name.lower() for w in self.lists.values()):
            raise ValueError("A watchlist with that name already exists.")
        first = not self.list_watchlists(user)
        i = next(self.ids)
        self.lists[i] = {"user": user, "name": name, "is_default": False, "items": {}}
        if make_default or first:
            self.set_default_watchlist(i, user)
        return i

    def _own(self, i, user):
        w = self.lists.get(int(i))
        return w if w and w["user"] == user else None

    def rename_watchlist(self, i, user, name):
        if any(w["user"] == user and w["name"].lower() == name.lower() and k != i for k, w in self.lists.items()):
            raise ValueError("A watchlist with that name already exists.")
        w = self._own(i, user)
        if w:
            w["name"] = name
        return bool(w)

    def set_default_watchlist(self, i, user):
        for k, w in self.lists.items():
            if w["user"] == user:
                w["is_default"] = k == int(i)
        return True

    def delete_watchlist(self, i, user):
        if self._own(i, user):
            del self.lists[int(i)]

    def get_watchlist_items(self, i, user):
        w = self._own(i, user)
        return [{"ticker": t, "date_added": dt.datetime(2026, 10, 1, tzinfo=UTC), "price_when_added": None,
                 "note": n} for t, n in sorted((w or {"items": {}})["items"].items())]

    def get_watchlist_tickers(self, i, user):
        return [x["ticker"] for x in self.get_watchlist_items(i, user)]

    def add_tickers_to_watchlist(self, user, tickers, watchlist_id=None, **k):
        w = self._own(watchlist_id, user)
        added = [t for t in tickers if t not in w["items"]]
        for t in added:
            w["items"][t] = None
        return {"added": added, "already_present": [t for t in tickers if t not in added], "invalid": []}

    def remove_from_watchlist(self, user, ticker, watchlist_id=None):
        w = self._own(watchlist_id, user)
        if ticker not in w["items"]:
            return False
        del w["items"][ticker]
        return True

    def update_watchlist_item_note(self, i, user, ticker, note):
        w = self._own(i, user)
        if not w or ticker not in w["items"]:
            return False
        w["items"][ticker] = note or None
        return True


class FakeAlerts:
    def __init__(self):
        self.ids = itertools.count(1)
        self.rows = []

    def list_alerts(self, user):
        return [dict(r) for r in reversed(self.rows) if r["user_id"] == user]

    def create_alert(self, user, alert_type, *, ticker=None, threshold=None, direction=None, watchlist_only=False):
        key = (user, alert_type, ticker, threshold, direction, watchlist_only)
        if any((r["user_id"], r["alert_type"], r["ticker"], r["threshold"], r["direction"], r["watchlist_only"]) == key
               for r in self.rows):
            raise ValueError("You already have this alert.")
        self.rows.append({"id": next(self.ids), "user_id": user, "alert_type": alert_type, "ticker": ticker,
                          "threshold": threshold, "direction": direction, "watchlist_only": watchlist_only,
                          "enabled": True, "last_fired_at": None,
                          "created_at": dt.datetime(2026, 10, 5, 13, 0, tzinfo=UTC)})

    def set_alert_enabled(self, i, user, enabled):
        for r in self.rows:
            if r["id"] == i and r["user_id"] == user:
                r["enabled"] = enabled

    def delete_alert(self, i, user):
        self.rows = [r for r in self.rows if not (r["id"] == i and r["user_id"] == user)]

    def list_recent_events(self, user, limit=20):
        return [{"id": 7, "alert_id": 1, "ticker": "NVDA", "message": "NVDA moved +5.2%",
                 "fired_at": dt.datetime(2026, 10, 5, 14, 0, tzinfo=UTC)}][:limit]


class DataApiBase(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.wl, self.al = FakeWatchlists(), FakeAlerts()
        for name in ("list_watchlists", "create_watchlist", "rename_watchlist", "set_default_watchlist",
                     "delete_watchlist", "get_watchlist_items", "get_watchlist_tickers",
                     "add_tickers_to_watchlist", "remove_from_watchlist", "update_watchlist_item_note"):
            mock.patch(f"db.watchlists.{name}", side_effect=getattr(self.wl, name)).start()
        for name in ("list_alerts", "create_alert", "set_alert_enabled", "delete_alert", "list_recent_events"):
            mock.patch(f"db.alerts.{name}", side_effect=getattr(self.al, name)).start()
        self.h = self.auth(self.login().json()["access_token"])
        self.accounts["free@example.com"] = {**self.accounts["pro@example.com"], "username": "free@example.com",
                                             "tier": "basic"}


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class WatchlistAndAlertApiTests(DataApiBase):
    def test_requires_sign_in(self):
        for method, path in (("get", "/v1/watchlists"), ("get", "/v1/alerts"), ("get", "/v1/scans/latest"),
                             ("get", "/v1/stocks/AAPL"), ("post", "/v1/alerts")):
            self.assertEqual(getattr(self.client, method)(path).status_code, 401, path)

    def test_watchlist_lifecycle(self):
        r = self.client.post("/v1/watchlists", json={"name": "Momentum"}, headers=self.h)
        self.assertEqual(r.status_code, 201)
        wid = r.json()["id"]
        self.assertTrue(r.json()["is_default"])  # first list is the default
        r = self.client.post(f"/v1/watchlists/{wid}/tickers", json={"tickers": ["nvda", "AMD", "bad ticker!"]},
                             headers=self.h)
        self.assertEqual(r.json(), {"added": ["NVDA", "AMD"], "already_present": [], "invalid": ["BAD TICKER!"]})
        self.assertEqual(self.client.patch(f"/v1/watchlists/{wid}/tickers/NVDA", json={"note": "earnings"},
                                           headers=self.h).status_code, 204)
        detail = self.client.get(f"/v1/watchlists/{wid}", headers=self.h).json()
        self.assertEqual([i["ticker"] for i in detail["items"]], ["AMD", "NVDA"])
        self.assertEqual(detail["items"][1]["note"], "earnings")
        self.assertEqual(detail["items"][0]["added_at"], "2026-10-01T00:00:00+00:00")
        self.assertEqual(self.client.delete(f"/v1/watchlists/{wid}/tickers/AMD", headers=self.h).status_code, 204)
        self.assertEqual(self.client.delete(f"/v1/watchlists/{wid}/tickers/AMD", headers=self.h).status_code, 404)
        r = self.client.patch(f"/v1/watchlists/{wid}", json={"name": "Swing"}, headers=self.h)
        self.assertEqual(r.json()["name"], "Swing")
        self.assertEqual(self.client.delete(f"/v1/watchlists/{wid}", headers=self.h).status_code, 204)
        self.assertEqual(self.client.get("/v1/watchlists", headers=self.h).json(), [])

    def test_duplicate_name_and_limit(self):
        self.client.post("/v1/watchlists", json={"name": "A"}, headers=self.h)
        self.assertEqual(self.client.post("/v1/watchlists", json={"name": "a"}, headers=self.h).status_code, 409)
        with mock.patch("api.user_data.MAX_WATCHLISTS", 1):
            r = self.client.post("/v1/watchlists", json={"name": "B"}, headers=self.h)
        self.assertEqual(r.status_code, 403)

    def test_cannot_touch_someone_elses_watchlist(self):
        other = self.wl.create_watchlist("boss@example.com", "Private")
        self.wl.add_tickers_to_watchlist("boss@example.com", ["TSLA"], other)
        for method, path, body in (("get", f"/v1/watchlists/{other}", None),
                                   ("patch", f"/v1/watchlists/{other}", {"name": "mine"}),
                                   ("delete", f"/v1/watchlists/{other}", None),
                                   ("post", f"/v1/watchlists/{other}/tickers", {"tickers": ["AAPL"]}),
                                   ("delete", f"/v1/watchlists/{other}/tickers/TSLA", None)):
            kw = {"json": body} if body else {}
            self.assertEqual(getattr(self.client, method)(path, headers=self.h, **kw).status_code, 404, path)
        self.assertEqual(self.wl.lists[other]["name"], "Private")
        self.assertIn("TSLA", self.wl.lists[other]["items"])

    def test_alerts_follow_plan_limits_and_rules(self):
        r = self.client.get("/v1/alerts", headers=self.h).json()
        self.assertEqual((r["limit"], r["used"], r["email_enabled"]), (5, 0, True))
        r = self.client.post("/v1/alerts", json={"type": "price", "ticker": "nvda", "threshold": 150,
                                                 "direction": "above"}, headers=self.h)
        self.assertEqual(r.status_code, 201)
        self.assertEqual(r.json()["ticker"], "NVDA")
        self.assertEqual(r.json()["created_at"], "2026-10-05T13:00:00+00:00")
        dup = self.client.post("/v1/alerts", json={"type": "price", "ticker": "NVDA", "threshold": 150,
                                                   "direction": "above"}, headers=self.h)
        self.assertEqual(dup.status_code, 409)
        bad = [{"type": "nope"}, {"type": "price", "ticker": "NVDA", "threshold": 0, "direction": "above"},
               {"type": "price", "ticker": "NVDA", "threshold": 10, "direction": "sideways"},
               {"type": "move", "ticker": "NVDA", "threshold": 0.1}, {"type": "rvol", "threshold": 2},
               {"type": "ema_cross", "ticker": "AMD", "direction": "up"}]
        for body in bad:
            self.assertEqual(self.client.post("/v1/alerts", json=body, headers=self.h).status_code, 422, body)

    def test_free_plan_gets_one_alert(self):
        h = self.auth(self.login("free@example.com").json()["access_token"])
        self.assertEqual(self.client.post("/v1/alerts", json={"type": "watchlist"}, headers=h).status_code, 201)
        r = self.client.post("/v1/alerts", json={"type": "breakout", "threshold": 80}, headers=h)
        self.assertEqual(r.status_code, 403)
        self.assertIn("maximum of 1 alert on", r.json()["detail"])
        self.assertFalse(self.client.get("/v1/alerts", headers=h).json()["email_enabled"])

    def test_toggle_delete_events_and_ownership(self):
        self.client.post("/v1/alerts", json={"type": "rvol", "ticker": "TSLA", "threshold": 2}, headers=self.h)
        aid = self.client.get("/v1/alerts", headers=self.h).json()["alerts"][0]["id"]
        r = self.client.patch(f"/v1/alerts/{aid}", json={"enabled": False}, headers=self.h)
        self.assertFalse(r.json()["enabled"])
        boss = self.auth(self.login("boss@example.com").json()["access_token"])
        self.assertEqual(self.client.delete(f"/v1/alerts/{aid}", headers=boss).status_code, 404)
        self.assertEqual(self.client.delete(f"/v1/alerts/{aid}", headers=self.h).status_code, 204)
        ev = self.client.get("/v1/alerts/events?limit=5", headers=self.h).json()
        self.assertEqual(ev[0]["fired_at"], "2026-10-05T14:00:00+00:00")

    def test_database_down_is_503(self):
        with mock.patch("db.watchlists.list_watchlists",
                        side_effect=RuntimeError("Neon is not available (missing URL or connection failed).")):
            r = self.client.get("/v1/watchlists", headers=self.h)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers["Retry-After"], "30")


@unittest.skipUnless(DEPS and HAS_PANDAS, "needs fastapi, httpx, PyJWT, bcrypt and pandas")
class ScanAndStockApiTests(DataApiBase):
    def setUp(self):
        super().setUp()
        from api import today
        from tests.test_api_today_and_store import MARKET_RUNS, RESULTS

        today.clear_cache()
        self.addCleanup(today.clear_cache)
        runs = [{**r, "created_at": dt.datetime.fromisoformat(r["created_at"])} for r in MARKET_RUNS]
        prob_rows = json.loads(RESULTS[11]) + [{"Ticker": "MXL", "Signal": "PreBreakout", "PreBreakoutProb%": 77.0}]
        for p in (mock.patch("db.runs.list_runs", side_effect=lambda username=None, **k: runs if username == "cron" else []),
                  mock.patch("db.runs.load_run_results",
                             side_effect=lambda rid: json.dumps(prob_rows, allow_nan=True) if rid == 11 else RESULTS.get(rid)),
                  mock.patch("db.signal_outcomes.fetch_ticker_opportunity_history", return_value=[
                      {"time": dt.datetime(2026, 9, 25, 14, tzinfo=UTC), "score": 60, "status": "WATCH",
                       "score_version": None, "signals": ["breakout", "gapper"]}]),
                  mock.patch("api.scans._calibration_records", return_value=[]),
                  mock.patch("db.opportunity_outcomes.get_similar_state_outcomes", return_value={"available": False}),
                  mock.patch("db.earnings.load_earnings_map", return_value={}),
                  mock.patch("db.prices.get_price_data_snapshot", side_effect=self._bars)):
            p.start()

    def _bars(self, symbols, max_age_minutes=15):
        import pandas as pd

        idx = pd.to_datetime(["2026-10-01", "2026-10-02"])
        df = pd.DataFrame({"Open": [10, 11], "High": [11, 12], "Low": [9, 10.5], "Close": [10.5, float("nan")],
                           "Volume": [1e6, 2e6]}, index=idx)
        return ({s: df for s in symbols if s == "MXL"}, set())

    def test_latest_scan_is_capped_by_plan_and_redacted(self):
        r = self.client.get("/v1/scans/latest", headers=self.h).json()
        self.assertEqual(r["scan_at"], "2026-09-28T19:35:00+00:00")
        self.assertEqual(r["max_results"], 100)
        self.assertEqual(r["setups"][0]["ticker"], "MXL")
        self.assertTrue(all(s["prob"] is None and "prebreakout" not in s["signals"] for s in r["setups"]))
        self.assertEqual(self.client.get("/v1/scans/latest?signal=prebreakout", headers=self.h).status_code, 403)
        page = self.client.get("/v1/scans/latest?limit=2&offset=1", headers=self.h).json()
        self.assertEqual([s["ticker"] for s in page["setups"]], [s["ticker"] for s in r["setups"][1:3]])
        with mock.patch("api.scans.max_results_for", return_value=2):
            capped = self.client.get("/v1/scans/latest", headers=self.h).json()
        self.assertEqual((len(capped["setups"]), capped["limited"]), (2, True))

    def test_premium_sees_model_output(self):
        self.accounts["prem@example.com"] = {**self.accounts["pro@example.com"], "username": "prem@example.com",
                                             "tier": "premium"}
        h = self.auth(self.login("prem@example.com").json()["access_token"])
        top = self.client.get("/v1/scans/latest?signal=prebreakout", headers=h).json()["setups"]
        self.assertEqual(top[0]["ticker"], "MXL")
        self.assertEqual(top[0]["prob"], 77.0)

    def test_stock_detail(self):
        self.client.post("/v1/alerts", json={"type": "move", "ticker": "MXL", "threshold": 5}, headers=self.h)
        wid = self.client.post("/v1/watchlists", json={"name": "Chips"}, headers=self.h).json()["id"]
        self.client.post(f"/v1/watchlists/{wid}/tickers", json={"tickers": ["MXL"]}, headers=self.h)
        r = self.client.get("/v1/stocks/mxl", headers=self.h)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual((d["ticker"], d["in_latest_scan"], d["has_setup"]), ("MXL", True, True))
        self.assertIsNotNone(d["hsf_score"])
        self.assertIsNone(d["prob"])  # Pro: Premium model output redacted
        self.assertNotIn("prebreakout", d["signals"])
        self.assertEqual(d["lifecycle"][0]["score"], 60)
        self.assertEqual([b["date"] for b in d["bars"]], ["2026-10-01"])  # NaN close dropped
        self.assertEqual(d["watchlists"], [{"id": wid, "name": "Chips"}])
        self.assertEqual(d["alerts"][0]["type"], "move")
        json.dumps(d, allow_nan=False)

    def test_unknown_and_invalid_tickers(self):
        d = self.client.get("/v1/stocks/ZZZZ", headers=self.h).json()
        self.assertFalse(d["in_latest_scan"])
        self.assertEqual(d["bars"], [])
        self.assertEqual(self.client.get("/v1/stocks/not%20a%20ticker", headers=self.h).status_code, 422)


if __name__ == "__main__":
    unittest.main()
