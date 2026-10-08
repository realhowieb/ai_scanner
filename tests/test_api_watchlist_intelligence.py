"""Watchlist Intelligence, watchlist changes, alert rules and the alert-event feed
through the API (api/main.py), end to end.

Watchlists and ticker alerts use the in-memory fakes from test_api_v1_data; alert
rules use the real db.alert_rules SQL on an in-memory SQLite database; scans are two
saved-run fixtures. No Postgres, no market-data provider, no network.
"""
import datetime as dt
import importlib.util
import json
import time
import unittest
from unittest import mock

from db.alert_rules import list_events as _real_rule_events
from tests.test_api_v1 import DEPS
from tests.test_api_v1_data import DataApiBase

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
UTC = dt.timezone.utc
RUNS = [{"id": 31, "username": "cron", "label": "US_MARKET", "created_at": dt.datetime(2026, 10, 8, 14, 35, tzinfo=UTC)},
        {"id": 30, "username": "cron", "label": "US_MARKET", "created_at": dt.datetime(2026, 10, 8, 13, 35, tzinfo=UTC)}]


def _rows(tickers, *, extra=None):
    rows = []
    for i, t in enumerate(tickers):
        rows.append({"Ticker": t, "Signal": "Breakout", "BreakoutScore": 95 - i * 4, "Last": 10.0 + i,
                     "PctChange": 2.5 + i, "VolRel20": 1.1})
        rows.append({"Ticker": t, "Signal": "Gapper", "GapPct": 4.0, "Last": 10.0 + i})
    return rows + (extra or [])


PREV = _rows(["STM", "MXL", "GRAL", "CIEN"])
CUR = _rows(["MXL", "STM", "GRAL", "NEWT"], extra=[
    {"Ticker": "MXL", "Signal": "PreBreakout", "PreBreakoutProb%": 77.0, "VolRel20": 3.1, "EMACross": "Golden"},
    {"Ticker": "QUIET", "Signal": "Gapper", "GapPct": 0.2, "Last": 5.5, "PctChange": 0.1, "VolRel20": 0.8,
     "EMACross": "Death"},
])
for _r in CUR:  # MXL's unusual volume today (the scan's first non-null VolRel20 is the canonical one)
    if _r["Ticker"] == "MXL" and "VolRel20" in _r:
        _r["VolRel20"] = 3.1
RESULTS = {30: json.dumps(PREV), 31: json.dumps(CUR)}


class SqliteRules:
    def __init__(self, accounts):
        import sqlite3

        from db import alert_rules

        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE users (username TEXT, tier TEXT, is_admin BOOLEAN, email_verified BOOLEAN, "
                          "is_active BOOLEAN)")
        for a in accounts.values():
            self.conn.execute("INSERT INTO users VALUES (?, ?, ?, 1, ?)",
                              (a["username"], a["tier"], a.get("is_admin", False), a.get("is_active", True)))
        self.conn.commit()
        alert_rules.set_connection_factory(lambda: self.conn)

    def close(self):
        from db import alert_rules

        alert_rules.set_connection_factory(None)
        self.conn.close()


@unittest.skipUnless(DEPS and HAS_PANDAS, "needs fastapi, httpx, PyJWT, bcrypt and pandas")
class IntelBase(DataApiBase):
    def setUp(self):
        super().setUp()
        from api import alert_rules as ar
        from api import today

        today.clear_cache()
        self.addCleanup(today.clear_cache)
        for email, tier in (("prem@example.com", "premium"), ("other@example.com", "pro")):
            self.accounts[email] = {**self.accounts["pro@example.com"], "username": email, "tier": tier}
        self.loads = []

        def load(rid):
            self.loads.append(rid)
            return RESULTS.get(rid)

        self.stale = False
        for p in (mock.patch("db.runs.list_runs", side_effect=lambda username=None, **k: RUNS if username == "cron" else []),
                  mock.patch("db.runs.load_run_results", side_effect=load),
                  mock.patch("api.watchlist_intel.observation_stale", side_effect=lambda *a, **k: self.stale),
                  mock.patch("db.alerts.list_events_filtered", side_effect=self._legacy_events),
                  # Watchlist Intelligence must never call a market-data provider
                  mock.patch("market_data.build_day_trader_metrics", side_effect=AssertionError("provider call")),
                  mock.patch("billing_service.realtime_alerts._latest_snapshots",
                             side_effect=AssertionError("provider call")),
                  mock.patch("api.alert_rules._send_email", return_value=(True, None)),
                  mock.patch("db.alert_rules.list_events", side_effect=_real_rule_events)):
            p.start()
        self.rules_db = SqliteRules(self.accounts)
        self.addCleanup(self.rules_db.close)
        self.ar = ar
        self.prem = self.auth(self.login("prem@example.com").json()["access_token"])
        self.other = self.auth(self.login("other@example.com").json()["access_token"])
        self.free = self.auth(self.login("free@example.com").json()["access_token"])
        self.rules_db.conn.execute("INSERT INTO users VALUES ('free@example.com', 'basic', 0, 1, 1)")
        self.rules_db.conn.commit()

    def _legacy_events(self, user, limit=20, *, ticker=None, after=None, before=None):
        rows = [{"id": 7, "alert_id": 1, "ticker": "NVDA", "message": "NVDA moved +5.2%",
                 "fired_at": dt.datetime(2026, 10, 8, 14, 0, tzinfo=UTC)}] if user == "pro@example.com" else []
        rows = [r for r in rows if (not ticker or r["ticker"] == ticker) and (before is None or r["fired_at"] <= before)
                and (after is None or r["fired_at"] >= after)]
        return rows[:limit]

    def watchlist(self, h, tickers, name="Mine"):
        wid = self.client.post("/v1/watchlists", json={"name": name}, headers=h).json()["id"]
        r = self.client.post(f"/v1/watchlists/{wid}/symbols", json={"tickers": tickers}, headers=h)
        self.assertEqual(r.status_code, 200)
        return wid

    def intel(self, h, wid):
        r = self.client.get(f"/v1/watchlists/{wid}/intelligence", headers=h)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


class WatchlistIntelligenceTests(IntelBase):
    def test_every_symbol_gets_canonical_intelligence(self):
        wid = self.watchlist(self.h, ["MXL", "STM", "NEWT", "QUIET", "ZZZZ"])
        d = self.intel(self.h, wid)
        self.assertEqual((d["watchlist_id"], d["name"]), (wid, "Mine"))
        self.assertEqual(d["last_scan_at"], "2026-10-08T14:35:00+00:00")
        self.assertEqual(d["previous_scan_at"], "2026-10-08T13:35:00+00:00")
        self.assertEqual(d["market_data_as_of"], d["last_scan_at"])
        self.assertIn(d["market_session"], ("premarket", "open", "afterhours", "closed"))
        self.assertEqual(d["coverage"], {"symbols": 5, "enriched": 4, "missing": 1})
        self.assertEqual(set(d["unavailable_fields"]), {"company_name", "price_change", "rsi"})
        items = {i["ticker"]: i for i in d["items"]}
        latest = {s["ticker"]: s for s in self.client.get("/v1/scans/latest", headers=self.h).json()["setups"]}
        mxl, stm = items["MXL"], items["STM"]
        self.assertEqual(mxl["hsf_score"], latest["MXL"]["score"])          # the Scanner's own score
        self.assertEqual(mxl["rank"], 1)
        self.assertEqual(mxl["previous_rank"], 3)
        self.assertEqual(mxl["rank_change"], 2)                              # moved up two places
        self.assertEqual((stm["previous_rank"], stm["rank"], stm["rank_change"]), (4, 4, 0))
        self.assertEqual(mxl["score_change"], mxl["hsf_score"] - mxl["previous_hsf_score"])
        self.assertEqual((mxl["price"], mxl["rvol"], mxl["ema_cross"]), (10.0, 3.1, "golden"))
        self.assertEqual(mxl["freshness"], "fresh")
        self.assertIsNone(mxl["prebreakout"])                                 # Pro: locked
        self.assertIsNone(mxl["prebreakout_score"])
        self.assertNotIn("prebreakout", mxl["signals"])
        self.assertTrue(d["prebreakout_locked"])
        new = items["NEWT"]
        self.assertIsNone(new["previous_rank"])
        self.assertIsNone(new["rank_change"])
        quiet = items["QUIET"]                                                # in the scan, not a setup
        self.assertEqual((quiet["in_latest_scan"], quiet["ranked"], quiet["hsf_score"], quiet["price"],
                          quiet["ema_cross"], quiet["freshness"]), (True, False, None, 5.5, "death", "fresh"))
        self.assertEqual(items["ZZZZ"]["freshness"], "missing")
        for i in d["items"]:
            self.assertIsNone(i["company_name"])
            self.assertIsNone(i["rsi"])
            self.assertEqual(i["active_alert_count"], 0)

    def test_premium_sees_prebreakout(self):
        wid = self.watchlist(self.prem, ["MXL", "STM"])
        items = {i["ticker"]: i for i in self.intel(self.prem, wid)["items"]}
        self.assertEqual((items["MXL"]["prebreakout"], items["MXL"]["prebreakout_score"]), (True, 77.0))
        self.assertIs(items["STM"]["prebreakout"], False)

    def test_stale_and_unavailable_scans(self):
        wid = self.watchlist(self.h, ["MXL"])
        self.stale = True
        d = self.intel(self.h, wid)
        self.assertEqual((d["stale"], d["items"][0]["freshness"]), (True, "stale"))
        with mock.patch("api.watchlist_intel.market_runs", side_effect=RuntimeError("down")):
            d = self.intel(self.h, wid)
        self.assertFalse(d["scan_available"])
        self.assertEqual(d["items"][0]["freshness"], "unavailable")
        self.assertEqual(d["coverage"]["missing"], 1)

    def test_empty_watchlist(self):
        wid = self.client.post("/v1/watchlists", json={"name": "Empty"}, headers=self.h).json()["id"]
        d = self.intel(self.h, wid)
        self.assertEqual((d["items"], d["coverage"]["symbols"]), ([], 0))

    def test_batch_no_n_plus_one(self):
        """Run results load once per run however many symbols; repeat requests load nothing."""
        small = self.watchlist(self.h, ["MXL"], name="Small")
        many = [f"T{i}" for i in range(150)] + ["MXL", "STM", "GRAL"]
        big = self.watchlist(self.h, many, name="Big")
        self.intel(self.h, small)
        first = len(self.loads)
        self.assertLessEqual(first, 2)                       # latest + previous run, nothing per symbol
        t0 = time.perf_counter()
        d = self.intel(self.h, big)
        ms = (time.perf_counter() - t0) * 1000
        self.assertEqual(len(self.loads), first)             # cached: no further loads
        self.assertEqual(d["coverage"]["symbols"], 153)
        self.assertLess(ms, 2000)

    def test_symbols_routes_and_ownership(self):
        wid = self.watchlist(self.h, ["MXL", "mxl", "BAD!"])
        r = self.client.post(f"/v1/watchlists/{wid}/symbols", json={"tickers": ["MXL", "STM", "!!"]}, headers=self.h)
        self.assertEqual(r.json(), {"added": ["STM"], "already_present": ["MXL"], "invalid": ["!!"]})
        self.assertEqual(self.client.delete(f"/v1/watchlists/{wid}/symbols/STM", headers=self.h).status_code, 204)
        self.assertEqual(self.client.delete(f"/v1/watchlists/{wid}/symbols/STM", headers=self.h).status_code, 404)
        for path in (f"/v1/watchlists/{wid}/intelligence", f"/v1/watchlists/{wid}/changes"):
            self.assertEqual(self.client.get(path, headers=self.other).status_code, 404, path)
            self.assertEqual(self.client.get(path).status_code, 401, path)
        self.assertEqual(self.client.post(f"/v1/watchlists/{wid}/symbols", json={"tickers": ["A"]},
                                          headers=self.other).status_code, 404)
        self.assertEqual(self.client.delete(f"/v1/watchlists/{wid}/symbols/MXL", headers=self.other).status_code, 404)
        self.assertEqual(self.client.get("/v1/watchlists/999/intelligence", headers=self.h).status_code, 404)

    def test_changes_between_the_two_latest_scans(self):
        wid = self.watchlist(self.prem, ["MXL", "STM", "NEWT", "CIEN", "QUIET"])
        d = self.client.get(f"/v1/watchlists/{wid}/changes", headers=self.prem).json()
        self.assertTrue(d["has_baseline"])
        kinds = {(c["ticker"], c["event_type"]) for c in d["changes"]}
        self.assertIn(("NEWT", "NEW_OPPORTUNITY"), kinds)
        self.assertIn(("CIEN", "DROPPED"), kinds)
        self.assertIn(("MXL", "SIGNAL_ADDED"), kinds)
        mxl = next(c for c in d["changes"] if c["ticker"] == "MXL")
        self.assertEqual((mxl["previous_rank"], mxl["rank"], mxl["rank_change"]), (3, 1, 2))
        self.assertTrue(all(c["ticker"] in {"MXL", "STM", "NEWT", "CIEN", "QUIET"} for c in d["changes"]))
        # below Premium the PreBreakout signal is not revealed through changes either
        wid2 = self.watchlist(self.h, ["MXL"])
        d2 = self.client.get(f"/v1/watchlists/{wid2}/changes", headers=self.h).json()
        self.assertFalse(any(c.get("signal") == "prebreakout" for c in d2["changes"]))


class AlertRuleApiTests(IntelBase):
    def create(self, h, **body):
        return self.client.post("/v1/alerts/rules", json=body, headers=h)

    def test_rule_lifecycle(self):
        r = self.create(self.h, rule_type="HSF_SCORE_CROSS_ABOVE", ticker="mxl", threshold=80)
        self.assertEqual(r.status_code, 201, r.text)
        rule = r.json()
        self.assertEqual((rule["ticker"], rule["operator"], rule["threshold"], rule["enabled"],
                          rule["delivery_channels"], rule["cooldown_seconds"]),
                         ("MXL", "crosses_above", 80.0, True, ["in_app"], 3600))
        self.assertNotIn("user_id", rule)
        lst = self.client.get("/v1/alerts/rules", headers=self.h).json()
        self.assertEqual((lst["limit"], lst["used"], len(lst["rules"])), (5, 1, 1))
        self.assertEqual(lst["capabilities"]["delivery_channels"], ["in_app", "email"])
        self.assertNotIn("PREBREAKOUT_ACTIVE", lst["capabilities"]["alert_rule_types"])
        rid = rule["id"]
        self.assertEqual(self.client.get(f"/v1/alerts/rules/{rid}", headers=self.h).json()["id"], rid)
        p = self.client.patch(f"/v1/alerts/rules/{rid}", json={"threshold": 85, "delivery_channels": ["in_app", "email"],
                                                               "cooldown_seconds": 0}, headers=self.h)
        self.assertEqual(p.status_code, 200, p.text)
        self.assertEqual((p.json()["threshold"], p.json()["delivery_channels"], p.json()["cooldown_seconds"]),
                         (85.0, ["in_app", "email"], 0))
        off = self.client.patch(f"/v1/alerts/rules/{rid}", json={"enabled": False}, headers=self.h).json()
        self.assertFalse(off["enabled"])
        self.assertEqual(self.client.get("/v1/alerts/rules", headers=self.h).json()["used"], 0)
        self.assertEqual(self.client.patch(f"/v1/alerts/rules/{rid}", json={"threshold": 101},
                                           headers=self.h).status_code, 422)
        self.assertEqual(self.client.delete(f"/v1/alerts/rules/{rid}", headers=self.h).status_code, 204)
        self.assertEqual(self.client.get(f"/v1/alerts/rules/{rid}", headers=self.h).status_code, 404)

    def test_cross_user_access_is_404_and_signed_out_is_401(self):
        rid = self.create(self.h, rule_type="RVOL_ABOVE", ticker="MXL", threshold=2).json()["id"]
        for method, kw in (("get", {}), ("patch", {"json": {"enabled": False}}), ("delete", {})):
            r = getattr(self.client, method)(f"/v1/alerts/rules/{rid}", headers=self.other, **kw)
            self.assertEqual(r.status_code, 404, method)
            self.assertEqual(getattr(self.client, method)(f"/v1/alerts/rules/{rid}", **kw).status_code, 401)
        self.assertEqual(self.client.get("/v1/alerts/rules", headers=self.other).json()["rules"], [])
        self.assertTrue(self.client.get(f"/v1/alerts/rules/{rid}", headers=self.h).json()["enabled"])
        theirs = self.watchlist(self.other, ["MXL"])
        r = self.create(self.h, rule_type="RVOL_ABOVE", watchlist_id=theirs, threshold=2)
        self.assertEqual(r.status_code, 404)                          # someone else's watchlist
        for path in ("/v1/alerts/rules", "/v1/alerts/rules/types", "/v1/alerts/events", "/v1/me/capabilities"):
            self.assertEqual(self.client.get(path).status_code, 401, path)

    def test_validation_entitlements_and_limits(self):
        cases = [({"rule_type": "NOPE", "ticker": "MXL"}, 422),
                 ({"rule_type": "HSF_SCORE_ABOVE", "ticker": "MXL"}, 422),                    # no threshold
                 ({"rule_type": "HSF_SCORE_ABOVE", "threshold": 80}, 422),                    # no scope
                 ({"rule_type": "HSF_SCORE_ABOVE", "threshold": 80, "ticker": "MXL", "watchlist_id": 1}, 422),
                 ({"rule_type": "HSF_SCORE_ABOVE", "threshold": 80, "ticker": "M$X"}, 422),
                 ({"rule_type": "HSF_SCORE_ABOVE", "threshold": 80, "ticker": "MXL",
                   "delivery_channels": ["sms"]}, 422),
                 ({"rule_type": "PREBREAKOUT_ACTIVE", "ticker": "MXL"}, 403),               # Premium only
                 ({"rule_type": "HSF_SCORE_ABOVE", "threshold": 80, "ticker": "MXL",
                   "delivery_channels": ["email"]}, 403)]                                    # Pro+ only
        for body, code in cases:
            self.assertEqual(self.create(self.free, **body).status_code, code, body)
        self.assertEqual(self.create(self.prem, rule_type="PREBREAKOUT_ACTIVE", ticker="MXL").status_code, 201)
        types = {t["type"]: t for t in self.client.get("/v1/alerts/rules/types", headers=self.free).json()}
        self.assertFalse(types["PREBREAKOUT_ACTIVE"]["available"])
        self.assertEqual(types["HSF_SCORE_CROSS_ABOVE"]["kind"], "transition")
        # Free: one active alert in total
        self.assertEqual(self.create(self.free, rule_type="RVOL_ABOVE", ticker="MXL", threshold=2).status_code, 201)
        r = self.create(self.free, rule_type="RVOL_ABOVE", ticker="STM", threshold=2)
        self.assertEqual(r.status_code, 403)
        self.assertIn("maximum of 1", r.json()["detail"])
        self.assertEqual(self.create(self.free, rule_type="RVOL_ABOVE", ticker="STM", threshold=2,
                                     enabled=False).status_code, 201)       # saved switched off is fine
        self.assertEqual(self.create(self.free, rule_type="RVOL_ABOVE", ticker="MXL",
                                     threshold=2).status_code, 409)         # duplicate
        caps = self.client.get("/v1/me/capabilities", headers=self.free).json()
        self.assertEqual((caps["max_active_alerts"], caps["delivery_channels"], caps["max_symbols_per_watchlist"]),
                         (1, ["in_app"], None))

    def test_database_down_is_503(self):
        from db import alert_rules

        alert_rules.set_connection_factory(lambda: None)
        with mock.patch("db.alert_rules.get_neon_conn", return_value=None):
            r = self.client.get("/v1/alerts/rules", headers=self.h)
        self.assertEqual(r.status_code, 503)

    def test_deleting_a_watchlist_switches_its_rules_off(self):
        wid = self.watchlist(self.h, ["MXL"])
        rid = self.create(self.h, rule_type="RVOL_ABOVE", watchlist_id=wid, threshold=2).json()["id"]
        self.assertEqual(self.intel(self.h, wid)["items"][0]["active_alert_count"], 1)
        self.assertEqual(self.client.delete(f"/v1/watchlists/{wid}", headers=self.h).status_code, 204)
        self.assertFalse(self.client.get(f"/v1/alerts/rules/{rid}", headers=self.h).json()["enabled"])


class EndToEndTests(IntelBase):
    """The success condition: watchlist -> intelligence -> rule -> deduplicated alert."""

    def evaluate(self):
        from api import watchlist_intel

        with mock.patch("db.alert_rules.watchlist_tickers", side_effect=self._wl_tickers):
            return self.ar.evaluate_once(dt.datetime(2026, 10, 8, 15, 0, tzinfo=UTC),
                                         observation=watchlist_intel.market_observation())

    def _wl_tickers(self, pairs):
        out = {}
        for user, wid in pairs:
            w = self.wl.lists.get(int(wid))
            if w and w["user"] == user:
                out[int(wid)] = sorted(w["items"])
        return out

    def test_watchlist_rule_triggers_once_and_shows_in_events(self):
        wid = self.watchlist(self.h, ["MXL", "STM", "QUIET"])
        self.assertEqual(self.create_rule(rule_type="RVOL_ABOVE", watchlist_id=wid, threshold=3), 201)
        self.assertEqual(self.create_rule(rule_type="HSF_SCORE_ABOVE", ticker="MXL", threshold=1), 201)
        self.assertEqual(self.intel(self.h, wid)["items"][0]["active_alert_count"], 2)
        r = self.evaluate()
        self.assertEqual(r["triggered"], 2)                       # MXL rvol 3.1, MXL score
        self.assertEqual(self.evaluate()["triggered"], 0)         # same scan again: idempotent
        evs = self.client.get("/v1/alerts/events", headers=self.h).json()
        rule_evs = [e for e in evs if e["source"] == "rule"]
        self.assertEqual(len(rule_evs), 2)
        rv = next(e for e in rule_evs if e["rule_type"] == "RVOL_ABOVE")
        self.assertEqual((rv["ticker"], rv["trigger_value"], rv["threshold"], rv["watchlist_id"]),
                         ("MXL", 3.1, 3.0, wid))
        self.assertEqual(rv["market_data_as_of"], "2026-10-08T14:35:00+00:00")
        self.assertEqual(rv["delivery"], {"in_app": "delivered"})
        self.assertIsNotNone(rv["hsf_score"])
        legacy = [e for e in evs if e["source"] == "alert"]
        self.assertEqual([e["event_id"] for e in legacy], ["alert:7"])
        self.assertEqual(len({e["event_id"] for e in evs}), len(evs))
        # filters
        q = lambda **p: self.client.get("/v1/alerts/events", params=p, headers=self.h)  # noqa: E731
        self.assertEqual({e["event_id"] for e in q(rule_id=rv["rule_id"]).json()}, {rv["event_id"]})
        self.assertEqual({e["source"] for e in q(watchlist_id=wid).json()}, {"rule"})
        self.assertEqual({e["ticker"] for e in q(ticker="nvda").json()}, {"NVDA"})
        self.assertEqual({e["source"] for e in q(source="alert").json()}, {"alert"})
        self.assertEqual(q(triggered_after="2030-01-01T00:00:00Z").json(), [])
        self.assertEqual({e["source"] for e in q(triggered_before="2026-10-08T14:30:00Z").json()}, {"alert"})
        # cursor paging walks every event exactly once
        seen, cursor = [], None
        while True:
            page = q(limit=1, **({"cursor": cursor} if cursor else {})).json()
            if not page:
                break
            seen.append(page[0]["event_id"])
            cursor = page[0]["cursor"]
        self.assertEqual(seen, [e["event_id"] for e in evs])
        self.assertEqual(q(cursor="garbage").status_code, 422)
        # nobody else sees them
        self.assertEqual(self.client.get("/v1/alerts/events", headers=self.other).json(), [])
        # what changed lists them
        ch = self.client.get(f"/v1/watchlists/{wid}/changes", headers=self.h).json()
        self.assertEqual(len(ch["alerts"]), 2)

    def create_rule(self, **body):
        return self.client.post("/v1/alerts/rules", json=body, headers=self.h).status_code


@unittest.skipUnless(DEPS, "needs fastapi")
class AccountDeletionCoversRulesTests(unittest.TestCase):
    def test_rule_tables_are_deleted_with_the_account(self):
        import inspect

        from db import account_deletion

        self.assertIn(("hsf_alert_rule_events", "user_id"), account_deletion.ACCOUNT_TABLES)
        src = inspect.getsource(account_deletion.delete_account)
        self.assertIn("DELETE FROM hsf_alert_rules", src)
        self.assertIn("DELETE FROM hsf_alert_rule_state", src)


if __name__ == "__main__":
    unittest.main()
