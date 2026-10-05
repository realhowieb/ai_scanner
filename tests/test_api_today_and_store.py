"""P1-59: /today payload builder (api/today.py) and refresh-token storage (api/store.py).

The builder runs on saved-run fixtures (db.runs mocked). The store tests run
against a throwaway Postgres when HSF_TEST_PG_URL is set.
"""
import datetime as dt
import importlib.util
import json
import math
import os
import unittest
from unittest import mock

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
PG_URL = os.environ.get("HSF_TEST_PG_URL")
UTC = dt.timezone.utc

TUE_840_ET = dt.datetime(2026, 9, 29, 12, 40, tzinfo=UTC)
TUE_NOON_ET = dt.datetime(2026, 9, 29, 16, 0, tzinfo=UTC)
TUE_6PM_ET = dt.datetime(2026, 9, 29, 22, 0, tzinfo=UTC)
SAT = dt.datetime(2026, 10, 3, 16, 0, tzinfo=UTC)

PRO = {"can_day_trader": True, "can_early_breakout": False}
FREE = {"can_day_trader": False, "can_early_breakout": False}

MARKET_RUNS = [  # cron / US_MARKET, newest first
    {"id": 11, "username": "cron", "label": "US_MARKET", "created_at": "2026-09-28T19:35:00+00:00"},
    {"id": 10, "username": "cron", "label": "US_MARKET", "created_at": "2026-09-28T13:35:00+00:00"},
]
SESSION_RUNS = [
    {"id": 21, "label": "premarket", "created_at": "2026-09-29T12:35:00+00:00"},
    {"id": 22, "label": "postmarket", "created_at": "2026-09-28T21:35:00+00:00"},
    {"id": 23, "label": "postmarket", "created_at": "2026-09-29T21:35:00+00:00"},
]


def _scan_rows(prefix=""):
    rows = []
    for i, t in enumerate(["MXL", "STM", "GRAL", "SOXL", "TER", "CIEN"]):
        rows.append({"Ticker": t, "Signal": "Breakout", "BreakoutScore": 95 - i * 4, "Last": 10.0 + i,
                     "PctChange": 2.5, "PMLast": 10.5 + i, "PMPctChange": 5.0 - i,
                     "AHLast": 9.0 + i, "AHPctChange": -3.0 + i * 0.1})
        rows.append({"Ticker": t, "Signal": "Gapper", "GapPct": 4.0, "Last": 10.0 + i})
    rows.append({"Ticker": "NANX", "Signal": "Breakout", "BreakoutScore": float("nan"), "Last": float("nan")})
    return rows


RESULTS = {rid: json.dumps(_scan_rows(), allow_nan=True) for rid in (10, 11, 21, 22, 23)}


@unittest.skipUnless(HAS_PANDAS, "needs pandas")
class TodayBuilderTests(unittest.TestCase):
    def setUp(self):
        from api import today

        today.clear_cache()
        self.addCleanup(today.clear_cache)

        self.now = SAT

        def list_runs(limit=50, include_snapshots=True, username=None, **k):
            runs = MARKET_RUNS if username == "cron" else SESSION_RUNS if username == "scheduler" else []
            # like the real table: nothing saved after "now"
            return [r for r in runs if dt.datetime.fromisoformat(r["created_at"]) <= self.now]

        for p in (mock.patch("db.runs.list_runs", side_effect=list_runs),
                  mock.patch("db.runs.load_run_results", side_effect=lambda rid: RESULTS.get(rid))):
            p.start()
        self.addCleanup(mock.patch.stopall)
        self.today = today

    def build(self, now, ent):
        self.now = now
        self.today.clear_cache()
        out = self.today.build_today(now, ent)
        json.dumps(out, allow_nan=False)  # must be strict JSON (no NaN/inf)
        return out

    def test_pre_open_pro(self):
        out = self.build(TUE_840_ET, PRO)
        self.assertEqual(out["errors"], [])
        self.assertEqual(out["market"]["phase"], "premarket")
        bo = out["before_open"]
        self.assertFalse(bo["locked"])
        self.assertEqual(bo["scan_at"], "2026-09-29T12:35:00+00:00")
        self.assertTrue(bo["movers"])
        self.assertEqual(bo["movers"][0]["ticker"], "MXL")
        self.assertEqual(bo["movers"][0]["pct"], 5.0)
        # last night's after-hours card still shows before the open
        self.assertEqual(out["after_close"]["scan_at"], "2026-09-28T21:35:00+00:00")

    def test_free_sees_locked_cards_without_movers(self):
        out = self.build(TUE_840_ET, FREE)
        self.assertEqual(out["before_open"], {"scan_at": "2026-09-29T12:35:00+00:00", "locked": True, "movers": []})
        self.assertTrue(out["after_close"]["locked"])
        self.assertEqual(out["after_close"]["movers"], [])

    def test_market_hours_hide_session_cards(self):
        out = self.build(TUE_NOON_ET, PRO)
        self.assertEqual(out["market"]["phase"], "open")
        self.assertIsNone(out["before_open"])
        self.assertIsNone(out["after_close"])

    def test_evening_shows_tonights_after_hours(self):
        out = self.build(TUE_6PM_ET, PRO)
        self.assertEqual(out["market"]["phase"], "afterhours")
        self.assertEqual(out["after_close"]["scan_at"], "2026-09-29T21:35:00+00:00")
        self.assertIsNone(out["before_open"])

    def test_top_setups_and_redaction(self):
        out = self.build(SAT, PRO)
        self.assertEqual(out["market"]["phase"], "closed")
        top = out["top_setups"]
        self.assertEqual(top["scan_at"], "2026-09-28T19:35:00+00:00")
        self.assertIn(top["state"], ("qualifying", "no_qualifying"))
        for s in top["setups"]:
            self.assertEqual(set(s), {"ticker", "score", "primary_setup", "status", "n_signals",
                                      "last", "chg_pct", "gap_pct", "rvol", "prob"})
            self.assertIsNone(s["prob"])  # Premium model output redacted below Premium
            for v in s.values():
                self.assertFalse(isinstance(v, float) and not math.isfinite(v))

    def test_recap(self):
        rec = self.build(SAT, PRO)["recap"]
        self.assertEqual(rec["day"], "2026-09-28")
        self.assertEqual(rec["scans"], 2)
        self.assertEqual(rec["postmarket_scans"], 1)

    def test_one_failing_section_does_not_blank_the_rest(self):
        with mock.patch("ui.today.today_top_setups", side_effect=RuntimeError("boom secret")):
            out = self.build(TUE_840_ET, PRO)
        self.assertIsNone(out["top_setups"])
        self.assertEqual(out["errors"], [{"section": "top_setups", "error": "RuntimeError"}])
        self.assertTrue(out["before_open"]["movers"])
        self.assertNotIn("secret", json.dumps(out))

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "needs pydantic")
    def test_payload_matches_the_response_model(self):
        from api.models import Today

        for now in (TUE_840_ET, TUE_NOON_ET, TUE_6PM_ET, SAT):
            Today(**self.build(now, PRO))
            Today(**self.build(now, FREE))

    def test_cache_drops_expired_entries_and_is_capped(self):
        t = self.today
        t.clear_cache()
        with mock.patch.object(t.time, "monotonic", return_value=1000.0):
            for i in range(t.CACHE_MAX_ENTRIES + 10):
                t._cached(("k", i), lambda: i)
        self.assertEqual(t.cache_size(), t.CACHE_MAX_ENTRIES)
        with mock.patch.object(t.time, "monotonic", return_value=1000.0 + t.CACHE_TTL_S + 1):
            t._cached("fresh", lambda: 1)
        self.assertEqual(t.cache_size(), 1)

    def test_no_scans_at_all(self):
        # an empty run list is only believed when the database answers (api.today._runs_or_outage)
        with mock.patch("db.runs.list_runs", return_value=[]), mock.patch("api.store.ping"):
            self.today.clear_cache()
            out = self.build(SAT, PRO)
        self.assertEqual(out["top_setups"]["state"], "empty_scan")
        self.assertIsNone(out["recap"])


@unittest.skipUnless(PG_URL, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class RefreshStorePostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            conn.execute("DROP TABLE IF EXISTS api_refresh_tokens")
        os.environ["DATABASE_URL"] = PG_URL
        from api import store

        self.store = store
        # schema_once already ran in this process; each test dropped the table, so recreate it
        with psycopg.connect(PG_URL) as conn:
            store.ensure_refresh_schema.__wrapped__(conn)
        self.addCleanup(os.environ.pop, "DATABASE_URL", None)

    def _age(self, token_hash, seconds=31):
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            conn.execute("UPDATE api_refresh_tokens SET revoked_at = NOW() - make_interval(secs => %s) "
                         "WHERE token_hash = %s", (seconds, token_hash))

    def test_rotate_reuse_logout_and_expiry(self):
        s = self.store
        s.save_refresh_token("h1", "pro@example.com", 3600, "ios")
        s.save_refresh_token("h2", "pro@example.com", 3600, "web")
        self.assertEqual(s.use_refresh_token("h1"), ("ok", "pro@example.com"))
        self._age("h1")  # replayed after the grace window: theft
        self.assertEqual(s.use_refresh_token("h1"), ("reused", "pro@example.com"))
        self.assertEqual(s.use_refresh_token("h2")[0], "reused")  # reuse revoked every session
        self.assertEqual(s.use_refresh_token("nope"), ("invalid", None))
        s.save_refresh_token("h3", "x@example.com", 0, None)
        self.assertEqual(s.use_refresh_token("h3"), ("invalid", None))  # expired
        s.save_refresh_token("h4", "x@example.com", 3600, None)
        s.revoke_refresh_token("h4")
        self.assertEqual(s.use_refresh_token("h4")[0], "reused")  # logout gets no grace

    def test_password_signout_replay_is_invalid_not_theft(self):
        """A device signed out by a password change retrying its old token must not
        trigger the reuse sweep (which would also kill the new session)."""
        s = self.store
        s.save_refresh_token("p1", "pw@example.com", 3600, "ios")
        s.revoke_all_refresh_tokens("pw@example.com", "password_change")
        s.save_refresh_token("p2", "pw@example.com", 3600, "web")   # the device that changed it
        self.assertEqual(s.use_refresh_token("p1"), ("invalid", None))
        self.assertEqual(s.use_refresh_token("p2"), ("ok", "pw@example.com"))

    def test_grace_window_for_a_just_rotated_token(self):
        s = self.store
        s.save_refresh_token("g1", "g@example.com", 3600, "ios")
        s.save_refresh_token("g2", "g@example.com", 3600, "web")
        self.assertEqual(s.use_refresh_token("g1"), ("ok", "g@example.com"))
        # retry right away (lost response / concurrent refresh): no theft verdict
        self.assertEqual(s.use_refresh_token("g1"), ("grace", "g@example.com"))
        self.assertEqual(s.use_refresh_token("g2"), ("ok", "g@example.com"))  # other session untouched
        # past the window, the same replay is treated as theft
        self._age("g1")
        s.save_refresh_token("g3", "g@example.com", 3600, None)
        self.assertEqual(s.use_refresh_token("g1"), ("reused", "g@example.com"))
        self.assertEqual(s.use_refresh_token("g3")[0], "reused")  # sweep revoked it


if __name__ == "__main__":
    unittest.main()
