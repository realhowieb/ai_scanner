"""POST /v1/scans: custom scans for API clients (Web v2 client-readiness run).

Plan rules are tested adversarially against api.custom_scans.plan_scan (the
server must reject what the web hides behind disabled widgets); routes run
against an in-memory job store; the real job store and a whole scan (engine
included, prices mocked) run against a throwaway Postgres when HSF_TEST_PG_URL
is set.
"""
import importlib.util
import os
import time
import unittest
from unittest import mock

from tests.test_api_v1 import DEPS, ApiTestCase, _hash

PG_URL = os.environ.get("HSF_TEST_PG_URL")
HAS_PANDAS = importlib.util.find_spec("pandas") is not None


def _ent(tier):
    from ui.app_session import compute_entitlements

    return dict(compute_entitlements(tier_obj=tier, is_admin=tier == "admin"))


def _plan(req, tier, session="regular"):
    from api.custom_scans import plan_scan
    from api.scans import max_results_for

    return plan_scan(req, entitlements=_ent(tier), tier=tier, is_admin=tier == "admin",
                     max_results=max_results_for(tier), now_session=session)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "needs the app's modules")
class PlanRuleTests(unittest.TestCase):
    def test_universes_by_plan(self):
        from api.custom_scans import PlanError

        allowed = {"basic": {"sp500", "ticker"}, "pro": {"sp500", "nasdaq", "combo", "ticker"},
                   "premium": {"sp500", "nasdaq", "combo", "us_market", "ticker"},
                   "admin": {"sp500", "nasdaq", "combo", "us_market", "ticker"}}
        for tier, ok in allowed.items():
            for u in ("sp500", "nasdaq", "combo", "us_market", "ticker"):
                req = {"universe": u, "ticker": "AAPL" if u == "ticker" else None}
                with self.subTest(tier=tier, universe=u):
                    if u in ok:
                        self.assertEqual(_plan(req, tier)["universe"], u)
                    else:
                        with self.assertRaises(PlanError):
                            _plan(req, tier)

    def test_rows_capped_by_plan(self):
        from api.custom_scans import PlanError

        for tier, cap in (("basic", 25), ("pro", 100), ("premium", 200)):
            with self.subTest(tier=tier):
                self.assertEqual(_plan({"universe": "sp500", "filters": {"top_n": cap}}, tier)["top_n"], cap)
                with self.assertRaises(PlanError):
                    _plan({"universe": "sp500", "filters": {"top_n": cap + 1}}, tier)
        self.assertEqual(_plan({"universe": "sp500"}, "basic")["top_n"], 25)     # default min(25, cap)

    def test_pro_features_rejected_for_free(self):
        from api.custom_scans import PlanError

        for f in ({"session": "premarket"}, {"session": "afterhours"}, {"unusual_volume": True},
                  {"apply_gap_filter": True}):
            with self.subTest(filters=f), self.assertRaises(PlanError):
                _plan({"universe": "sp500", "filters": f}, "basic")
            self.assertTrue(_plan({"universe": "sp500", "filters": f}, "pro"))

    def test_extended_session_outside_its_hours_scans_regular(self):
        p = _plan({"universe": "sp500", "filters": {"session": "premarket"}}, "pro", session="regular")
        self.assertEqual((p["session_requested"], p["session"]), ("premarket", "regular"))
        p = _plan({"universe": "sp500", "filters": {"session": "premarket"}}, "pro", session="premarket")
        self.assertEqual(p["session"], "premarket")

    def test_pro_cannot_uncap_premium_can(self):
        from api.custom_scans import PlanError

        with self.assertRaises(PlanError):
            _plan({"universe": "nasdaq", "filters": {"max_nasdaq": 99_999}}, "pro")
        with self.assertRaises(PlanError):
            _plan({"universe": "combo", "filters": {"max_combo": 7000}}, "pro")
        pro = _plan({"universe": "combo"}, "pro")
        self.assertEqual((pro["full_lists"], pro["max_nasdaq"], pro["max_combo"]), (False, 1200, 1000))
        prem = _plan({"universe": "combo", "filters": {"max_combo": 100}}, "premium")
        self.assertEqual((prem["full_lists"], prem["max_combo"]), (True, None))   # caps ignored

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            _plan({"universe": "sp500", "filters": {"min_price": 50, "max_price": 10}}, "pro")
        with self.assertRaises(ValueError):
            _plan({"universe": "ticker"}, "pro")
        with self.assertRaises(ValueError):
            _plan({"universe": "watchlist"}, "pro")

    def test_premium_model_output_follows_the_plan(self):
        self.assertFalse(_plan({"universe": "sp500"}, "pro")["early_breakout"])
        self.assertTrue(_plan({"universe": "sp500"}, "premium")["early_breakout"])


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class ScanRouteTests(ApiTestCase):
    """Routes with an in-memory job store; the scan itself is stubbed."""

    def setUp(self):
        super().setUp()
        self.accounts["free@example.com"] = {"username": "free@example.com", "full_name": "Free", "tier": "basic",
                                             "password": _hash("right pw"), "is_admin": False, "is_active": True}
        self.accounts["prem@example.com"] = {"username": "prem@example.com", "full_name": "Prem", "tier": "premium",
                                             "password": _hash("right pw"), "is_admin": False, "is_active": True}
        self.jobs, self.submitted = {}, []
        p = mock.patch
        for target, fn in (
            ("api.scan_jobs.create_job", self._create),
            ("api.scan_jobs.get_job", lambda u, i: self.jobs.get(i) if self.jobs.get(i, {}).get("username") == u else None),
            ("api.scan_jobs.list_jobs", lambda u, n: [j for j in self.jobs.values() if j["username"] == u][:n]),
            ("api.scan_jobs.submit", lambda job_id, work: self.submitted.append((job_id, work))),
            ("api.user_data.get_watchlist", self._watchlist),
        ):
            p(target, side_effect=fn).start()

    def _create(self, user, universe, params):
        from api.scan_jobs import ScanInProgress

        for j in self.jobs.values():
            if j["username"] == user and j["status"] in ("queued", "running"):
                raise ScanInProgress(j["id"])
        jid = f"{len(self.jobs):032x}"
        self.jobs[jid] = {"id": jid, "username": user, "status": "queued", "universe": universe, "params": params,
                          "progress": {"phase": "queued"}, "result": None, "error": None,
                          "created_at": "2026-10-06T12:00:00+00:00", "started_at": None, "finished_at": None}
        return dict(self.jobs[jid])

    def _watchlist(self, user, wid):
        from api.user_data import NotFound

        if user == "pro@example.com" and wid == 7:
            return {"id": 7, "items": [{"ticker": "AAPL"}]}
        raise NotFound("watchlist")

    def h(self, email):
        return self.auth(self.login(email).json()["access_token"])

    def post(self, email, body):
        return self.client.post("/v1/scans", headers=self.h(email), json=body)

    def test_queue_then_poll(self):
        r = self.post("pro@example.com", {"universe": "nasdaq", "filters": {"top_n": 50}})
        self.assertEqual(r.status_code, 202, r.text)
        job = r.json()
        self.assertEqual((job["status"], job["universe"], job["params"]["top_n"]), ("queued", "nasdaq", 50))
        self.assertEqual(len(job["scan_id"]), 32)
        self.assertNotIn("is_admin", job["params"])
        self.assertEqual(len(self.submitted), 1)
        got = self.client.get(f"/v1/scans/{job['scan_id']}", headers=self.h("pro@example.com"))
        self.assertEqual(got.json()["status"], "queued")
        self.assertEqual(len(self.client.get("/v1/scans", headers=self.h("pro@example.com")).json()), 1)

    def test_adversarial_plan_bypass_is_rejected_server_side(self):
        cases = [("free@example.com", {"universe": "us_market"}),
                 ("free@example.com", {"universe": "nasdaq"}),
                 ("free@example.com", {"universe": "combo"}),
                 ("free@example.com", {"universe": "sp500", "filters": {"top_n": 200}}),
                 ("free@example.com", {"universe": "sp500", "filters": {"session": "premarket"}}),
                 ("pro@example.com", {"universe": "us_market"}),
                 ("pro@example.com", {"universe": "combo", "filters": {"max_combo": 50_000}}),
                 ("pro@example.com", {"universe": "sp500", "filters": {"top_n": 101}})]
        for email, body in cases:
            with self.subTest(email=email, body=body):
                r = self.post(email, body)
                self.assertEqual(r.status_code, 403, r.text)
        self.assertEqual(self.submitted, [])

    def test_premium_and_admin_full_market(self):
        r = self.post("prem@example.com", {"universe": "us_market", "filters": {"top_n": 200}})
        self.assertEqual(r.status_code, 202, r.text)
        self.assertTrue(r.json()["params"]["full_lists"])
        r = self.post("boss@example.com", {"universe": "us_market"})          # admin
        self.assertEqual(r.status_code, 202, r.text)

    def test_one_scan_at_a_time(self):
        first = self.post("pro@example.com", {"universe": "sp500"}).json()
        r = self.post("pro@example.com", {"universe": "sp500"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["scan_id"], first["scan_id"])

    def test_other_accounts_scans_and_watchlists_are_invisible(self):
        job = self.post("pro@example.com", {"universe": "sp500"}).json()
        self.assertEqual(self.client.get(f"/v1/scans/{job['scan_id']}", headers=self.h("prem@example.com")).status_code, 404)
        self.assertEqual(self.client.get("/v1/scans", headers=self.h("prem@example.com")).json(), [])
        self.assertEqual(self.post("prem@example.com", {"universe": "watchlist", "watchlist_id": 7}).status_code, 404)

    def test_watchlist_and_ticker_scans(self):
        r = self.post("pro@example.com", {"universe": "watchlist", "watchlist_id": 7, "score_all": True})
        self.assertEqual(r.status_code, 202, r.text)
        self.jobs[r.json()["scan_id"]]["status"] = "complete"
        r = self.post("free@example.com", {"universe": "ticker", "ticker": "brk.b"})
        self.assertEqual(r.status_code, 202, r.text)
        self.assertEqual(r.json()["params"]["ticker"], "BRK.B")

    def test_validation_and_auth(self):
        self.assertEqual(self.client.post("/v1/scans", json={"universe": "sp500"}).status_code, 401)
        self.assertEqual(self.post("pro@example.com", {"universe": "everything"}).status_code, 422)
        self.assertEqual(self.post("pro@example.com", {"universe": "ticker", "ticker": "$$$"}).status_code, 422)
        self.assertEqual(self.post("pro@example.com", {"universe": "ticker"}).status_code, 422)
        self.assertEqual(self.post("pro@example.com", {"universe": "sp500", "filters": {"min_price": 0}}).status_code, 422)
        self.assertEqual(self.client.get("/v1/scans/not-an-id", headers=self.h("pro@example.com")).status_code, 422)

    def test_hourly_limit(self):
        from api import ratelimit

        with mock.patch.dict(ratelimit.LIMITS, {"scan": (2, 3600)}):
            for _ in range(2):
                job = self.post("pro@example.com", {"universe": "sp500"}).json()
                self.jobs[job["scan_id"]]["status"] = "complete"
            r = self.post("pro@example.com", {"universe": "sp500"})
        self.assertEqual(r.status_code, 429)
        self.assertIn("Retry-After", r.headers)

    def test_busy_service(self):
        from api.scan_jobs import ScanBusy

        with mock.patch("api.scan_jobs.create_job", side_effect=ScanBusy("x")):
            r = self.post("pro@example.com", {"universe": "sp500"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers.get("Retry-After"), "60")

    def test_openapi_documents_the_contract(self):
        spec = self.client.get("/openapi.json").json()
        post = spec["paths"]["/v1/scans"]["post"]
        for code in ("202", "403", "409", "429", "503"):
            self.assertIn(code, post["responses"])
        universe = spec["components"]["schemas"]["ScanCreate"]["properties"]["universe"]
        self.assertEqual(set(universe["enum"]), {"sp500", "nasdaq", "combo", "us_market", "watchlist", "ticker"})


def _fake_prices(tickers, **_k):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2026-07-01", periods=60, freq="B")
    out = {}
    for i, t in enumerate(tickers):
        close = np.linspace(20 + i, (20 + i) * 1.3, 60)
        out[t] = pd.DataFrame({"Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98,
                               "Close": close, "Volume": np.full(60, 2_000_000.0)}, index=idx)
    return out, []


@unittest.skipUnless(PG_URL and DEPS and HAS_PANDAS, "set HSF_TEST_PG_URL to a throwaway Postgres to run")
class ScanJobPostgresTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        with psycopg.connect(PG_URL) as conn:
            conn.execute("DROP TABLE IF EXISTS api_scan_jobs")
        os.environ["DATABASE_URL"] = PG_URL
        self.addCleanup(os.environ.pop, "DATABASE_URL", None)
        from api import scan_jobs

        self.sj = scan_jobs
        with psycopg.connect(PG_URL) as conn:
            scan_jobs.ensure_scan_jobs_schema.__wrapped__(conn)

    def _wait(self, job_id, user="pro@example.com", timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.sj.get_job(user, job_id)
            if job["status"] in ("complete", "failed"):
                return job
            time.sleep(0.2)
        self.fail("scan job did not finish")

    def test_store_rules_and_stale_expiry(self):
        import psycopg

        sj = self.sj
        job = sj.create_job("pro@example.com", "sp500", {"universe": "sp500"})
        self.assertEqual(job["status"], "queued")
        with self.assertRaises(sj.ScanInProgress):
            sj.create_job("pro@example.com", "sp500", {})
        self.assertIsNone(sj.get_job("other@example.com", job["id"]))
        with psycopg.connect(PG_URL) as conn:
            conn.execute("UPDATE api_scan_jobs SET updated_at = NOW() - interval '30 minutes'")
        self.assertEqual(sj.get_job("pro@example.com", job["id"])["status"], "queued")   # waiting in line is fine
        with psycopg.connect(PG_URL) as conn:
            conn.execute("UPDATE api_scan_jobs SET updated_at = NOW() - interval '61 minutes'")
        stale = sj.get_job("pro@example.com", job["id"])
        self.assertEqual((stale["status"], stale["error"]), ("failed", "interrupted"))
        self.assertEqual(sj.create_job("pro@example.com", "sp500", {})["status"], "queued")  # slot free again
        with mock.patch.object(sj, "MAX_ACTIVE_JOBS", 1), self.assertRaises(sj.ScanBusy):
            sj.create_job("free@example.com", "sp500", {})
        # an expired job that later reaches a worker doesn't run or change status
        ran = []
        sj.submit(job["id"], lambda report: ran.append(1) or {})
        time.sleep(1)
        self.assertEqual(ran, [])
        self.assertEqual(sj.get_job("pro@example.com", job["id"])["error"], "interrupted")

    def test_running_job_without_heartbeat_expires(self):
        import psycopg

        sj = self.sj
        job = sj.create_job("pro@example.com", "sp500", {})
        self.assertTrue(sj._mark_running(job["id"]))
        with psycopg.connect(PG_URL) as conn:
            conn.execute("UPDATE api_scan_jobs SET updated_at = NOW() - interval '21 minutes'")
        self.assertEqual(sj.get_job("pro@example.com", job["id"])["status"], "failed")
        sj._finish(job["id"], "complete", {"phase": "complete"}, result={})        # a late finish can't revive it
        self.assertEqual(sj.get_job("pro@example.com", job["id"])["status"], "failed")

    def test_a_whole_scan_runs_in_the_background(self):
        from api import custom_scans

        params = _plan({"universe": "sp500", "filters": {"top_n": 10, "min_dollar_vol": 1_000_000, "min_gap": 0}}, "pro")
        syms = [f"T{i:03d}" for i in range(40)]
        job = self.sj.create_job("pro@example.com", "sp500", params)
        with mock.patch("ui.universe.load_sp500_universe", return_value=syms), \
                mock.patch("data.prices.fetch_price_data_parallel", side_effect=_fake_prices), \
                mock.patch("data.prices.fetch_price_data_batch", side_effect=_fake_prices), \
                mock.patch("ui.scan_providers.apply_alpaca_extended_prices", side_effect=lambda df: df), \
                mock.patch("db.runs.save_run") as save:
            self.sj.submit(job["id"], lambda report: custom_scans.run_scan(params, "pro@example.com", report))
            done = self._wait(job["id"])
        self.assertEqual(done["status"], "complete", done.get("error"))
        res = done["result"]
        self.assertEqual(res["symbols_scanned"], 40)
        self.assertEqual(res["label"], "SP500")
        self.assertGreater(res["total"], 0)
        self.assertLessEqual(res["total"], 10)
        row = res["setups"][0]
        self.assertTrue({"ticker", "score", "signals", "last"} <= set(row))
        self.assertIsNone(row["prob"])                                     # PreBreakout redacted below Premium
        self.assertEqual(save.call_args.kwargs["username"], "pro@example.com")   # in the web's scan history
        self.assertEqual(done["progress"]["phase"], "complete")

    def test_a_failed_scan_reports_a_safe_message(self):
        from api import custom_scans

        params = _plan({"universe": "us_market"}, "premium")
        job = self.sj.create_job("prem@example.com", "us_market", params)
        with mock.patch("ui.universe_us.us_market_symbols", return_value=[]):
            self.sj.submit(job["id"], lambda report: custom_scans.run_scan(params, "prem@example.com", report))
            done = self._wait(job["id"], user="prem@example.com")
        self.assertEqual(done["status"], "failed")
        self.assertIn("US market list isn't available", done["error"])


if __name__ == "__main__":
    unittest.main()


class TimestampTests(unittest.TestCase):
    def test_api_timestamps_always_carry_a_timezone(self):
        import datetime as dt

        from api.scans import json_safe

        naive = dt.datetime(2026, 10, 6, 13, 35)
        self.assertEqual(json_safe({"t": naive})["t"], "2026-10-06T13:35:00+00:00")
        aware = dt.datetime(2026, 10, 6, 9, 35, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        self.assertEqual(json_safe(aware), "2026-10-06T09:35:00-04:00")
        self.assertEqual(json_safe(dt.date(2026, 10, 6)), "2026-10-06")
