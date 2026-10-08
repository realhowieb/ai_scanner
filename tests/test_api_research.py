"""Internal research endpoints: admin gate, separation of features and outcomes,
pagination, validation, outage handling, and immutable dataset versions."""
import datetime as dt
import re
import unittest
from unittest import mock

from tests.research_fixtures import opp, scan
from tests.test_api_v1 import DEPS, ApiTestCase

OUTCOME_KEY = re.compile(r"return|mfe|mae|benchmark|excess|matur|outcome|label|certified", re.I)
ROUTES = ("/v1/research/datasets", "/v1/research/datasets/hsf-ml-2026-10-08-v1", "/v1/research/coverage",
          "/v1/research/observations", "/v1/research/observations/1", "/v1/research/features/1")


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


@unittest.skipUnless(DEPS, "needs fastapi, httpx, PyJWT and bcrypt")
class ResearchApiTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        from api import research

        research.clear_cache()
        self.addCleanup(research.clear_cache)
        today = dt.datetime.now(dt.timezone.utc).date()
        self.d1 = (today - dt.timedelta(days=20)).isoformat()
        self.d2 = (today - dt.timedelta(days=19)).isoformat()
        self.rows = [
            opp(1, "AAA", f"{self.d1}T16:40:00", score=81, scored=True, r1=0.01, r3=0.02, r5=0.05, mfe=0.07,
                mae=-0.02, b1=0.001, b3=0.002, b5=0.003),
            opp(2, "BBB", f"{self.d1}T16:40:00", score=64),
            opp(3, "AAA", f"{self.d2}T16:40:00", score=70),
        ]
        self.scans = [scan("AAA", f"{self.d1}T16:35:00", price=12.0)]
        self.versions = {}
        p = mock.patch
        p("db.research_datasets.fetch_opportunity_rows", side_effect=self._rows).start()
        p("db.research_datasets.fetch_scan_records", side_effect=lambda t, s, e, **k: list(self.scans)).start()
        p("db.research_datasets.fetch_opportunity_row", side_effect=self._row).start()
        p("db.research_datasets.list_dataset_versions", side_effect=lambda **k: list(self.versions.values())).start()
        p("db.research_datasets.get_dataset_version", side_effect=lambda n, **k: self.versions.get(n)).start()

    def _rows(self, start, end, **k):
        return [r for r in self.rows if start <= r["fired_at"] < end]

    def _row(self, oid, **k):
        row = next((r for r in self.rows if r["id"] == oid), None)
        if row is None:
            return None
        return {"row": row, "snapshot": [r for r in self.rows if r["fired_at"] == row["fired_at"]]}

    def admin(self):
        return self.auth(self.login("boss@example.com").json()["access_token"])

    def get(self, path, **kw):
        return self.client.get(path, headers=self.admin(), **kw)

    # ---- auth
    def test_signed_out_is_401_and_non_admin_is_403(self):
        pro = self.auth(self.login("pro@example.com").json()["access_token"])
        for path in ROUTES:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 401)
                r = self.client.get(path, headers=pro)
                self.assertEqual(r.status_code, 403)
                self.assertIn("admin", r.json()["detail"])

    def test_routes_are_read_only_and_marked_internal(self):
        from api import main

        spec = main.create_app(self.settings).openapi()
        research = {p: v for p, v in spec["paths"].items() if p.startswith("/v1/research")}
        self.assertEqual(len(research), 6)
        for path, ops in research.items():
            self.assertEqual(set(ops), {"get"}, path)
            self.assertTrue(ops["get"]["x-internal"])
            self.assertIn("Internal", ops["get"]["summary"])
            self.assertEqual(ops["get"]["tags"], ["research (internal, admin only)"])

    # ---- features vs outcomes
    def test_feature_endpoint_never_returns_outcomes(self):
        r = self.get("/v1/research/features/1")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["features"]["hsf_score"], 81)
        self.assertEqual(body["features"]["price"], 12.0)
        self.assertEqual([k for k in _keys(body) if OUTCOME_KEY.search(k)], [])
        for v in (0.05, 0.07, -0.02, 0.003):
            self.assertNotIn(v, body["features"].values())

    def test_observation_outcome_only_when_asked_and_separate(self):
        plain = self.get("/v1/research/observations/1").json()
        self.assertNotIn("outcome", plain)
        self.assertEqual([k for k in _keys(plain["features"]) if OUTCOME_KEY.search(k)], [])
        full = self.get("/v1/research/observations/1?include_outcome=true").json()
        self.assertEqual(full["outcome"]["labels"]["return_5d"], 0.05)
        self.assertNotIn("return_5d", full["features"])
        self.assertEqual(full["observation"]["rank"], 1)

    def test_unknown_observation_is_404(self):
        self.assertEqual(self.get("/v1/research/features/999").status_code, 404)
        self.assertEqual(self.get("/v1/research/observations/999").status_code, 404)

    # ---- list + pagination + filters
    def test_pagination(self):
        first = self.get("/v1/research/observations?limit=2").json()
        self.assertEqual(first["total"], 3)
        self.assertEqual([i["observation"]["observation_id"] for i in first["items"]], [1, 2])
        self.assertEqual(first["next_offset"], 2)
        self.assertTrue(all("outcome" not in i for i in first["items"]))
        last = self.get("/v1/research/observations?limit=2&offset=2").json()
        self.assertEqual([i["observation"]["observation_id"] for i in last["items"]], [3])
        self.assertIsNone(last["next_offset"])
        self.assertEqual(self.get("/v1/research/observations?limit=501").status_code, 422)

    def test_filters_are_explicit_and_echoed(self):
        r = self.get("/v1/research/observations?ticker=aaa&min_score=75&include_outcomes=true").json()
        self.assertEqual([i["observation"]["observation_id"] for i in r["items"]], [1])
        self.assertEqual(r["filters"]["ticker"], "AAA")
        self.assertIsNone(r["filters"]["setup"])
        self.assertIn("outcome", r["items"][0])
        self.assertEqual(self.get("/v1/research/observations?matured_only=true").json()["total"], 1)
        self.assertEqual(self.get("/v1/research/observations?certified_only=true").json()["total"], 1)

    def test_bad_filters_are_422(self):
        for q in ("horizon=20", "horizon=10", "start_date=2026-10-02&end_date=2026-10-01",
                  "start_date=2024-01-01&end_date=2026-10-01", "min_score=101"):
            with self.subTest(q=q):
                self.assertEqual(self.get(f"/v1/research/observations?{q}").status_code, 422)

    def test_empty_window(self):
        r = self.get("/v1/research/observations?start_date=2026-01-01&end_date=2026-01-02").json()
        self.assertEqual((r["total"], r["items"]), (0, []))
        c = self.get("/v1/research/coverage?start_date=2026-01-01&end_date=2026-01-02").json()
        self.assertEqual(c["coverage"]["total_observations"], 0)

    # ---- coverage
    def test_coverage_is_real(self):
        r = self.get("/v1/research/coverage")
        self.assertEqual(r.status_code, 200, r.text)
        cov = r.json()["coverage"]
        self.assertEqual(cov["total_observations"], 3)
        self.assertEqual(cov["matured_observations"], 1)
        self.assertEqual(cov["pending_observations"], 2)
        self.assertEqual(cov["features"]["hsf_score"], 1.0)
        self.assertEqual(cov["features"]["price"], round(1 / 3, 4))
        self.assertEqual(cov["benchmark"]["5d"]["present"], 1)
        self.assertIn("timings", r.json())

    def test_window_is_cached(self):
        from db import research_datasets

        self.get("/v1/research/coverage")
        self.get("/v1/research/observations")
        self.assertEqual(research_datasets.fetch_opportunity_rows.call_count, 1)

    def test_database_outage_is_503_not_empty(self):
        from db.research_datasets import ResearchDataUnavailable

        with mock.patch("db.research_datasets.fetch_opportunity_rows", side_effect=ResearchDataUnavailable("x")):
            r = self.get("/v1/research/coverage?start_date=2026-02-01&end_date=2026-02-02")
        self.assertEqual(r.status_code, 503)

    # ---- dataset versions
    def _finalize(self, name="hsf-ml-2026-10-08-v1"):
        from analytics import research_dataset as rd

        filters = {"start_date": self.d1, "end_date": self.d2}
        built = rd.build_dataset(self.rows, self.scans, filters=filters)
        self.versions[name] = {"dataset_version": name, "created_at": dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc),
                               "feature_schema_version": 1, "label_schema_version": 1,
                               "fingerprint": built["metadata"]["fingerprint"],
                               "observation_count": built["metadata"]["observation_count"],
                               "observation_ids": built["observation_ids"], "metadata": built["metadata"]}
        return built

    def test_datasets_list_and_schemas(self):
        self._finalize()
        body = self.get("/v1/research/datasets").json()
        self.assertEqual([d["dataset_version"] for d in body["items"]], ["hsf-ml-2026-10-08-v1"])
        self.assertTrue(body["items"][0]["finalized"])
        self.assertEqual(body["feature_schema_version"], 1)
        self.assertEqual(len(body["feature_schema"]), 33)
        self.assertEqual({lab["horizon_days"] for lab in body["label_schema"]}, {1, 3, 5})

    def test_dataset_verify_reproduces_and_detects_drift(self):
        self._finalize()
        r = self.get("/v1/research/datasets/hsf-ml-2026-10-08-v1?verify=true").json()
        self.assertTrue(r["verification"]["reproducible"])
        self.rows[1] = {**self.rows[1], "outcome_computed_at": self.rows[1]["fired_at"] + dt.timedelta(days=8),
                        "return_1d": 0.02}
        r = self.get("/v1/research/datasets/hsf-ml-2026-10-08-v1?verify=true").json()
        self.assertFalse(r["verification"]["reproducible"])
        self.assertEqual(r["fingerprint"], r["verification"]["stored_fingerprint"])  # stored version untouched
        self.assertEqual(self.get("/v1/research/datasets/hsf-ml-1999-01-01-v1").status_code, 404)

    def test_observations_by_dataset_version(self):
        self._finalize()
        self.rows.append(opp(4, "CCC", f"{self.d2}T16:40:00"))   # arrived after finalization
        r = self.get("/v1/research/observations?dataset_version=hsf-ml-2026-10-08-v1").json()
        self.assertEqual([i["observation"]["observation_id"] for i in r["items"]], [1, 2, 3])
        self.assertEqual(self.get("/v1/research/observations?dataset_version=hsf-ml-1999-01-01-v1").status_code, 404)


if __name__ == "__main__":
    unittest.main()
