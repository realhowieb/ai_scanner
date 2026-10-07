"""System Health: parity freshness from the latest audit run, and scan slots that
never got a GitHub runner.

2026-10-07: STALE_ARTIFACT_MATURATION_PARITY_AUDIT kept growing after recovery
re-ran the audit (the read-only workflow only uploads an artifact; Health read
the committed file), and FAILED_SCAN recommended re-dispatching two 2026-10-05
slots whose jobs were cancelled before GitHub assigned a runner.
"""
import datetime as dt
import io
import json
import unittest
import zipfile
from unittest import mock

from analytics import market_calendar as mc
from analytics import recovery_policy as rp
from analytics import system_health as sh

UTC = dt.timezone.utc
WED = dt.datetime(2026, 9, 30, 23, 50, tzinfo=UTC)      # trading day, after all cycles


def _slot_runs(end, days=3):
    out = []
    d = (end - dt.timedelta(days=days)).date()
    while d <= end.date():
        for s in mc.expected_scan_slots(d):
            if s + dt.timedelta(minutes=5) <= end:
                out.append({"created_at": (s + dt.timedelta(seconds=10)).isoformat(),
                            "updated_at": (s + dt.timedelta(minutes=3)).isoformat(),
                            "conclusion": "success", "status": "completed", "event": "workflow_dispatch"})
        d += dt.timedelta(days=1)
    return out


def _codes(sub):
    return {f["code"]: f for f in sub["findings"]}


class ScanNoRunnerTests(unittest.TestCase):
    def _runs_with(self, idx_flags):
        runs = _slot_runs(WED)
        tuesday = [r for r in runs if r["created_at"].startswith("2026-09-29")]
        for i, no_runner in idx_flags:
            tuesday[i].update(conclusion="cancelled", no_runner=no_runner)
        return runs

    def test_runnerless_slots_are_not_failed_scans(self):
        sub = sh.eval_scanner(self._runs_with([(3, True), (4, True)]), [], WED)
        codes = _codes(sub)
        self.assertNotIn("FAILED_SCAN", codes)
        f = codes["SCAN_NO_RUNNER"]
        self.assertEqual(f["severity"], "INFO")
        self.assertFalse(f["automation_candidate"])
        self.assertEqual(len(f["evidence"]), 2)

    def test_real_failures_still_reported_next_to_runnerless(self):
        sub = sh.eval_scanner(self._runs_with([(3, True), (4, False)]), [], WED)
        codes = _codes(sub)
        self.assertEqual(len(codes["FAILED_SCAN"]["evidence"]), 1)
        self.assertEqual(len(codes["SCAN_NO_RUNNER"]["evidence"]), 1)

    def test_unknown_runner_state_counts_as_failed(self):
        runs = _slot_runs(WED)
        [r for r in runs if r["created_at"].startswith("2026-09-29")][2]["conclusion"] = "cancelled"
        self.assertIn("FAILED_SCAN", _codes(sh.eval_scanner(runs, [], WED)))

    def test_whole_day_without_runners_is_still_scanner_down(self):
        runs = _slot_runs(WED)
        for r in runs:
            if r["created_at"].startswith("2026-09-30"):
                r.update(conclusion="cancelled", no_runner=True)
        f = _codes(sh.eval_scanner(runs, [], WED))["STALE_SCANNER"]
        self.assertEqual(f["severity"], "CRITICAL")

    def test_recovery_watches_instead_of_redispatching(self):
        incident = {"incident_id": "scanner:SCAN_NO_RUNNER", "subsystem": "scanner", "severity": "INFO"}
        d = rp.decide(incident, {}, [], WED, {"open": False, "reasons": []})
        self.assertEqual(d["policy"], "WATCH")
        self.assertIsNone(d["action"])


class _Resp:
    def __init__(self, payload=None, content=b"", status=200):
        self._p, self.content, self.status_code = payload, content, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class MarkRunnerlessTests(unittest.TestCase):
    NOW = dt.datetime(2026, 10, 7, 0, 10, tzinfo=UTC)

    def test_flags_only_recent_unsuccessful_runs_without_a_runner(self):
        from scripts import system_health as script
        jobs = {
            1: [{"runner_id": 0, "runner_name": "", "steps": []}],
            2: [{"runner_id": 1000013463, "runner_name": "GitHub Actions 1", "steps": [{"name": "Run"}]}],
        }
        session = mock.Mock()
        session.get.side_effect = lambda url, timeout: _Resp({"jobs": jobs[int(url.split("/")[-2])]})
        runs = [
            {"id": 1, "created_at": "2026-10-05T19:35:12Z", "conclusion": "cancelled", "status": "completed"},
            {"id": 2, "created_at": "2026-10-05T20:35:12Z", "conclusion": "failure", "status": "completed"},
            {"id": 3, "created_at": "2026-10-06T19:35:12Z", "conclusion": "success", "status": "completed"},
            {"id": 4, "created_at": "2026-09-20T19:35:12Z", "conclusion": "cancelled", "status": "completed"},
        ]
        script.mark_runnerless(session, "o/r", runs, self.NOW)
        self.assertTrue(runs[0]["no_runner"])
        self.assertFalse(runs[1]["no_runner"])
        self.assertNotIn("no_runner", runs[2])
        self.assertNotIn("no_runner", runs[3])
        self.assertEqual(session.get.call_count, 2)

    def test_jobs_lookup_failure_leaves_run_unflagged(self):
        from scripts import system_health as script
        session = mock.Mock()
        session.get.return_value = _Resp(status=403)
        runs = [{"id": 1, "created_at": "2026-10-05T19:35:12Z", "conclusion": "cancelled", "status": "completed"}]
        script.mark_runnerless(session, "o/r", runs, self.NOW)
        self.assertNotIn("no_runner", runs[0])


def _zip(name, payload):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, json.dumps(payload))
    return buf.getvalue()


class ParityReportTests(unittest.TestCase):
    COMMITTED = {"generated_at": "2026-09-26T08:13:57+00:00", "src": "committed"}
    FRESH = {"generated_at": "2026-10-07T00:08:00+00:00", "src": "artifact"}

    def _with(self, committed, latest):
        from scripts import system_health as script
        with mock.patch.object(script, "_committed_parity_report", return_value=committed), \
                mock.patch.object(script, "_latest_parity_artifact", **latest):
            return script.parity_report()

    def test_latest_audit_run_wins_over_older_committed_file(self):
        self.assertEqual(self._with(self.COMMITTED, {"return_value": self.FRESH})["src"], "artifact")

    def test_committed_file_used_when_github_unavailable(self):
        rep = self._with(self.COMMITTED, {"side_effect": RuntimeError("no GitHub token")})
        self.assertEqual(rep["src"], "committed")

    def test_older_artifact_does_not_replace_newer_commit(self):
        old = {"generated_at": "2026-09-01T00:00:00+00:00", "src": "artifact"}
        self.assertEqual(self._with(self.COMMITTED, {"return_value": old})["src"], "committed")

    def test_fresh_report_clears_the_staleness_finding(self):
        from scripts import system_health as script
        now = dt.datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
        for rep, stale in ((self.COMMITTED, True), (self.FRESH, False)):
            with mock.patch.object(script, "_committed_parity_report", return_value=self.COMMITTED), \
                    mock.patch.object(script, "_latest_parity_artifact", return_value=rep):
                gen = script.parity_report()["generated_at"]
            sub = sh.eval_artifacts({"maturation_parity_audit": {"generated_at": gen}}, now)
            self.assertEqual("STALE_ARTIFACT_MATURATION_PARITY_AUDIT" in _codes(sub), stale)

    def test_reads_the_json_from_the_audit_artifact(self):
        from scripts import system_health as script
        session = mock.Mock()

        def get(url, **_):
            if url.endswith("/artifacts"):
                return _Resp({"artifacts": [{"name": "maturation-parity-audit", "expired": False,
                                             "archive_download_url": "https://x/zip"}]})
            return _Resp(content=_zip("maturation_parity_audit.json", self.FRESH))
        session.get.side_effect = get
        rep = script._run_artifact_json(session, "o/r", 9, "maturation-parity-audit",
                                        "maturation_parity_audit.json")
        self.assertEqual(rep, self.FRESH)


if __name__ == "__main__":
    unittest.main()
