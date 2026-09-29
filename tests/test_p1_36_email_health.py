"""P1-36 — a warning when HSF email stops going out.

Incident 2026-09-28: the morning digest and alert emails sent nothing all morning
and nobody knew until the owner noticed a missing email.
"""
import datetime as dt
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from analytics import email_health as eh

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None
UTC = dt.timezone.utc
MON_NOON = dt.datetime(2026, 9, 28, 16, 0, tzinfo=UTC)      # Mon 12:00 ET
MON_EVENING = dt.datetime(2026, 9, 28, 23, 30, tzinfo=UTC)  # Mon 19:30 ET
SAT = dt.datetime(2026, 9, 26, 15, 0, tzinfo=UTC)


def run(job, at, **stats):
    return {"job": job, "at": at, "stats": stats}


class EvaluateTests(unittest.TestCase):
    def test_this_mornings_incident_is_a_warning_with_the_reasons(self):
        runs = [run("digest", "2026-09-28T12:35:00Z", sent=0, skipped={"plan_below_pro": 29, "unverified": 6}),
                run("digest", "2026-09-28T13:35:00Z", sent=0, skipped={"plan_below_pro": 29, "unverified": 6})]
        r = eh.evaluate(runs, now=MON_NOON)
        self.assertEqual(r["status"], "WARNING")
        d = r["jobs"]["digest"]
        self.assertEqual(d["status"], "WARNING")
        self.assertIn("no account qualified", d["detail"])
        self.assertIn("plan_below_pro=29", d["detail"])
        self.assertEqual(d["runs"], 2)

    def test_failed_sends_point_at_the_email_setup(self):
        r = eh.evaluate([run("digest", "2026-09-28T12:35:00Z", sent=0, skipped={"send_failed": 3})], now=MON_NOON)
        self.assertEqual(r["jobs"]["digest"]["status"], "WARNING")
        self.assertIn("3 send(s) failed", r["jobs"]["digest"]["detail"])

    def test_one_successful_run_is_healthy(self):
        runs = [run("digest", "2026-09-28T12:35:00Z", sent=0, skipped={"unverified": 1}),
                run("digest", "2026-09-28T13:35:00Z", sent=1, skipped={})]
        self.assertEqual(eh.evaluate(runs, now=MON_NOON)["jobs"]["digest"]["status"], "HEALTHY")

    def test_waiting_before_due_unknown_after(self):
        r = eh.evaluate([run("digest", "2026-09-28T12:35:00Z", sent=1)], now=MON_NOON)
        self.assertEqual(r["jobs"]["evening"]["status"], "WAITING")      # wrap due after 18:30 ET
        self.assertEqual(r["status"], "UNKNOWN")                         # alerts: nothing recorded
        r = eh.evaluate([run("digest", "2026-09-28T12:35:00Z", sent=1)], now=MON_EVENING)
        self.assertEqual(r["jobs"]["evening"]["status"], "UNKNOWN")

    def test_waiting_counts_as_healthy_overall(self):
        runs = [run("digest", "2026-09-28T12:35:00Z", sent=1), run("alerts", "2026-09-28T13:35:00Z",
                                                                   fired=2, emailed=2, email_failed=0)]
        self.assertEqual(eh.evaluate(runs, now=MON_NOON)["status"], "HEALTHY")

    def test_alert_failures_warn(self):
        r = eh.evaluate([run("alerts", "2026-09-28T13:35:00Z", fired=8, emailed=0, email_failed=8)], now=MON_NOON)
        self.assertEqual(r["jobs"]["alerts"]["status"], "WARNING")
        self.assertIn("8 alert email(s) failed", r["jobs"]["alerts"]["detail"])

    def test_weekend_judges_the_last_trading_day_and_ignores_older_days(self):
        runs = [run("digest", "2026-09-25T12:35:00Z", sent=1),
                run("digest", "2026-09-24T12:35:00Z", sent=0, skipped={"unverified": 1})]
        r = eh.evaluate(runs, now=SAT)
        self.assertEqual(r["day"], "2026-09-25")
        self.assertEqual(r["jobs"]["digest"]["status"], "HEALTHY")
        self.assertEqual(r["jobs"]["digest"]["runs"], 1)

    def test_bad_rows_are_ignored(self):
        r = eh.evaluate([{"job": "digest", "at": "garbage"}, {"job": "other", "at": "2026-09-28T12:35:00Z"}],
                        now=MON_NOON)
        self.assertEqual(r["jobs"]["digest"]["runs"], 0)


class RecordingTests(unittest.TestCase):
    def test_record_email_job_stores_counts_and_reports_failures_to_sentry(self):
        import scheduler.morning_digest as md

        with mock.patch("db.email_job_runs.record_email_run") as rec, mock.patch.object(md, "_capture") as cap:
            md.record_email_job("digest", {"sent": 2, "skipped": {"unverified": 1}})
            cap.assert_not_called()
            md.record_email_job("digest", {"sent": 0, "skipped": {"send_failed": 3}})
            md.record_email_job("alerts", {"fired": 1, "emailed": 0, "email_failed": 1})
        self.assertEqual(rec.call_count, 3)
        msgs = [str(c.args[0]) for c in cap.call_args_list]
        self.assertEqual(msgs, ["digest email: 3 send(s) failed this run", "alerts email: 1 send(s) failed this run"])

    def test_recording_never_raises(self):
        import scheduler.morning_digest as md

        with mock.patch("db.email_job_runs.record_email_run", side_effect=RuntimeError("db down")):
            md.record_email_job("digest", {"sent": 1})

    def test_digest_and_wrap_record_their_run(self):
        from tests.test_p1_41_email_prefs import _digest_run

        for job in ("digest", "evening"):
            with mock.patch("db.email_job_runs.record_email_run") as rec:
                _digest_run(opted_out=("off@example.com",), job=job)
            self.assertEqual(rec.call_args.args[0], job)
            self.assertEqual(rec.call_args.args[1]["sent"], 2)
            self.assertEqual(rec.call_args.args[1]["skipped"], {"unsubscribed": 1})

    def test_store_drops_unknown_jobs_and_survives_no_database(self):
        from db import email_job_runs

        self.assertFalse(email_job_runs.record_email_run("bogus", {}))
        with mock.patch("db.email_job_runs.get_neon_conn", return_value=None):
            self.assertFalse(email_job_runs.record_email_run("digest", {"sent": 1}))
            self.assertEqual(email_job_runs.recent_email_runs(), [])

    def test_alert_runner_counts_failed_sends_and_records(self):
        src = (ROOT / "scheduler" / "alert_runner.py").read_text()
        self.assertIn("            else:\n                email_failed += 1", src)
        self.assertIn('record_email_job("alerts", {"fired": fired, "emailed": emailed, "email_failed": email_failed})',
                      src)


class FrozenModelTests(unittest.TestCase):
    def test_run59_health_model_is_unchanged(self):
        from analytics import system_health

        self.assertEqual(system_health.SUBSYSTEMS, (
            "universe", "scanner", "research_capture", "maturation", "market_data",
            "cohort_parity", "forward_evidence", "workflows", "database", "artifact_freshness"))
        self.assertNotIn("email", (ROOT / "analytics" / "system_health.py").read_text().lower().split("subsystems")[1][:400])


@unittest.skipUnless(HAS_ST, "needs streamlit")
class CardTests(unittest.TestCase):
    def test_card_renders_each_job(self):
        from streamlit.testing.v1 import AppTest

        result = eh.evaluate([run("digest", "2026-09-28T12:35:00Z", sent=0, skipped={"unverified": 6})],
                             now=MON_NOON)
        at = AppTest.from_string("import streamlit as st\n"
                                 "from ui.email_health_view import render_email_health\n"
                                 "render_email_health(st.session_state['r'])\n", default_timeout=30)
        at.session_state["r"] = result
        at.run()
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("✉️ Email delivery", text)
        self.assertIn("Morning digest", text)
        self.assertIn("unverified=6", text)
        self.assertIn("Separate from the System Health status", " ".join(c.value for c in at.caption))

    def test_admin_tab_renders_the_card_after_system_health(self):
        src = (ROOT / "ui" / "admin_results_tab.py").read_text()
        self.assertLess(src.index("render_system_health()"), src.index("render_email_health()"))


if __name__ == "__main__":
    unittest.main()
