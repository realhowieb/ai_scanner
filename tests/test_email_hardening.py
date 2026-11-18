"""Email hardening after the Nov 14-17 2026 outage (Resend un-verified the sender domain)."""
import datetime as dt
import smtplib
import unittest
from unittest import mock

UTC = dt.timezone.utc


class _Res:
    def __init__(self, code, body):
        self.status_code, self._body = code, body

    def json(self):
        return self._body


class _Session:
    def __init__(self, code=200, body=None, exc=None):
        self.code, self.body, self.exc, self.calls = code, body, exc, []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if self.exc:
            raise self.exc
        return _Res(self.code, self.body)


class ResendDomainStatus(unittest.TestCase):
    def setUp(self):
        from ui import email_setup

        email_setup._domain_cache.clear()
        self.es = email_setup

    def status(self, session, domain="hsfinest.com", key="re_key"):
        return self.es.resend_domain_status(domain, key=key, session=session, now=0.0)

    def test_verified_unverified_missing(self):
        rows = {"data": [{"name": "hsfinest.com", "status": "verified"},
                         {"name": "ai.hsfinest.com", "status": "failed"}]}
        self.assertEqual(self.status(_Session(body=rows))["state"], "verified")
        self.es._domain_cache.clear()
        s = self.status(_Session(body=rows), domain="ai.hsfinest.com")
        self.assertEqual((s["state"], s["resend_status"]), ("unverified", "failed"))
        self.es._domain_cache.clear()
        self.assertEqual(self.status(_Session(body=rows), domain="other.com")["state"], "missing")

    def test_unknown_never_alarms(self):
        for session in (_Session(code=401, body={}), _Session(exc=OSError("down"))):
            self.es._domain_cache.clear()
            self.assertEqual(self.status(session)["state"], "unknown")
        self.es._domain_cache.clear()
        with mock.patch.object(self.es, "_resend_key", return_value=None):
            s = self.es.resend_domain_status("hsfinest.com", session=_Session(body={}), now=0.0)
        self.assertEqual((s["state"], s["why"]), ("unknown", "no Resend API key"))

    def test_cached_and_key_not_returned(self):
        sess = _Session(body={"data": [{"name": "hsfinest.com", "status": "verified"}]})
        s1 = self.status(sess, key="re_secret")
        self.es.resend_domain_status("hsfinest.com", key="re_secret", session=sess, now=10.0)
        self.assertEqual(len(sess.calls), 1)
        self.assertNotIn("re_secret", repr(s1))

    def test_row_and_problem(self):
        d = self.es.describe_smtp("smtp.resend.com", "resend", "p", "alerts@hsfinest.com")
        bad = _Session(body={"data": [{"name": "hsfinest.com", "status": "not_started"}]})
        with mock.patch.object(self.es, "_resend_key", return_value="re_k"):
            row = self.es.sender_domain_row(d, session=bad)
            self.assertEqual(row["status"], "WARNING")
            self.assertIn("not verified", row["detail"])
            self.es._domain_cache.clear()
            with mock.patch.object(self.es, "describe_from_config", return_value=d):
                self.assertIsNotNone(self.es.sender_domain_problem(session=bad))
                self.es._domain_cache.clear()
                self.assertIsNone(self.es.sender_domain_problem(session=_Session(code=500, body={})))


class SendFailureCaptureThrottle(unittest.TestCase):
    def test_one_capture_per_kind_per_15_min(self):
        from ui import email_utils

        email_utils._last_capture.clear()
        err = smtplib.SMTPDataError(550, b"domain is not verified")
        with mock.patch.object(email_utils, "_capture") as cap:
            self.assertTrue(email_utils._capture_send_failure(err, now=0.0))
            self.assertFalse(email_utils._capture_send_failure(err, now=60.0))
            self.assertTrue(email_utils._capture_send_failure(OSError("x"), now=60.0))
            self.assertTrue(email_utils._capture_send_failure(err, now=901.0))
        self.assertEqual(cap.call_count, 3)


class RetryFailedSends(unittest.TestCase):
    def setUp(self):
        from api import ops_jobs

        ops_jobs._retry_at.clear()
        self.oj = ops_jobs

    def run_once(self, fn, t, claimed=True):
        with mock.patch("services.ops_notify.once", return_value=claimed) as once, \
             mock.patch("services.ops_notify.release") as release:
            self.oj._run_once("weekly_recap", "2026-W46", fn, retry_failed_sends=True, clock=lambda: t)
        return once, release

    def test_failed_sends_release_and_wait_an_hour(self):
        fn = mock.Mock(return_value={"sent": 0, "failed": 19})
        _, release = self.run_once(fn, 0.0)
        release.assert_called_once_with("weekly_recap", "2026-W46")
        once, _ = self.run_once(fn, 1800.0)  # too soon: not even claimed
        once.assert_not_called()
        self.assertEqual(fn.call_count, 1)
        fn.return_value = {"sent": 19, "failed": 0}
        _, release = self.run_once(fn, 3601.0)
        release.assert_not_called()
        self.assertEqual(fn.call_count, 2)
        self.assertNotIn(("weekly_recap", "2026-W46"), self.oj._retry_at)

    def test_without_flag_failed_sends_keep_the_claim(self):
        fn = mock.Mock(return_value={"failed": 3})
        with mock.patch("services.ops_notify.once", return_value=True), \
             mock.patch("services.ops_notify.release") as release:
            self.oj._run_once("x", "k", fn)
        release.assert_not_called()

    def test_recap_due_through_sunday(self):
        sat = dt.datetime(2026, 11, 14, 15, 0, tzinfo=UTC)
        sun = dt.datetime(2026, 11, 15, 20, 0, tzinfo=UTC)
        self.assertEqual(self.oj.weekly_recap_due(sat), self.oj.weekly_recap_due(sun))
        self.assertIsNone(self.oj.weekly_recap_due(sat.replace(hour=13)))
        self.assertIsNone(self.oj.weekly_recap_due(dt.datetime(2026, 11, 16, 1, 0, tzinfo=UTC)))

    def test_sender_check_reports_to_sentry(self):
        with mock.patch("ui.email_setup.sender_domain_problem", return_value="x is not verified"), \
             mock.patch("ui.monitoring.capture") as cap:
            self.assertEqual(self.oj.check_sender_domain(), "x is not verified")
        cap.assert_called_once()
        with mock.patch("ui.email_setup.sender_domain_problem", return_value=None), \
             mock.patch("ui.monitoring.capture") as cap:
            self.assertIsNone(self.oj.check_sender_domain())
        cap.assert_not_called()



class AdminEmailCard(unittest.TestCase):
    def test_sender_row_first_and_worst_status_wins(self):
        from api import admin

        row = {"job": "sender", "label": "Sender domain", "status": "WARNING", "detail": "not verified"}
        healthy = {"status": "HEALTHY", "day": "2026-11-17",
                   "jobs": {"digest": {"label": "Morning digest", "status": "HEALTHY", "detail": "3 sent"}}}
        with mock.patch("db.email_job_runs.recent_email_runs", return_value=[]), \
             mock.patch("analytics.email_health.evaluate", return_value=healthy), \
             mock.patch("ui.email_setup.sender_domain_row", return_value=row):
            out = admin.email_health()
        self.assertEqual(out["status"], "WARNING")
        self.assertEqual([j["job"] for j in out["jobs"]], ["sender", "digest"])


if __name__ == "__main__":
    unittest.main()
