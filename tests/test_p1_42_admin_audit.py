"""P1-42 — admin audit log: every successful admin action is recorded
(actor, action, account, details), never with secrets, and shown on the Admin page."""
import importlib.util
import json
import unittest
from unittest import mock

HAS_ST = True
HAS_BCRYPT = importlib.util.find_spec("bcrypt") is not None


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, actor, action, target, detail=None):
        self.events.append((actor, action, target, detail or {}))
        return True


class StoreTests(unittest.TestCase):
    def test_records_lowercase_and_strips_secrets(self):
        from db import admin_audit

        conn = mock.MagicMock()
        cur = conn.cursor.return_value
        with mock.patch("db.admin_audit.get_neon_conn", return_value=conn):
            ok = admin_audit.record_admin_event("Me@Example.com", "update_user", "You@Example.com",
                                                {"tier": ["basic", "pro"], "password": "x", "reset_token": "y"})
        self.assertTrue(ok)
        insert = [c for c in cur.execute.call_args_list if "INSERT INTO hsf_admin_events" in c.args[0]][0]
        actor, action, target, detail = insert.args[1]
        self.assertEqual((actor, action, target), ("me@example.com", "update_user", "you@example.com"))
        self.assertEqual(json.loads(detail), {"tier": ["basic", "pro"]})

    def test_never_raises(self):
        from db import admin_audit

        with mock.patch("db.admin_audit.get_neon_conn", side_effect=RuntimeError("down")):
            self.assertFalse(admin_audit.record_admin_event("a", "x", "b"))
            self.assertEqual(admin_audit.recent_admin_events(), [])
        with mock.patch("db.admin_audit.get_neon_conn", return_value=None):
            self.assertFalse(admin_audit.record_admin_event("a", "x", "b"))

    def test_recent_events_parse_and_describe(self):
        from db import admin_audit

        conn = mock.MagicMock()
        conn.cursor.return_value.fetchall.return_value = [
            {"at": "t2", "actor": "me@example.com", "action": "update_user", "target": "a@example.com",
             "detail": json.dumps({"tier": ["basic", "pro"], "active": [True, False]})},
            ("t1", "me@example.com", "grant_admin", "b@example.com", None),
        ]
        with mock.patch("db.admin_audit.get_neon_conn", return_value=conn):
            events = admin_audit.recent_admin_events(limit=10)
        self.assertEqual([e["action"] for e in events], ["update_user", "grant_admin"])
        self.assertEqual(admin_audit.describe(events[0]["detail"]), "active: True → False, tier: basic → pro")
        self.assertEqual(events[1]["detail"], {})


class ActionRecordingTests(unittest.TestCase):
    def setUp(self):
        from tests.test_p1_37_38_admin_users import FakeDB

        self.FakeDB = FakeDB
        self.rec = Recorder()
        p = mock.patch("db.admin_audit.record_admin_event", side_effect=self.rec)
        p.start()
        self.addCleanup(p.stop)

    @unittest.skipUnless(HAS_BCRYPT, "needs bcrypt (installed in production)")
    def test_create_user_is_recorded_without_the_password(self):
        from db.user_admin import create_user

        with mock.patch("db.user_admin._conn", return_value=self.FakeDB()):
            self.assertTrue(create_user("New@Example.com", "N", "Maple-river-42!", tier="pro", actor="me@example.com")[0])
        self.assertEqual(self.rec.events, [("me@example.com", "create_user", "new@example.com",
                                            {"tier": "pro", "active": True})])
        self.assertNotIn("Maple-river-42!", json.dumps(self.rec.events))

    def test_verify_and_admin_changes_are_recorded(self):
        from db.user_admin import mark_email_verified, set_admin

        with mock.patch("db.user_admin._conn", return_value=self.FakeDB({"a@example.com": {}})):
            mark_email_verified("a@example.com", actor="me@example.com")
            set_admin("a@example.com", True, acting_user="me@example.com")
            set_admin("a@example.com", False, acting_user="me@example.com")
        self.assertEqual([e[1] for e in self.rec.events], ["mark_email_verified", "grant_admin", "revoke_admin"])
        self.assertTrue(all(e[0] == "me@example.com" and e[2] == "a@example.com" for e in self.rec.events))

    def test_failed_or_refused_actions_are_not_recorded(self):
        from db.user_admin import create_user, mark_email_verified, set_admin

        with mock.patch("db.user_admin._conn", return_value=self.FakeDB()):
            create_user("k21", "K", "pw", actor="me@example.com")                 # not an email
            mark_email_verified("missing@example.com", actor="me@example.com")    # no such row
            set_admin("me@example.com", False, acting_user="me@example.com")      # self-revoke refused
        self.assertEqual(self.rec.events, [])


SCRIPT = '''
from utils.st_compat import st
from ui.admin_users import render_admin_users_panel
render_admin_users_panel("me@example.com", set(), "ok")
'''


if __name__ == "__main__":
    unittest.main()
