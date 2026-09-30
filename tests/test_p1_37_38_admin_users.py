"""P1-37 / P1-38 — Admin Users: email usernames, bcrypt passwords, and
verification / admin management without hand-written SQL."""
import importlib.util
import unittest
from unittest import mock

HAS_ST = importlib.util.find_spec("streamlit") is not None
HAS_BCRYPT = importlib.util.find_spec("bcrypt") is not None


class FakeCursor:
    def __init__(self, db):
        self.db, self.rowcount, self._result = db, 0, None

    def execute(self, sql, params=()):
        self.db.sql.append((" ".join(sql.split()), params))
        s = sql.lower()
        if s.lstrip().startswith("select 1 from users"):
            self._result = (1,) if params[0] in self.db.users else None
        elif s.lstrip().startswith("insert into users"):
            self.db.users[params[0]] = {"password": params[2], "tier": params[3]}
            self.rowcount = 1
        elif s.lstrip().startswith("update users"):
            self.rowcount = 1 if params[-1] in self.db.users else 0

    def fetchone(self):
        return self._result

    def fetchall(self):
        return []

    def close(self):
        pass


class FakeDB:
    def __init__(self, users=None):
        self.users, self.sql = dict(users or {}), []

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def close(self):
        pass


def patched(db):
    return mock.patch("db.user_admin._conn", return_value=db)


class CreateUserTests(unittest.TestCase):
    def test_rejects_non_email_usernames(self):
        from db.user_admin import create_user

        db = FakeDB()
        with patched(db):
            for bad in ("howard", "k21", "a@b", "x @y.com"):
                ok, msg = create_user(bad, "Name", "pw12345")
                self.assertFalse(ok, bad)
                self.assertIn("valid email", msg)
        self.assertEqual(db.sql, [])                       # never reached the database

    @unittest.skipUnless(HAS_BCRYPT, "needs bcrypt (installed in production)")
    def test_stores_lowercase_email_and_bcrypt_hash(self):
        import bcrypt

        from db.user_admin import create_user

        db = FakeDB()
        with patched(db):
            ok, msg = create_user("  Jane.Doe@Example.COM ", "Jane", "Maple-river-42!", tier="pro")
        self.assertTrue(ok, msg)
        self.assertIn("jane.doe@example.com", db.users)
        stored = db.users["jane.doe@example.com"]["password"]
        self.assertNotEqual(stored, "Maple-river-42!")             # never plain text
        self.assertTrue(bcrypt.checkpw(b"Maple-river-42!", stored.encode()))
        self.assertIn("isn't verified yet", msg)

    def test_existing_account_is_reported_not_overwritten(self):
        from db.user_admin import create_user

        db = FakeDB({"jane@example.com": {"password": "old", "tier": "premium"}})
        with patched(db):
            ok, msg = create_user("JANE@example.com", "Jane", "Maple-river-42!")
        self.assertFalse(ok)
        self.assertIn("already exists", msg)
        self.assertEqual(db.users["jane@example.com"]["password"], "old")
        self.assertFalse(any(s.startswith("INSERT") for s, _ in db.sql))

    def test_required_fields_and_plan(self):
        from db.user_admin import create_user

        with patched(FakeDB()):
            self.assertFalse(create_user("a@example.com", "", "pw")[0])
            self.assertFalse(create_user("a@example.com", "A", "")[0])
            self.assertFalse(create_user("a@example.com", "A", "pw", tier="admin")[0])


class StatusActionTests(unittest.TestCase):
    def test_mark_verified_only_for_email_accounts(self):
        from db.user_admin import mark_email_verified

        db = FakeDB({"a@example.com": {}})
        with patched(db):
            self.assertFalse(mark_email_verified("k21")[0])
            self.assertTrue(mark_email_verified("A@Example.com")[0])
        self.assertIn(("UPDATE users SET email_verified = TRUE WHERE lower(username) = %s",
                       ("a@example.com",)), db.sql)

    def test_grant_admin_keeps_the_plan(self):
        from db.user_admin import set_admin

        db = FakeDB({"a@example.com": {}})
        with patched(db):
            ok, _ = set_admin("a@example.com", True, acting_user="me@example.com")
        self.assertTrue(ok)
        self.assertEqual(db.sql[-1][0], "UPDATE users SET is_admin = TRUE WHERE lower(username) = %s")

    def test_revoke_admin_clears_legacy_admin_tier_only(self):
        from db.user_admin import set_admin

        db = FakeDB({"a@example.com": {}})
        with patched(db):
            ok, _ = set_admin("a@example.com", False, acting_user="me@example.com")
        self.assertTrue(ok)
        sql = db.sql[-1][0]
        self.assertIn("is_admin = FALSE", sql)
        self.assertIn("CASE WHEN lower(COALESCE(tier, '')) = 'admin' THEN 'basic' ELSE tier END", sql)

    def test_cannot_revoke_your_own_admin(self):
        from db.user_admin import set_admin

        db = FakeDB({"me@example.com": {}})
        with patched(db):
            ok, msg = set_admin("ME@example.com", False, acting_user="me@example.com")
        self.assertFalse(ok)
        self.assertIn("your own", msg)
        self.assertEqual(db.sql, [])

    def test_no_database_means_no_change(self):
        from db.user_admin import create_user, mark_email_verified, set_admin

        with mock.patch("db.user_admin._conn", return_value=None):
            self.assertFalse(create_user("a@example.com", "A", "pw")[0])
            self.assertFalse(mark_email_verified("a@example.com")[0])
            self.assertFalse(set_admin("a@example.com", True, acting_user="x@example.com")[0])


SCRIPT = '''
import streamlit as st
from ui.admin_users import render_admin_users_panel
render_admin_users_panel(st.session_state["me"], st.session_state["admins"], "ok")
'''


@unittest.skipUnless(HAS_ST, "needs streamlit")
class PanelTests(unittest.TestCase):
    ROWS = [{"username": "me@example.com", "full_name": "Me", "tier": "premium", "is_active": True,
             "is_admin": True, "email_verified": True, "created_at": None},
            {"username": "new@example.com", "full_name": "New", "tier": "basic", "is_active": True,
             "is_admin": False, "email_verified": False, "created_at": None},
            {"username": "k21", "full_name": "K", "tier": "pro", "is_active": True,
             "is_admin": False, "email_verified": False, "created_at": None}]

    def run_panel(self, me, admins, is_admin, select=None):
        import pandas as pd
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        at.session_state["me"], at.session_state["admins"] = me, admins
        at.session_state["is_admin"] = is_admin
        base = pd.DataFrame([{k: r[k] for k in ("username", "full_name", "tier", "is_active", "created_at")}
                             for r in self.ROWS])
        with mock.patch("ui.admin_users.fetch_all_users", return_value=base), \
             mock.patch("db.user_admin.fetch_users_for_admin", return_value=self.ROWS):
            at.run()
            if at.checkbox:
                at.checkbox(key="enable_admin_users").check().run()
                if select:
                    next(sb for sb in at.selectbox if sb.label == "Select user to edit").set_value(select).run()
        return at

    def test_hidden_for_non_admins(self):
        at = self.run_panel("user@example.com", set(), False)
        self.assertFalse(at.exception)
        self.assertEqual(len(at.expander), 0)

    def test_db_admin_sees_panel_without_the_secret(self):
        at = self.run_panel("me@example.com", {"howard"}, True)
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        self.assertEqual(len(at.expander), 1)
        cols = list(at.dataframe[0].value.columns)
        self.assertIn("email_verified", cols)
        self.assertIn("is_admin", cols)

    def test_unverified_email_account_gets_verify_actions(self):
        at = self.run_panel("me@example.com", set(), True, select="new@example.com")
        self.assertFalse(at.exception, [str(e.value)[:300] for e in at.exception])
        labels = [b.label for b in at.button]
        self.assertIn("Mark email verified", labels)
        self.assertIn("Resend verification email", labels)
        self.assertIn("Grant admin", labels)
        self.assertTrue(at.button(key="admin_role_btn").disabled)   # needs the confirm box

    def test_non_email_account_explains_instead_of_verify(self):
        at = self.run_panel("me@example.com", set(), True, select="k21")
        labels = [b.label for b in at.button]
        self.assertNotIn("Mark email verified", labels)
        self.assertIn("isn't an email address", " ".join(c.value for c in at.caption))

    def test_create_user_rejects_non_email_on_the_page(self):
        import pandas as pd
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_string(SCRIPT, default_timeout=60)
        at.session_state["me"], at.session_state["admins"], at.session_state["is_admin"] = "me@example.com", set(), True
        with mock.patch("ui.admin_users.fetch_all_users", return_value=pd.DataFrame()), \
             mock.patch("db.user_admin._conn") as conn:
            at.run()
            at.checkbox(key="enable_admin_users").check().run()
            at.text_input(key="create_user_email").input("k21")
            at.text_input(key="create_user_name").input("K")
            at.text_input(key="create_user_password").input("pw")
            at.button[0].click().run()
        self.assertIn("valid email", at.error[0].value)
        conn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
