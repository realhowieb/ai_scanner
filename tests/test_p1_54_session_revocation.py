"""P1-54: a password reset or a deactivated account ends existing sessions."""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_STREAMLIT = importlib.util.find_spec("streamlit") is not None


class _Cursor:
    def __init__(self, row=None, rowcount=0, raises=None):
        self.row, self.rowcount, self.raises = row, rowcount, raises
        self.sql = []

    def execute(self, sql, params=None):
        if self.raises:
            raise self.raises
        self.sql.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.row

    def close(self):
        pass


class _Conn:
    def __init__(self, cursor):
        self.cur = cursor
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def close(self):
        pass


@unittest.skipUnless(_STREAMLIT, "ui.auth_sessions imports streamlit")
class SessionLookupTests(unittest.TestCase):
    def _lookup(self, row):
        from ui import auth_sessions as s

        cur = _Cursor(row=row)
        with mock.patch.object(s, "get_neon_conn", return_value=_Conn(cur)), \
                mock.patch.object(s, "ensure_auth_sessions_schema"):
            return s.get_username_for_session("sid-1"), cur

    def test_lookup_requires_an_active_account(self):
        user, cur = self._lookup(("member@example.com",))
        self.assertEqual(user, "member@example.com")
        select = cur.sql[0][0]
        self.assertIn("JOIN users u ON lower(u.username) = lower(s.username)", select)
        self.assertIn("u.is_active IS NOT FALSE", select)
        self.assertIn("s.expires_at > now()", select)

    def test_deactivated_or_unknown_account_gets_no_session(self):
        # The join filters the deactivated account out, so the database returns no row.
        user, cur = self._lookup(None)
        self.assertIsNone(user)
        self.assertEqual(len(cur.sql), 1)  # no last_seen_at update for a dead session

    def test_revoke_deletes_every_session_of_the_user(self):
        from ui import auth_sessions as s

        cur = _Cursor(rowcount=3)
        conn = _Conn(cur)
        with mock.patch.object(s, "get_neon_conn", return_value=conn), \
                mock.patch.object(s, "ensure_auth_sessions_schema"):
            self.assertEqual(s.revoke_user_sessions(" Member@Example.com "), 3)
        self.assertEqual(cur.sql, [("DELETE FROM auth_sessions WHERE lower(username) = %s;", ("member@example.com",))])
        self.assertTrue(conn.committed)

    def test_revoke_without_database_or_user_is_a_no_op(self):
        from ui import auth_sessions as s

        with mock.patch.object(s, "get_neon_conn", return_value=None):
            self.assertIsNone(s.revoke_user_sessions("member@example.com"))
        self.assertIsNone(s.revoke_user_sessions(""))


class PasswordUpdateResultTests(unittest.TestCase):
    def _update(self, cursor=None, conn=True):
        from db import users

        target = _Conn(cursor) if conn else None
        with mock.patch.object(users, "get_neon_conn", return_value=target):
            return users.update_neon_user_password("member@example.com", "$2b$hash")

    def test_true_only_when_a_row_changed(self):
        self.assertTrue(self._update(_Cursor(rowcount=1)))
        self.assertFalse(self._update(_Cursor(rowcount=0)))

    def test_false_on_database_error_or_no_connection(self):
        self.assertFalse(self._update(_Cursor(raises=RuntimeError("down"))))
        self.assertFalse(self._update(conn=False))


class ResetPageTests(unittest.TestCase):
    SRC = (ROOT / "pages" / "reset_password.py").read_text()

    def test_reset_checks_the_write_then_revokes_sessions_before_success(self):
        src = self.SRC
        write = src.index("if not update_neon_user_password(username, hashed):")
        revoke = src.index("revoke_user_sessions(username)")
        success = src.index('st.success("Password updated!')
        self.assertLess(write, revoke)
        self.assertLess(revoke, success)

    def test_reset_signs_out_this_tab(self):
        self.assertIn('st.session_state.pop("username", None)', self.SRC)


if __name__ == "__main__":
    unittest.main()
