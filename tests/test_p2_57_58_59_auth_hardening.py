"""P2-57 one sign-in error; P2-58 session ids stored hashed; P2-59 list_runs scoped."""
import importlib.util
import re
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_STREAMLIT = importlib.util.find_spec("streamlit") is not None


class _FakeSessionsDB:
    """Just enough of auth_sessions (+ users) to run the session helpers."""

    def __init__(self, rows=None, inactive=()):
        self.rows = list(rows or [])  # dicts: session_id, session_hash, username, expired
        self.inactive = set(inactive)

    def connect(self):
        return _FakeConn(self)


class _FakeConn:
    def __init__(self, db):
        self.db = db

    def cursor(self):
        return _FakeCursor(self.db)

    def commit(self):
        pass

    def close(self):
        pass


class _FakeCursor:
    def __init__(self, db):
        self.db, self._result, self.rowcount = db, [], 0

    def execute(self, sql, params=()):
        q = " ".join(sql.split())
        rows = self.db.rows
        if q.startswith(("CREATE", "ALTER")):
            return
        if q.startswith("SELECT session_id::text FROM auth_sessions WHERE session_hash IS NULL"):
            self._result = [(r["session_id"],) for r in rows if r["session_hash"] is None]
        elif q.startswith("UPDATE auth_sessions SET session_hash = %s, session_id = gen_random_uuid()"):
            for r in rows:
                if r["session_id"] == params[1] and r["session_hash"] is None:
                    r["session_hash"], r["session_id"] = params[0], str(uuid.uuid4())
        elif q.startswith("INSERT INTO auth_sessions"):
            rows.append({"session_id": str(uuid.uuid4()), "username": params[0],
                         "session_hash": params[2], "expired": False})
        elif q.startswith("SELECT s.username FROM auth_sessions s JOIN users u"):
            assert "s.session_hash = %s" in q and "u.is_active IS NOT FALSE" in q
            self._result = [(r["username"],) for r in rows
                            if r["session_hash"] == params[0] and not r["expired"]
                            and r["username"] not in self.db.inactive][:1]
        elif q.startswith("UPDATE auth_sessions SET last_seen_at"):
            pass
        elif q.startswith("DELETE FROM auth_sessions WHERE session_hash = %s"):
            before = len(rows)
            self.db.rows = [r for r in rows if r["session_hash"] != params[0]]
            self.rowcount = before - len(self.db.rows)
        else:
            raise AssertionError(f"unexpected SQL: {q}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result

    def close(self):
        pass


@unittest.skipUnless(_STREAMLIT, "ui.auth_sessions imports streamlit")
class HashedSessionTests(unittest.TestCase):
    def _patch(self, db):
        from ui import auth_sessions as s

        return mock.patch.object(s, "get_neon_conn", side_effect=db.connect)

    def test_new_session_stores_only_the_hash(self):
        from ui import auth_sessions as s

        db = _FakeSessionsDB()
        with self._patch(db):
            token = s.create_session("Member@Example.com")
            self.assertTrue(token and len(token) >= 40)  # 256-bit token_urlsafe
            stored = db.rows[0]
            self.assertEqual(stored["session_hash"], s.session_hash(token))
            self.assertNotIn(token, (stored["session_id"], stored["session_hash"]))
            self.assertEqual(s.get_username_for_session(token), "member@example.com")
            self.assertIsNone(s.get_username_for_session(stored["session_hash"]))  # a leaked hash is useless

    def test_existing_sessions_are_migrated_and_their_cookies_keep_working(self):
        from ui import auth_sessions as s

        legacy = str(uuid.uuid4())  # what the browser holds from before P2-58
        db = _FakeSessionsDB(rows=[{"session_id": legacy, "session_hash": None,
                                    "username": "member@example.com", "expired": False}])
        with self._patch(db):
            self.assertEqual(s.get_username_for_session(legacy), "member@example.com")
        row = db.rows[0]
        self.assertEqual(row["session_hash"], s.session_hash(legacy))
        self.assertNotEqual(row["session_id"], legacy)  # the raw id is no longer stored

    def test_logout_deletes_by_hash_and_deactivated_accounts_get_nothing(self):
        from ui import auth_sessions as s

        db = _FakeSessionsDB(inactive={"gone@example.com"})
        with self._patch(db), mock.patch.object(s, "st") as fake_st:
            fake_st.session_state = {}
            keep = s.create_session("member@example.com")
            gone = s.create_session("gone@example.com")
            self.assertIsNone(s.get_username_for_session(gone))
            s.delete_session(keep)
            self.assertIsNone(s.get_username_for_session(keep))
        self.assertEqual([r["username"] for r in db.rows], ["gone@example.com"])


class ListRunsScopeTests(unittest.TestCase):
    def test_no_username_and_no_all_users_returns_nothing_without_a_query(self):
        from db import runs

        with mock.patch.object(runs, "get_neon_conn") as conn:
            self.assertEqual(runs.list_runs(limit=5), [])
            self.assertEqual(runs.list_runs(limit=5, username=None), [])
        conn.assert_not_called()

    def test_every_caller_scopes_its_runs(self):
        unscoped = []
        for path in ROOT.rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(("tests/", ".venv/")) or rel == "db/runs.py":
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                code = line.strip()
                if code.startswith(("def ", "-", "#")):  # definitions, docstrings, comments
                    continue
                if re.search(r"\blist_runs\(\s*[^)\s]", code) and not re.search(r"username=|all_users=True", code):
                    unscoped.append(f"{rel}:{n}: {line.strip()}")
        self.assertEqual(unscoped, [])

    def test_session_runs_come_only_from_the_scheduler(self):
        for rel in ("ui/day_trader.py", "ui/after_close.py"):
            self.assertIn('include_snapshots=False, username="scheduler")', (ROOT / rel).read_text(), rel)


class SignInMessageTests(unittest.TestCase):
    def test_unknown_email_and_wrong_password_look_the_same(self):
        src = (ROOT / "ui" / "auth.py").read_text()
        self.assertNotIn("User not found", src)
        self.assertNotIn("Incorrect password", src)
        self.assertNotIn("missing a password field", src)
        for reason in ("user_not_found", "no_password_field", "wrong_password"):
            self.assertIn(f'_fail("Email or password is incorrect.", reason="{reason}")', src)


if __name__ == "__main__":
    unittest.main()
