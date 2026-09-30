"""P1-52: one password rule for sign-up, password reset and Admin create-user."""
import unittest
from pathlib import Path
from unittest import mock

from ui.password_policy import MIN_LENGTH, password_problem

ROOT = Path(__file__).resolve().parents[1]


class RuleTests(unittest.TestCase):
    def test_rejects(self):
        for pw in ("short1!", "qwerty12345", "1234567890", "asdfghjkl123", "Password123", "aaaaaaaaaaaa",
                   "ababababab12", "qwertyuiop", "zxcvbnm12345", "1q2w3e4r5t6y", "abcdefghij"):
            self.assertIsNotNone(password_problem(pw), pw)

    def test_accepts(self):
        for pw in ("correct horse battery", "Maple-river-42", "sunny tulip garden", "Tr4d3r!Harbor", "vivid oat 9 plum"):
            self.assertIsNone(password_problem(pw), pw)

    def test_length_message(self):
        self.assertEqual(password_problem("abc"), f"Password must be at least {MIN_LENGTH} characters.")

    def test_email_and_username(self):
        self.assertIn("email", password_problem("maria-likes-stocks", email="maria@example.com"))
        self.assertIn("email", password_problem("Stock-Hunter-2026", username="hunter"))
        self.assertIsNone(password_problem("Stock-Hunter-2026", email="bob@example.com"))
        self.assertIsNone(password_problem("the-abc-garden", username="abc"))    # < 4 chars isn't checked


class WiringTests(unittest.TestCase):
    def test_signup_uses_the_rule(self):
        src = (ROOT / "ui" / "auth.py").read_text()
        self.assertIn("password_problem(p1, email=email_raw, username=username_raw)", src)
        self.assertNotIn("len(p1) < 8", src)

    def test_reset_checks_before_the_link_is_used(self):
        src = (ROOT / "pages" / "reset_password.py").read_text()
        self.assertLess(src.index("password_problem(new_pw, email=account)"), src.index("consume_reset_token(token)"))
        self.assertNotIn("_PW_MIN_LEN", src)

    def test_admin_create_uses_the_rule(self):
        from db import user_admin

        ok, msg = user_admin.create_user("new@example.com", "New User", "qwerty12345")
        self.assertFalse(ok)
        self.assertIn("keyboard", msg)

    def test_peek_does_not_consume(self):
        from db import password_reset as pr

        cur = mock.MagicMock()
        cur.fetchone.return_value = ("a@example.com",)
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        with mock.patch.object(pr, "get_neon_conn", return_value=conn), mock.patch.object(pr, "_ensure_schema"):
            self.assertEqual(pr.peek_reset_token("tok"), "a@example.com")
        sql = " ".join(str(c.args[0]) for c in cur.execute.call_args_list)
        self.assertNotIn("UPDATE", sql)
        conn.commit.assert_not_called()


class LandingTests(unittest.TestCase):
    def test_hero_clears_the_header_bar(self):
        from ui import landing

        self.assertIn(".hsf-hero{display:flex;align-items:center;gap:18px;margin:2.75rem 0 6px}", landing.hero_html())


if __name__ == "__main__":
    unittest.main()
