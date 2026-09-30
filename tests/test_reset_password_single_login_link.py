"""The password-reset page shows one login link, not two (2026-09-30)."""
import unittest
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "pages" / "reset_password.py").read_text()


class ResetPageLinkTests(unittest.TestCase):
    def test_one_login_link(self):
        self.assertEqual(SRC.count('st.page_link("app.py"'), 1)
        self.assertIn('st.page_link("app.py", label="← Back to login")', SRC)
        self.assertNotIn("Go to login →", SRC)


if __name__ == "__main__":
    unittest.main()
