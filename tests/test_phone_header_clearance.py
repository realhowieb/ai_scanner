"""Phones: content clears Streamlit's 60px header, and the "»" sidebar button is
hidden (Streamlit 1.54 renamed it stExpandSidebarButton) — owner screenshot 2026-09-30."""
import unittest

from ui.chrome import CHROME_CSS


class PhoneHeaderTests(unittest.TestCase):
    def phone_css(self):
        return CHROME_CSS.split("@media (max-width:640px){", 1)[1]

    def test_content_starts_below_the_header(self):
        self.assertIn("padding-top:4rem !important", self.phone_css())

    def test_new_sidebar_button_name_is_hidden(self):
        css = self.phone_css()
        self.assertIn('[data-testid="stExpandSidebarButton"]', css)
        self.assertIn("display:none !important", css.split('[data-testid="stExpandSidebarButton"]', 1)[1][:40])

    def test_landing_does_not_double_the_gap_on_phones(self):
        from ui import landing

        self.assertIn("@media (max-width:640px){.hsf-hero{margin-top:4px}}", landing.hero_html())


if __name__ == "__main__":
    unittest.main()
