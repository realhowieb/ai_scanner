from pathlib import Path
import unittest


class LogoBrandingTests(unittest.TestCase):
    def test_uploaded_logo_asset_exists(self):
        self.assertTrue(Path("assets/hsfinest_logo_512.png").exists())

    def test_logo_candidate_prefers_uploaded_asset(self):
        source = Path("ui/header.py").read_text()
        self.assertIn("def render_page_logo", source)
        self.assertIn("def render_logo_heading", source)
        first_candidate = source.index('"assets/hsfinest_logo_512.png"')
        fallback_candidate = source.index('"assets/hsfailogo_transparent_opt.png"')
        self.assertLess(first_candidate, fallback_candidate)

    def test_results_section_uses_logo_heading(self):
        source = Path("ui/results.py").read_text()
        self.assertIn("from ui.header import render_logo_heading", source)
        self.assertIn('render_logo_heading("Results")', source)

    def test_standalone_pages_render_shared_logo(self):
        for path in (
            "pages/alerts.py",
            "pages/day_trader.py",
            "pages/billing.py",
            "pages/reset_password.py",
            "pages/verify_email.py",
        ):
            with self.subTest(path=path):
                source = Path(path).read_text()
                self.assertIn("render_page_logo", source)


if __name__ == "__main__":
    unittest.main()
