import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CleanupSourceChecks(unittest.TestCase):
    def test_price_fetcher_logs_provider_summary(self):
        source = (ROOT / "data" / "prices.py").read_text()

        self.assertIn("summarize_provider_skips", source)
        self.assertIn("[prices] {summary.message}", source)

    def test_scan_engine_surfaces_provider_diagnostics(self):
        source = (ROOT / "scan" / "engine.py").read_text()

        self.assertIn("provider_skipped", source)
        self.assertIn("Provider skip sample", source)
        self.assertIn("Price provider warning", source)

    def test_ci_has_dependency_import_smoke_job(self):
        source = (ROOT / ".github" / "workflows" / "smoke.yml").read_text()

        self.assertIn("core-dependency-import-smoke", source)
        self.assertIn("full-dependency-import-smoke", source)
        self.assertIn("python -m pip install --prefer-binary -r requirements-core.txt", source)
        self.assertIn("Run dependency-backed unit tests", source)
        self.assertIn("python -m pip install --prefer-binary -r requirements.txt", source)
        self.assertIn("import scan.engine", source)

    def test_single_ticker_tools_are_extracted_from_scan_ui(self):
        scans_source = (ROOT / "ui" / "scans.py").read_text()
        single_source = (ROOT / "ui" / "single_ticker.py").read_text()

        self.assertIn("render_single_ticker_panel", scans_source)
        self.assertIn("handle_single_ticker_actions", scans_source)
        self.assertNotIn("def _get_live_quote", scans_source)
        self.assertNotIn("def _render_single_symbol_chart", scans_source)
        self.assertIn("def get_live_quote", single_source)
        self.assertIn("def render_single_symbol_chart", single_source)

    def test_watchlist_scan_tools_are_extracted_from_scan_ui(self):
        scans_source = (ROOT / "ui" / "scans.py").read_text()
        watchlist_source = (ROOT / "ui" / "watchlists.py").read_text()

        self.assertIn("render_active_watchlist_tools", scans_source)
        self.assertIn("handle_active_watchlist_actions", scans_source)
        self.assertNotIn("def build_watchlist_df", scans_source)
        self.assertIn("def build_watchlist_df", watchlist_source)
        self.assertIn("def render_active_watchlist_tools", watchlist_source)
        self.assertIn("def handle_active_watchlist_actions", watchlist_source)


if __name__ == "__main__":
    unittest.main()
