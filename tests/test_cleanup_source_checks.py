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

        self.assertIn("dependency-import-smoke", source)
        self.assertIn("python -m pip install --prefer-binary -r requirements.txt", source)
        self.assertIn("import scan.engine", source)


if __name__ == "__main__":
    unittest.main()
