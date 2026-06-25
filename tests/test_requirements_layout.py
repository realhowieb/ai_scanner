import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read_lines(name):
    return [
        line.strip()
        for line in (ROOT / name).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


class RequirementsLayoutTests(unittest.TestCase):
    def test_full_requirements_delegates_to_groups(self):
        lines = _read_lines("requirements.txt")

        self.assertEqual(
            lines,
            [
                "-r requirements-core.txt",
                "-r requirements-ml.txt",
                "-r requirements-extended.txt",
            ],
        )

    def test_core_requirements_keep_scanner_runtime_deps(self):
        core = set(_read_lines("requirements-core.txt"))

        for dep in {
            "streamlit==1.49.1",
            "pandas>=2.0",
            "numpy>=1.23",
            "yfinance>=0.2.27",
            "requests",
            "SQLAlchemy>=2.0",
            "apscheduler",
            "psycopg[binary]>=3.2",
        }:
            self.assertIn(dep, core)

    def test_optional_heavy_deps_are_not_in_core(self):
        core = set(_read_lines("requirements-core.txt"))
        optional = set(_read_lines("requirements-ml.txt")) | set(_read_lines("requirements-extended.txt"))

        for dep in {"xgboost", "scikit-learn", "joblib", "alpaca-py"}:
            self.assertNotIn(dep, core)
            self.assertIn(dep, optional)

        for conflicting_dep in {"pyppeteer", "yahoo_fin"}:
            self.assertNotIn(conflicting_dep, core)
            self.assertNotIn(conflicting_dep, optional)

    def test_pyproject_keeps_heavy_deps_optional(self):
        text = (ROOT / "pyproject.toml").read_text()
        dependencies_block = text.split("[project.optional-dependencies]", 1)[0]

        for dep in {"xgboost", "scikit-learn", "joblib", "alpaca-py"}:
            self.assertNotIn(f'"{dep}"', dependencies_block)
            self.assertIn(f'"{dep}"', text)

        for conflicting_dep in {"pyppeteer", "yahoo_fin"}:
            self.assertNotIn(f'"{conflicting_dep}"', text)

    def test_ml_module_keeps_optional_imports_guarded(self):
        text = (ROOT / "ml_prebreakout.py").read_text()

        self.assertIn("joblib = None", text)
        self.assertIn("XGBClassifier = None", text)
        self.assertIn("Install requirements-ml.txt", text)


if __name__ == "__main__":
    unittest.main()
