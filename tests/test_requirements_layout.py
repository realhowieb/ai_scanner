import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _read_lines(name):
    return [
        line.strip()
        for line in (ROOT / name).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _pkg_names(name):
    """Package names only (strip version specifiers/extras) for spec-agnostic checks."""
    import re
    names = set()
    for line in _read_lines(name):
        if line.startswith("-r"):
            continue
        names.add(re.split(r"[<>=!~\[ ]", line, 1)[0].lower())
    return names


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
        core = _pkg_names("requirements-core.txt")

        # Streamlit stays exactly pinned; others checked by package name so
        # version-spec changes (e.g. adding upper bounds) don't break the test.
        self.assertIn("streamlit==1.49.1", set(_read_lines("requirements-core.txt")))
        for pkg in {"pandas", "numpy", "yfinance", "requests", "sqlalchemy", "apscheduler", "psycopg"}:
            self.assertIn(pkg, core)

    def test_optional_heavy_deps_are_not_in_core(self):
        core = _pkg_names("requirements-core.txt")
        optional = _pkg_names("requirements-ml.txt") | _pkg_names("requirements-extended.txt")

        for pkg in {"xgboost", "scikit-learn", "joblib", "alpaca-py"}:
            self.assertNotIn(pkg, core)
            self.assertIn(pkg, optional)

        for conflicting in {"pyppeteer", "yahoo_fin"}:
            self.assertNotIn(conflicting, core)
            self.assertNotIn(conflicting, optional)

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
