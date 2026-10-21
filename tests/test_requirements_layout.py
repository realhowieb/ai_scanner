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
        names.add(re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].lower())
    return names


class RequirementsLayoutTests(unittest.TestCase):
    def test_full_requirements_delegates_to_groups(self):
        lines = _read_lines("requirements.txt")

        self.assertEqual(
            lines,
            [
                "-c requirements.lock",
                "-r requirements-core.txt",
                "-r requirements-ml.txt",
                "-r requirements-extended.txt",
            ],
        )

    def test_lock_exists_and_pins_native_stack(self):
        """The lock must exist and exact-pin the ABI-sensitive native deps."""
        lock_lines = _read_lines("requirements.lock")
        pinned = {line.split("==", 1)[0].lower() for line in lock_lines if "==" in line}

        for pkg in {"numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "joblib"}:
            self.assertIn(pkg, pinned, f"{pkg} missing from requirements.lock")
        # Constraints files must be plain pins (no includes, no local paths).
        for line in lock_lines:
            self.assertNotIn("-r ", line)
            self.assertNotIn(" @ file:", line)

    def test_cron_workflow_installs_with_lock_constraints(self):
        source = (ROOT / ".github" / "workflows" / "scheduled-scans.yml").read_text()

        self.assertIn("-r requirements-core.txt -c requirements.lock", source)
        self.assertIn("-r requirements-ml.txt -c requirements.lock", source)
        self.assertIn("python-version: '3.13'", source)

    def test_core_requirements_keep_scanner_runtime_deps(self):
        core = _pkg_names("requirements-core.txt")

        # Checked by package name so version-spec changes don't break the test.
        for pkg in {"pandas", "numpy", "yfinance", "requests", "sqlalchemy", "psycopg"}:
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

    def test_streamlit_is_gone(self):
        # The Streamlit app was retired on 2026-10-21; its packages must not creep back in.
        lock = {line.split("==", 1)[0].lower() for line in _read_lines("requirements.lock") if "==" in line}
        for name in ("requirements-core.txt", "requirements-ml.txt", "requirements-extended.txt", "api/requirements.txt"):
            self.assertFalse({p for p in _pkg_names(name) if p.startswith("streamlit")}, name)
        self.assertFalse({p for p in lock if "streamlit" in p or p in ("altair", "plotly", "pillow")})
        self.assertNotIn("streamlit", (ROOT / "pyproject.toml").read_text().lower().split("[project.optional-dependencies]")[0])

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
