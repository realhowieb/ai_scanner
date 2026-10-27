import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.repo_files import python_files


class SharedServicesTests(unittest.TestCase):
    def test_pure_plan_and_password_rules_import_without_ui_runtime(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import domain.plans, domain.password_policy, domain.plan_labels, domain.upgrades; "
                "assert 'streamlit' not in sys.modules; assert 'ui' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_compatibility_imports_share_the_canonical_implementation(self):
        for old, new in [
            ("ui.password_policy", "domain.password_policy"),
            ("ui.plan_labels", "domain.plan_labels"),
            ("ui.checkout", "services.billing"),
            ("ui.auth_tokens", "services.account_tokens"),
        ]:
            self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_repository_inventory_does_not_walk_nested_untracked_copies(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("pass\n")
            nested = root / "ai_scanner"
            nested.mkdir()
            (nested / "untracked.py").write_text("raise RuntimeError\n")
            with patch("tests.repo_files.check_output", return_value=b"app.py\0") as inventory:
                self.assertEqual(python_files(root), [root / "app.py"])
                self.assertEqual(inventory.call_args.args[0][:3], ["git", "ls-files", "-z"])
