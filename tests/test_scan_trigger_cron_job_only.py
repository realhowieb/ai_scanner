"""Scheduled scans are started only by cron-job.org (workflow_dispatch on main).

GitHub's own `schedule:` trigger was removed on 2026-09-29: it fired hours late
and duplicated cron-job.org's slots (the 21:10 UTC slot ran at 00:27 UTC and
sent the morning digest at 8:27 PM ET).
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "scheduled-scans.yml"


class ScanTriggerTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML not installed")
    def test_only_workflow_dispatch_triggers_the_scan(self):
        import yaml

        triggers = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))[True]   # YAML reads `on:` as True
        self.assertEqual(list(triggers), ["workflow_dispatch"])
        inputs = triggers["workflow_dispatch"]["inputs"]
        self.assertIn("force", inputs)        # cron-job.org passes force=false
        self.assertIn("session", inputs)      # and session=auto

    def test_no_cron_lines_in_the_workflow(self):
        lines = [ln.strip() for ln in WORKFLOW.read_text(encoding="utf-8").splitlines()]
        self.assertFalse([ln for ln in lines if ln.startswith("- cron:") or ln == "schedule:"])


if __name__ == "__main__":
    unittest.main()
