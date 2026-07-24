"""Market-data health banner: correct branch per (configured, working, admin)."""
from __future__ import annotations

import unittest
from unittest import mock


class _FakeSt:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def error(self, msg, *a, **k):
        self.errors.append(str(msg))


class DataHealthBannerTests(unittest.TestCase):
    def _render(self, probe, *, is_admin):
        import ui.data_health as dh

        fake = _FakeSt()
        with mock.patch.object(dh, "st", fake), \
             mock.patch.object(dh, "_probe_cached", return_value=probe):
            dh.render_data_health_banner(is_admin=is_admin)
        return fake

    def test_healthy_shows_nothing(self):
        fake = self._render((True, True), is_admin=True)
        self.assertEqual(fake.warnings, [])
        self.assertEqual(fake.errors, [])

    def test_rejected_key_admin_gets_actionable_error(self):
        fake = self._render((True, False), is_admin=True)
        self.assertEqual(fake.warnings, [])
        self.assertTrue(any("rejected" in e for e in fake.errors))

    def test_not_configured_admin_gets_config_error(self):
        fake = self._render((False, False), is_admin=True)
        self.assertTrue(any("not configured" in e for e in fake.errors))

    def test_broken_non_admin_gets_soft_warning_only(self):
        fake = self._render((True, False), is_admin=False)
        self.assertEqual(fake.errors, [])
        self.assertTrue(any("temporarily unavailable" in w for w in fake.warnings))


if __name__ == "__main__":
    unittest.main()
