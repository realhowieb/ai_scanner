import unittest

from scripts.ml_dataset_fingerprint_audit import diagnose


class FingerprintDiagnosisTests(unittest.TestCase):
    def test_membership_and_retained_quality_differences_are_separate(self):
        entry = {"dataset_version": "v1", "created_at": "2026-10-08", "fingerprint": "original",
                 "observation_ids": [1, 2], "metadata": {"code_revision": "old", "data_quality": {"maturity_5d": {"pending": 2}}}}
        current = {"observation_ids": [1, 2], "metadata": {"fingerprint": "now", "data_quality": {"maturity_5d": {"matured": 2}}}}
        result = diagnose(entry, current, current)
        self.assertFalse(result["original_component_hashes_available"])
        self.assertFalse(result["rebuilds"]["current"]["matches_original"])
        self.assertEqual(result["rebuilds"]["current"]["missing_members"], 0)
        self.assertIn("maturity_5d", result["rebuilds"]["current"]["quality_changes"])
        self.assertIn("do not identify every changed cell", result["limits"])

    def test_reports_missing_members_and_component_change_when_evidence_exists(self):
        entry = {"dataset_version": "v1", "created_at": "today", "fingerprint": "x", "observation_ids": [1, 2],
                 "metadata": {"component_fingerprints": {"features": "old"}}}
        built = {"observation_ids": [2, 3], "metadata": {"fingerprint": "new", "component_fingerprints": {"features": "new"}}}
        part = diagnose(entry, built, built)["rebuilds"]["as_of_freeze"]
        self.assertEqual((part["missing_members"], part["unexpected_members"]), (1, 1))
        self.assertEqual(part["changed_components"], ["features"])
