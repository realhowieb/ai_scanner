"""ML v4 lifecycle rules (pure) and the register script's file checks."""
import json
import unittest

from analytics import ml_v4_lifecycle as L

OK = dict(actor="howard", reason="test")


class LifecycleTest(unittest.TestCase):
    def test_happy_path_needs_evidence_and_approval(self):
        L.check_transition("candidate", "validated", evidence={"evaluation": {"auc": 0.6}}, **OK)
        L.check_transition("validated", "shadow", **OK)
        L.check_transition("shadow", "champion", evidence={"promotion_gates_passed": True}, approved=True, **OK)
        L.check_transition("champion", "retired", **OK)

    def test_skipping_stages_is_refused(self):
        for frm, to in [("candidate", "shadow"), ("candidate", "champion"), ("validated", "champion"),
                        ("champion", "shadow"), ("retired", "candidate"), ("shadow", "validated")]:
            with self.subTest(frm=frm, to=to), self.assertRaises(L.LifecycleError):
                L.check_transition(frm, to, evidence={"evaluation": 1, "promotion_gates_passed": True},
                                   approved=True, was_champion_before=True, **OK)

    def test_any_stage_can_be_retired(self):
        for status in L.STATUSES[:-1]:
            with self.subTest(status=status):
                L.check_transition(status, "retired", **OK)

    def test_actor_and_reason_required(self):
        with self.assertRaisesRegex(L.LifecycleError, "admin"):
            L.check_transition("validated", "shadow", actor=" ", reason="x")
        with self.assertRaisesRegex(L.LifecycleError, "reason"):
            L.check_transition("validated", "shadow", actor="howard", reason=None)

    def test_unknown_status(self):
        with self.assertRaisesRegex(L.LifecycleError, "unknown"):
            L.check_transition("candidate", "live", **OK)

    def test_validated_needs_evaluation(self):
        with self.assertRaisesRegex(L.LifecycleError, "evaluation"):
            L.check_transition("candidate", "validated", **OK)

    def test_champion_needs_approval_and_gates(self):
        with self.assertRaisesRegex(L.LifecycleError, "approval"):
            L.check_transition("shadow", "champion", evidence={"promotion_gates_passed": True}, **OK)
        with self.assertRaisesRegex(L.LifecycleError, "promotion-gate"):
            L.check_transition("shadow", "champion", approved=True, **OK)

    def test_retired_returns_to_champion_only_as_rollback(self):
        with self.assertRaisesRegex(L.LifecycleError, "rollback"):
            L.check_transition("retired", "champion", approved=True, **OK)
        L.check_transition("retired", "champion", approved=True, was_champion_before=True, **OK)

    def test_rollback_target_picks_latest_retired_former_champion(self):
        hist = [
            {"model_id": 1, "to_status": "champion", "created_at": "2026-11-01T10:00", "status": "retired"},
            {"model_id": 2, "to_status": "champion", "created_at": "2026-11-02T10:00", "status": "retired"},
            {"model_id": 3, "to_status": "champion", "created_at": "2026-11-03T10:00", "status": "champion"},
            {"model_id": 4, "to_status": "shadow", "created_at": "2026-11-03T11:00", "status": "retired"},
        ]
        self.assertEqual(L.rollback_target(hist, current_champion_id=3), 2)
        self.assertEqual(L.rollback_target(hist[:1], current_champion_id=None), 1)
        self.assertIsNone(L.rollback_target(hist[2:], current_champion_id=3))


class RegisterScriptTest(unittest.TestCase):
    def _write(self, folder, version="ml-v4-xgboost-production-f5d-abc", blob=b"model-bytes", sha=None):
        import hashlib

        (folder / f"{version}.joblib").write_bytes(blob)
        man = {"model_version": version, "artifact_file": f"{version}.joblib", "provider": "xgboost",
               "artifact_sha256": sha or hashlib.sha256(blob).hexdigest(), "artifact_bytes": len(blob),
               "target": {"label": "F_clean_path_5d", "horizon_days": 5}}
        (folder / f"{version}.json").write_text(json.dumps(man))
        return man

    def test_dry_run_and_checksum(self):
        import sys
        import tempfile
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import ml_v4_register as S

        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            self._write(folder)
            (folder / S.SUMMARY_FILE).write_text(json.dumps({"models": []}))
            self.assertEqual([m["model_version"] for m in S.load_manifests(folder)],
                             ["ml-v4-xgboost-production-f5d-abc"])
            self.assertEqual(S.main(["--dir", d, "--dry-run"]), 0)
            with self.assertRaises(SystemExit):  # a real run must say where the files live
                S.main(["--dir", d, "--actor", "howard"])
            self._write(folder, version="ml-v4-lightgbm-production-f5d-def", sha="0" * 64)
            with self.assertRaisesRegex(SystemExit, "does not match"):
                S.load_manifests(folder)
