"""Run 61 — HSF Autonomous Research Mode v1 certification suite (gates A–X).

Integrated behaviour only; component behaviour stays in the Run 59/60 suites.
Runs locally and in CI with fixtures/mocks — no production mutation.
"""
import json
import unittest
from pathlib import Path

from analytics import autonomy_certification as ac
from analytics import forward_readiness as fr

ROOT = Path(__file__).resolve().parents[1]
_CACHE = {}


def gates():
    if "r" not in _CACHE:
        health = json.loads((ROOT / "artifacts" / "health" / "system_health.json").read_text())
        plan = json.loads((ROOT / "artifacts" / "health" / "recovery_plan.json").read_text())
        _CACHE["r"] = ac.run_all(current_health=health, current_plan=plan,
                                 extra_artifacts=sorted((ROOT / "artifacts" / "health").glob("*.json")))
    return _CACHE["r"]


def _make(letter):
    def test(self):
        g = gates()["gates"][letter]
        failing = {k: v for k, v in (g["evidence"].get("checks") or {}).items() if not v}
        self.assertEqual(g["status"], "PASS", f"Gate {letter} ({g['title']}) failed: {failing or g['evidence']}")
        self.assertTrue(g["mandatory"])
    test.__name__ = f"test_gate_{letter}"
    return test


class CertificationGateTests(unittest.TestCase):
    pass


for _letter in ac.MANDATORY_GATES:
    setattr(CertificationGateTests, f"test_gate_{_letter}", _make(_letter))


class VerdictTests(unittest.TestCase):
    def test_all_24_gates_evaluated(self):
        r = gates()
        self.assertEqual(set(ac.MANDATORY_GATES) - set(r["gates"]), set())
        self.assertEqual(r["mandatory_gates"], 24)

    def test_verdict_is_one_of_two_values(self):
        self.assertIn(gates()["verdict"], (ac.CERTIFIED, ac.NOT_CERTIFIED))

    def test_verdict_logic_is_strict(self):
        passing = {k: {"status": "PASS", "mandatory": True} for k in ac.MANDATORY_GATES}
        self.assertEqual(ac.verdict_from(passing), ac.CERTIFIED)
        for k in ac.MANDATORY_GATES:
            one_fail = dict(passing, **{k: {"status": "FAIL", "mandatory": True}})
            self.assertEqual(ac.verdict_from(one_fail), ac.NOT_CERTIFIED, k)
        missing = dict(passing)
        missing.pop("X")
        self.assertEqual(ac.verdict_from(missing), ac.NOT_CERTIFIED)
        demoted = dict(passing, A={"status": "PASS", "mandatory": False})
        self.assertEqual(ac.verdict_from(demoted), ac.NOT_CERTIFIED)
        warned = dict(passing, INFO_1={"status": "WARN", "mandatory": False})
        self.assertEqual(ac.verdict_from(warned), ac.CERTIFIED)   # informational never blocks

    def test_autonomy_level_follows_verdict(self):
        r = gates()
        self.assertEqual(r["autonomy_level"], "AUTONOMOUS" if r["verdict"] == ac.CERTIFIED else "RECOVERY_READY")

    def test_certification_is_deterministic(self):
        a = ac.gate_x_thirty_days()
        b = ac.gate_x_thirty_days()
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def test_epoch_untouched_by_certification(self):
        gates()
        self.assertEqual(fr.FORWARD_EPOCH["forward_epoch_start_timestamp"], "2026-09-26T07:23:11+00:00")

    def test_certification_never_enables_recovery(self):
        wf = (ROOT / ".github" / "workflows" / "autonomy-certification.yml").read_text()
        self.assertNotIn("schedule:", wf)
        self.assertNotIn("recover", wf.replace("autonomous-recovery", "").lower().split("hsf_autonomy_mode")[-1][:40])
        src = (ROOT / "scripts" / "autonomy_certification.py").read_text()
        for bad in ("append_event(", "save_snapshot(", "gh variable", "/actions/variables"):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()
