"""P1-31 — the billing service's dependencies are pinned, audited and in step.

Render builds billing_service/ from its own requirements.txt, which CI's audit
never covered; the 2026-09-27 build had starlette and python-dotenv advisories.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "billing_service" / "requirements.txt"


def _pins(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-r"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^\s;]+)$", line)
        if m is None:
            raise AssertionError(f"{path.name}: not an exact pin: {line!r}")
        out[m.group(1).lower().replace("_", "-")] = m.group(3)
    return out


class BillingDepsTests(unittest.TestCase):
    def test_every_line_is_an_exact_pin(self):
        pins = _pins(REQ)
        for direct in ("fastapi", "starlette", "uvicorn", "stripe", "httpx", "psycopg2-binary"):
            self.assertIn(direct, pins)

    def test_patched_versions(self):
        pins = _pins(REQ)
        ver = lambda v: tuple(int(x) for x in re.findall(r"\d+", v)[:3])  # noqa: E731
        self.assertGreaterEqual(ver(pins["starlette"]), (1, 3, 1))      # PYSEC-2026-161/248/249/1941/1942/2280/2281
        self.assertGreaterEqual(ver(pins["python-dotenv"]), (1, 2, 2))  # PYSEC-2026-2270 (transitive via uvicorn)

    def test_ci_billing_test_env_uses_the_same_fastapi(self):
        test_pins = _pins_loose(ROOT / "requirements-billing-test.txt")
        self.assertEqual(test_pins["fastapi"], _pins(REQ)["fastapi"])

    def test_service_does_not_use_dotenv(self):
        for py in (ROOT / "billing_service").glob("*.py"):
            self.assertNotRegex(py.read_text(), r"\bdotenv\b", py.name)

    def test_python_version_pinned_for_render(self):
        self.assertEqual((ROOT / "billing_service" / ".python-version").read_text().strip(), "3.13")

    def test_ci_audits_the_billing_file(self):
        wf = (ROOT / ".github" / "workflows" / "smoke.yml").read_text()
        self.assertIn("pip-audit -r billing_service/requirements.txt --strict", wf)


def _pins_loose(path):
    """Pins from a file that may also contain -r includes and unpinned tools."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^\s;]+)$", line)
        if m:
            out[m.group(1).lower()] = m.group(3)
    return out


if __name__ == "__main__":
    unittest.main()
