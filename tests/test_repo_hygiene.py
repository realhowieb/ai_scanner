"""Repo-wide hygiene guards (run in Smoke Checks and the Security Scan workflow).

- No real personal email address in any tracked file. The repo and its Actions
  logs are public; a real address in a fixture once let anyone unmask the masked
  recipient printed by the scheduled jobs (P2-33).
- Every exact (==) pin is the same wherever the same package is pinned, so a
  security bump in one file can't leave a stale, vulnerable pin in another.
"""
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_EMAIL = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
# Consumer mailbox providers: an address here is almost certainly a real person.
_PERSONAL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "live.com", "msn.com", "icloud.com", "me.com", "mac.com", "aol.com",
    "proton.me", "protonmail.com", "gmx.com", "mail.com", "yandex.com",
}
# Synthetic addresses used as examples on those domains.
_ALLOWED_LOCAL = {"sample.customer"}


def _tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    for rel in out.stdout.splitlines():
        path = ROOT / rel
        if path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pkl", ".joblib", ".json.gz", ".zip"}:
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def _pins(path: Path) -> dict:
    pins = {}
    for line in path.read_text().splitlines():
        m = re.match(r'^\s*"?([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^",;\s]+)', line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


class RepoHygieneTests(unittest.TestCase):
    def test_no_real_personal_email_in_tracked_files(self):
        hits = []
        for rel, text in _tracked_text_files():
            for m in _EMAIL.finditer(text):
                local, domain = m.group(1), m.group(2).lower()
                if domain in _PERSONAL_DOMAINS and local.lower() not in _ALLOWED_LOCAL:
                    hits.append(f"{rel}: {local[:2]}***@{domain}")
        self.assertEqual(hits, [], "real-looking personal addresses in tracked files")

    def test_app_pins_match_the_lock(self):
        lock = _pins(ROOT / "requirements.lock")
        pyproject = (ROOT / "pyproject.toml").read_text().split("[project.optional-dependencies]")[0]
        app = {**_pins(ROOT / "requirements-core.txt")}
        for line in pyproject.splitlines():
            app.update(_pins_from_line(line))
        mismatched = {k: (v, lock.get(k)) for k, v in app.items() if lock.get(k) != v}
        self.assertEqual(mismatched, {}, "pin differs from requirements.lock")

    def test_billing_extra_matches_billing_service_pins(self):
        text = (ROOT / "pyproject.toml").read_text()
        block = re.search(r"(?ms)^billing = \[(.*?)\]", text).group(1)
        extra = {}
        for line in block.splitlines():
            extra.update(_pins_from_line(line))
        service = _pins(ROOT / "billing_service" / "requirements.txt")
        self.assertTrue(extra)
        mismatched = {k: (v, service.get(k)) for k, v in extra.items() if service.get(k) != v}
        self.assertEqual(mismatched, {}, "pyproject billing extra differs from billing_service/requirements.txt")


def _pins_from_line(line: str) -> dict:
    m = re.match(r'^\s*"([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^"]+)"', line)
    return {m.group(1).lower().replace("_", "-"): m.group(2)} if m else {}


if __name__ == "__main__":
    unittest.main()
