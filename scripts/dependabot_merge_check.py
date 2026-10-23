"""Decide whether a Dependabot PR may merge itself (.github/workflows/dependabot-automerge.yml).

    python scripts/dependabot_merge_check.py "<PR title>" "<base branch>"

Prints "merge" or "hold: <reason>". Merges only single-package "Bump X from A to B"
PRs into dev where B keeps A's major version (for 0.x, also its minor), and X isn't in
DEPENDABOT_HOLD (comma-separated package names). Standard library only.
"""
from __future__ import annotations

import os
import re
import sys

TITLE = re.compile(r"^(?:\[[^\]]*\]\s*)?Bump (?P<pkg>\S+) from (?P<a>v?[\d][\w.\-+]*) to (?P<b>v?[\d][\w.\-+]*)")


def _parts(v: str) -> list:
    return [int(x) for x in re.findall(r"\d+", v.lstrip("v"))[:3]] + [0, 0, 0]


def decide(title: str, base: str, hold: str = "") -> str:
    if base != "dev":
        return f"hold: targets {base}, not dev"
    m = TITLE.match(title or "")
    if not m:
        return "hold: not a single-package version bump"
    pkg = m.group("pkg")
    held = {h.strip().lower() for h in hold.split(",") if h.strip()}
    if pkg.lower() in held:
        return f"hold: {pkg} is on the hold list"
    a, b = _parts(m.group("a")), _parts(m.group("b"))
    if a[0] != b[0]:
        return f"hold: major update {m.group('a')} -> {m.group('b')}"
    if a[0] == 0 and a[1] != b[1]:
        return f"hold: 0.x minor update {m.group('a')} -> {m.group('b')}"
    return "merge"


if __name__ == "__main__":
    print(decide(sys.argv[1], sys.argv[2], os.environ.get("DEPENDABOT_HOLD", "")))
