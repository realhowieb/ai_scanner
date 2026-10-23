"""Wait until hsf-api-v1 and hsf-web-v1 serve a given commit (post-promote live check).

Both services report the commit they were deployed from: the API in /readyz `build`, the
web server in /api/healthz `build` (Render's RENDER_GIT_COMMIT, first 12 characters).

Render only rebuilds the web service when something under web/ changed, so the web side
also counts as ready when its build differs from the target but `git diff` shows no web/
change between the two (needs the full git history checked out).

    python scripts/wait_for_deploy.py --sha <commit> [--timeout-min 30]

Exit 0 when both are ready, 1 on timeout. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from typing import Callable, Optional

API = os.environ.get("API_BASE_URL", "https://hsf-api-v1.onrender.com").rstrip("/")
WEB = os.environ.get("WEB_BASE_URL", "https://hsf-web-v1.onrender.com").rstrip("/")


def read_build(url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return (json.loads(r.read().decode() or "{}").get("build") or None)
    except Exception:
        return None


def same_commit(build: Optional[str], sha: str) -> bool:
    return bool(build) and sha.startswith(build)


def web_unchanged(build: str, sha: str) -> bool:
    """True when no file under web/ changed between the served build and the target."""
    try:
        r = subprocess.run(["git", "diff", "--quiet", build, sha, "--", "web/"], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def ready(sha: str, api_build: Optional[str], web_build: Optional[str],
          unchanged: Callable[[str, str], bool] = web_unchanged) -> tuple:
    api_ok = same_commit(api_build, sha)
    web_ok = same_commit(web_build, sha) or (bool(web_build) and unchanged(web_build, sha))
    return api_ok, web_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True)
    ap.add_argument("--timeout-min", type=float, default=30.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    a = ap.parse_args(argv)
    deadline = time.time() + a.timeout_min * 60
    while True:
        api_b = read_build(f"{API}/readyz")
        web_b = read_build(f"{WEB}/api/healthz")
        api_ok, web_ok = ready(a.sha, api_b, web_b)
        print(f"api={api_b} ({'ready' if api_ok else 'waiting'})  web={web_b} ({'ready' if web_ok else 'waiting'})",
              flush=True)
        if api_ok and web_ok:
            # Render switches traffic as soon as the new instance is healthy; give warmers a moment.
            time.sleep(60)
            return 0
        if time.time() >= deadline:
            print(f"::error::deploy of {a.sha[:12]} not live after {a.timeout_min:g} min "
                  f"(api={api_b}, web={web_b})")
            return 1
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
