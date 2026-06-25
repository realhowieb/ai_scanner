"""Start the Streamlit app and verify the local server responds.

This is intentionally small and dependency-light so CI can catch startup
regressions without relying on live market data, external APIs, or a database.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_url(url: str, timeout: float) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read(4096).decode("utf-8", errors="replace")


def _validate_browser_shell(html: str) -> None:
    """Validate that the root page contains a usable Streamlit browser shell."""
    lowered = html.lower()
    required_markers = ("streamlit", "<script", "static/js")
    missing = [marker for marker in required_markers if marker not in lowered]
    if missing:
        raise RuntimeError(f"Streamlit browser shell missing markers: {', '.join(missing)}")


def _wait_for_streamlit(port: int, timeout: float, *, browser_shell_check: bool = True) -> None:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    root_url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            health = _read_url(health_url, timeout=2.0).strip().lower()
            home = _read_url(root_url, timeout=2.0)
            if health == "ok" and "streamlit" in home.lower():
                if browser_shell_check:
                    _validate_browser_shell(home)
                return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(0.5)

    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"Streamlit did not become healthy on port {port}{detail}")


def run_smoke(timeout: float = 45.0, *, browser_shell_check: bool = True) -> None:
    port = _free_port()
    env = os.environ.copy()
    env.setdefault("AI_SCANNER_SMOKE_TEST", "1")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_streamlit(port, timeout=timeout, browser_shell_check=browser_shell_check)
    except Exception:
        output = ""
        if proc.stdout is not None:
            with contextlib.suppress(Exception):
                output = proc.stdout.read()
        if output:
            print(output[-4000:], file=sys.stderr)
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--no-browser-shell-check",
        action="store_true",
        help="only check Streamlit health/root response, not browser shell markers",
    )
    args = parser.parse_args()
    run_smoke(timeout=args.timeout, browser_shell_check=not args.no_browser_shell_check)


if __name__ == "__main__":
    main()
