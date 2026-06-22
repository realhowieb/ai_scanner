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


def _wait_for_streamlit(port: int, timeout: float) -> None:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    root_url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            health = _read_url(health_url, timeout=2.0).strip().lower()
            home = _read_url(root_url, timeout=2.0).lower()
            if health == "ok" and "streamlit" in home:
                return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(0.5)

    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"Streamlit did not become healthy on port {port}{detail}")


def run_smoke(timeout: float = 45.0) -> None:
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
        _wait_for_streamlit(port, timeout=timeout)
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
    args = parser.parse_args()
    run_smoke(timeout=args.timeout)


if __name__ == "__main__":
    main()
