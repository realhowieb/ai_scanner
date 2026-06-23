"""Run a real browser smoke flow against the Streamlit login shell.

This optional check complements scripts/streamlit_smoke.py. It requires
Playwright and a browser install, so CI can keep using the dependency-light
server smoke while local/release validation can exercise the browser runtime.
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
LOGIN_MARKERS = ("AI Scanner", "Login", "Username", "Password", "Streamlit")
ERROR_MARKERS = (
    "Traceback",
    "ModuleNotFoundError",
    "ImportError",
    "StreamlitSecretNotFoundError",
    "App failed during startup",
)
BROWSER_FLOW_ERRORS = (RuntimeError, TimeoutError, OSError)


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, timeout: float) -> None:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urlopen(health_url, timeout=2.0) as response:
                if response.read(64).decode("utf-8", errors="replace").strip().lower() == "ok":
                    return
        except (OSError, URLError) as exc:
            last_error = exc
        time.sleep(0.5)

    detail = f": {last_error}" if last_error else ""
    raise TimeoutError(f"Streamlit did not become healthy on port {port}{detail}")


def _start_streamlit(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("AI_SCANNER_SMOKE_TEST", "1")
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    return subprocess.Popen(
        [
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
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_app_markers(page, timeout_ms: int) -> str:
    """Wait for Streamlit to render app/login text and return validated page text."""
    page.wait_for_function(
        """
        (markers) => {
            const root = document.querySelector('#root, [data-testid="stAppViewContainer"], .stApp');
            const text = (document.body?.innerText || document.body?.textContent || document.documentElement?.textContent || '');
            const lower = text.toLowerCase();
            return Boolean(root) && markers.some((marker) => lower.includes(marker.toLowerCase()));
        }
        """,
        arg=list(LOGIN_MARKERS),
        timeout=timeout_ms,
    )
    return page.evaluate(
        "() => document.body?.innerText || document.body?.textContent || document.documentElement?.textContent || ''"
    )


def _stop_streamlit(proc: subprocess.Popen[str]) -> str:
    """Stop Streamlit and return a bounded chunk of combined output."""
    output = ""
    proc.terminate()
    try:
        output, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate(timeout=10)
    return output or ""


def run_browser_flow(timeout: float = 60.0, *, headless: bool = True) -> None:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run "
            "`python -m pip install -r requirements-browser.txt` and "
            "`python -m playwright install chromium`."
        ) from exc

    flow_errors = (*BROWSER_FLOW_ERRORS, PlaywrightError)
    port = _free_port()
    proc = _start_streamlit(port)

    try:
        _wait_for_health(port, timeout=timeout)
        url = f"http://127.0.0.1:{port}/"
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=int(timeout * 1000))
            body_text = _wait_for_app_markers(page, timeout_ms=int(timeout * 1000))
            lower_text = body_text.lower()
            for marker in ERROR_MARKERS:
                if marker.lower() in lower_text:
                    raise RuntimeError(f"Browser flow found app error marker: {marker}")
            browser.close()

        if not any(marker.lower() in lower_text for marker in LOGIN_MARKERS):
            raise RuntimeError("Browser flow did not find expected login/app markers.")
    except flow_errors:
        output = _stop_streamlit(proc)
        if output:
            print(output[-4000:], file=sys.stderr)
        raise
    finally:
        if proc.poll() is None:
            _stop_streamlit(proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    run_browser_flow(timeout=args.timeout, headless=not args.headed)


if __name__ == "__main__":
    main()
