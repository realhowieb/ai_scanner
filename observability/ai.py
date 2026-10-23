"""Metadata-only PostHog AI events. No prompts, responses, account IDs or raw errors.

Uses the public capture API and existing requests dependency. One lazy daemon worker
and a bounded queue keep network latency/failure out of the model request path.
Telemetry is lossy on overload/process shutdown; it is not usage or billing data.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import queue
import re
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any

_HOSTS = {"https://us.i.posthog.com", "https://eu.i.posthog.com"}
_FEATURES = {"scan_summary", "ticker_deepdive", "results_chat", "market_brief_narrative", "watchlist_note", "nl_screener",
             "ask_hsf", "journal_coach", "scan_builder", "alert_why"}
_queue: queue.Queue = queue.Queue(maxsize=64)
_lock = threading.Lock()
_worker: threading.Thread | None = None


def configuration() -> tuple[str, str] | None:
    key = os.getenv("POSTHOG_PROJECT_KEY", "").strip()
    host = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com").strip().rstrip("/")
    if os.getenv("POSTHOG_AI_ENABLED", "0") != "1" or not key or host not in _HOSTS:
        return None
    return key, host


def _deliver(item: tuple[str, str, dict]) -> None:
    import requests

    key, host, event = item
    # Never follow redirects with credentials. No retries or raw payload logging.
    with requests.post(host + "/i/v0/e/", json={"api_key": key, **event},
                       timeout=(2, 3), allow_redirects=False) as response:
        response.raise_for_status()


def _run() -> None:
    while True:
        item = _queue.get()
        try:
            _deliver(item)
        except Exception:
            pass
        finally:
            _queue.task_done()


def _enqueue(config: tuple[str, str], event: dict) -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="hsf-ai-telemetry", daemon=True)
            _worker.start()
    try:
        _queue.put_nowait((*config, event))
    except queue.Full:
        pass


def _count(value: Any) -> int | None:
    return value if type(value) is int and 0 <= value <= 1_000_000_000 else None


class Generation:
    def __init__(self, feature: str | None, model: str):
        self.feature, self.model = feature, model
        self.started = time.monotonic()
        self.response = None

    def finish(self, error: BaseException | None) -> None:
        config = configuration()
        if config is None:
            return
        elapsed = time.monotonic() - self.started
        props: dict[str, Any] = {
            "$ai_provider": "anthropic", "$ai_trace_id": str(uuid.uuid4()),
            "$ai_span_id": str(uuid.uuid4()), "$ai_is_error": error is not None,
            "$process_person_profile": False,
            "feature": self.feature if self.feature in _FEATURES else "other",
        }
        if math.isfinite(elapsed):
            props["$ai_latency"] = max(0, elapsed)
        # Models are operational config, restricted to Anthropic model identifiers.
        if re.fullmatch(r"claude-[a-zA-Z0-9._-]{1,100}", self.model):
            props["$ai_model"] = self.model
        if error is not None:
            name = type(error).__name__
            props["$ai_error"] = name if name in {"AuthenticationError", "RateLimitError", "APITimeoutError", "APIError", "APIConnectionError", "InternalServerError"} else "ProviderError"
        usage = getattr(self.response, "usage", None)
        for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = _count(getattr(usage, field, None))
            if value is not None:
                props["$ai_" + field] = value
        _enqueue(config, {"event": "$ai_generation", "distinct_id": str(uuid.uuid4()),
                          "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "properties": props})


@contextmanager
def generation(feature: str | None, model: str):
    """Observes a provider attempt without changing results, exceptions or limits."""
    span = Generation(feature, model)
    error = None
    try:
        yield span
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            span.finish(error)
        except Exception:
            pass
