"""No telemetry in these tests leaves the process."""
import json
import queue
from types import SimpleNamespace
from unittest import mock

import pytest

from observability import ai


@pytest.fixture(autouse=True)
def disabled(monkeypatch):
    monkeypatch.delenv("POSTHOG_AI_ENABLED", raising=False)
    monkeypatch.delenv("POSTHOG_PROJECT_KEY", raising=False)


def configured(monkeypatch):
    monkeypatch.setenv("POSTHOG_AI_ENABLED", "1")
    monkeypatch.setenv("POSTHOG_PROJECT_KEY", "fixture-only")
    monkeypatch.setenv("POSTHOG_HOST", "https://eu.i.posthog.com")


def test_disabled_or_invalid_host_never_starts_export(monkeypatch):
    with mock.patch.object(ai, "_enqueue") as enqueue:
        with ai.generation("scan_summary", "claude-fixture"):
            pass
        configured(monkeypatch)
        monkeypatch.setenv("POSTHOG_HOST", "https://untrusted.example")
        with ai.generation("scan_summary", "claude-fixture"):
            pass
    enqueue.assert_not_called()


def test_metadata_only_success_and_cache_usage(monkeypatch):
    configured(monkeypatch)
    response = SimpleNamespace(content="PRIVATE response", usage=SimpleNamespace(
        input_tokens=12, output_tokens=8, cache_read_input_tokens=7, cache_creation_input_tokens=3))
    with mock.patch.object(ai, "_enqueue") as enqueue:
        with ai.generation("watchlist_note", "claude-fixture") as span:
            span.response = response
    config, event = enqueue.call_args.args
    assert config == ("fixture-only", "https://eu.i.posthog.com")
    props = event["properties"]
    assert event["event"] == "$ai_generation"
    assert props["$ai_input_tokens"] == 12
    assert props["$ai_cache_read_input_tokens"] == 7
    assert props["$ai_cache_creation_input_tokens"] == 3
    assert props["$ai_is_error"] is False
    assert props["$process_person_profile"] is False
    assert props["$ai_latency"] >= 0
    assert "PRIVATE" not in json.dumps(event)
    assert "$ai_input" not in props and "$ai_output_choices" not in props


def test_errors_are_coarse_and_original_exception_survives(monkeypatch):
    configured(monkeypatch)
    error = RuntimeError("private@example.com secret prompt")
    with mock.patch.object(ai, "_enqueue") as enqueue:
        with pytest.raises(RuntimeError) as raised:
            with ai.generation("PRIVATE feature", "email@example.com"):
                raise error
    assert raised.value is error
    event = enqueue.call_args.args[1]
    assert event["properties"]["$ai_error"] == "ProviderError"
    assert event["properties"]["feature"] == "other"
    assert "secret" not in json.dumps(event) and "@" not in json.dumps(event)


def test_failed_or_full_export_does_not_change_model_result(monkeypatch):
    configured(monkeypatch)
    with mock.patch.object(ai, "_enqueue", side_effect=OSError("down")):
        with ai.generation(None, "claude-fixture"):
            value = "unchanged"
    assert value == "unchanged"
    full = queue.Queue(maxsize=1)
    full.put("occupied")
    with mock.patch.object(ai, "_queue", full), mock.patch.object(ai, "_worker"):
        ai._enqueue(("fixture", "https://us.i.posthog.com"), {})
    assert full.qsize() == 1


def test_delivery_uses_fixed_region_timeout_and_no_redirects():
    requests = pytest.importorskip("requests")
    with mock.patch.object(requests, "post") as post:
        ai._deliver(("fixture", "https://eu.i.posthog.com", {"event": "$ai_generation"}))
    assert post.call_args.args == ("https://eu.i.posthog.com/i/v0/e/",)
    assert post.call_args.kwargs["timeout"] == (2, 3)
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize("chat", [False, True])
def test_shared_single_and_chat_call_paths_are_observed_without_changing_limits(chat, monkeypatch):
    import config
    import ui.ai as helper
    from tests.test_run85_ai_claims import _FakeAnthropic

    configured(monkeypatch)
    monkeypatch.setattr(config, "AI_ENABLED", True)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "fixture-only")
    fake = _FakeAnthropic(reply="Research conditions remain unchanged.")
    kwargs = {"messages": [{"role": "user", "content": "private prompt"}]} if chat else {"user": "private prompt"}
    call = helper.ask_claude_chat if chat else helper.ask_claude
    with mock.patch.dict("sys.modules", {"anthropic": fake}), mock.patch.object(ai, "_enqueue") as enqueue:
        text, error = call(system="private system", feature="scan_summary", **kwargs)
    assert text and error is None
    assert len(fake.calls) == 1
    assert enqueue.call_count == 1
    assert "private" not in json.dumps(enqueue.call_args.args[1])
    with mock.patch.object(helper, "_over_daily_limit", return_value=True), mock.patch.object(ai, "_enqueue") as enqueue:
        text, error = call(system="s", username="fixture", **kwargs)
    assert text is None and "usage limit" in error
    enqueue.assert_not_called()
