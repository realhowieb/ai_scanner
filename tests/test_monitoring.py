from __future__ import annotations

from ui.monitoring import _before_send, _is_handled_cookie_component_directory_read


def _event(*, logger: str, exception_type: str, value: str) -> dict:
    return {
        "logger": logger,
        "exception": {
            "values": [
                {
                    "type": exception_type,
                    "value": value,
                }
            ]
        },
    }


def test_filters_handled_cookie_component_directory_probe() -> None:
    event = _event(
        logger="streamlit.web.server.component_request_handler",
        exception_type="IsADirectoryError",
        value=(
            "[Errno 21] Is a directory: "
            "'/home/adminuser/venv/lib/python3.14/site-packages/streamlit_cookies_manager/build'"
        ),
    )

    assert _is_handled_cookie_component_directory_read(event) is True
    assert _before_send(event, None) is None


def test_keeps_other_component_read_errors() -> None:
    event = _event(
        logger="streamlit.web.server.component_request_handler",
        exception_type="FileNotFoundError",
        value="[Errno 2] No such file: '/tmp/component/build/index.html'",
    )

    assert _is_handled_cookie_component_directory_read(event) is False
    assert _before_send(event, None) is event


def test_keeps_directory_errors_from_application_code() -> None:
    event = _event(
        logger="hsf.application",
        exception_type="IsADirectoryError",
        value="[Errno 21] Is a directory: '/tmp/streamlit_cookies_manager/build'",
    )

    assert _is_handled_cookie_component_directory_read(event) is False
    assert _before_send(event, None) is event


def test_keeps_other_streamlit_component_directories() -> None:
    event = _event(
        logger="streamlit.web.server.component_request_handler",
        exception_type="IsADirectoryError",
        value="[Errno 21] Is a directory: '/tmp/another_component/build'",
    )

    assert _is_handled_cookie_component_directory_read(event) is False
    assert _before_send(event, None) is event


def test_api_and_browser_reports_reach_sentry_in_memory(monkeypatch):
    """Exercise the real SDK transport without sending a production error."""
    from unittest import mock

    import pytest

    sentry_sdk = pytest.importorskip("sentry_sdk")
    from sentry_sdk.transport import Transport

    from api.monitoring import capture_client_error, init_api_monitoring
    from ui import monitoring

    events = []

    class MemoryTransport(Transport):
        def capture_envelope(self, envelope):
            event = envelope.get_event()
            if event:
                events.append(event)

    monkeypatch.setenv("SENTRY_DSN", "https://fixture@errors.example/1")
    monkeypatch.setenv("SENTRY_ENV", "test")
    monkeypatch.setattr(monitoring, "_initialized", False)
    with mock.patch.object(sentry_sdk, "init") as init:
        assert init_api_monitoring() is True
    options = init.call_args.kwargs
    assert options["send_default_pii"] is False
    assert options["include_local_variables"] is False
    assert options["max_request_body_size"] == "never"
    client = sentry_sdk.Client(**options, transport=MemoryTransport(), default_integrations=False)
    try:
        with sentry_sdk.new_scope() as scope:
            scope.set_client(client)
            capture_client_error({"kind": "boundary", "message": "Synthetic browser failure", "path": "/scanner"})
            monitoring.capture(RuntimeError("Synthetic API failure"))
        client.flush()
    finally:
        client.close()
    assert len(events) == 2
    assert events[0]["message"] == "Web: Synthetic browser failure"
    assert events[0]["tags"]["component"] == "web"
    assert events[1]["exception"]["values"][0]["type"] == "RuntimeError"


def test_missing_sentry_dsn_does_not_initialize(monkeypatch):
    from unittest import mock

    import pytest

    pytest.importorskip("sentry_sdk")

    from ui import monitoring

    monkeypatch.setattr(monitoring, "_initialized", False)
    with mock.patch.object(monitoring, "_get_dsn", return_value=None), mock.patch("sentry_sdk.init") as init:
        assert monitoring.init_sentry("api") is False
    init.assert_not_called()
