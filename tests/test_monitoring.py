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
