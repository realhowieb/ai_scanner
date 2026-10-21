import json
from unittest import mock

# The core-runtime job uses unittest without development dependencies. These
# pytest checks run in the required web job with the full API dependencies.
try:
    import pytest
except ImportError as exc:
    import unittest

    raise unittest.SkipTest("pytest integration checks run in the web CI job") from exc

from scripts.integration_readiness import configuration_report, public_probe


def test_configuration_report_never_returns_credentials_or_claims_delivery():
    report = configuration_report({
        "POSTHOG_AI_ENABLED": "1", "POSTHOG_PROJECT_KEY": "private-project-key",
        "SENTRY_DSN": "https://private-dsn@example.com/1", "VAPID_PRIVATE_KEY": "private-vapid",
        "VAPID_SUBJECT": "mailto:private@example.com", "NEXT_PUBLIC_POSTHOG_KEY": "private-browser-key",
    })
    encoded = json.dumps(report)
    assert all(value not in encoded for value in ("private-project-key", "private-vapid", "private-browser-key", "example.com"))
    assert all(report[name]["delivery_verified"] is False for name in ("sentry", "web_push", "ai_observability", "browser_analytics"))
    assert report["sentry"]["dsn_present"] is True


@pytest.mark.parametrize("url", ["http://example.com", "https://key@example.com", "https://example.com?secret=x", "https://example.com/path"])
def test_probe_rejects_non_origin_or_insecure_targets(url):
    with pytest.raises(ValueError, match="HTTPS origins"):
        public_probe(url, "/healthz")


def test_public_probe_reports_liveness_and_build_csp_only():
    requests = pytest.importorskip("requests")
    health = mock.Mock(status_code=200, headers={})
    health.json.return_value = {"ok": True}
    page = mock.Mock(headers={"content-security-policy": "default-src 'self'; connect-src 'self' https://us.i.posthog.com"})
    with mock.patch.object(requests, "get", side_effect=[health, page]) as get:
        report = public_probe("https://app.example.test", "/api/healthz")
    assert report == {"reachable": True, "status": 200, "ok": True, "analytics_origin_in_csp": True}
    assert get.call_count == 2
    assert all(call.kwargs["allow_redirects"] is False for call in get.call_args_list)
