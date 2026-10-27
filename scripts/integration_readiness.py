"""Read-only integration readiness. Never sends events, errors, AI calls or pushes.

Run in the API/web host environment for configuration evidence. Optional --web-url
and --api-url check public liveness; they cannot verify authenticated delivery.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from urllib.parse import urlsplit

_HOSTS = {"https://us.i.posthog.com", "https://eu.i.posthog.com"}


def configuration_report(env: Mapping[str, str]) -> dict:
    def present(name: str) -> bool:
        return bool(env.get(name, "").strip())

    host = env.get("POSTHOG_HOST", "https://us.i.posthog.com").rstrip("/")
    browser_host = env.get("NEXT_PUBLIC_POSTHOG_HOST", "https://us.i.posthog.com").rstrip("/")
    return {
        "scope": "current_process_environment_only",
        "browser_analytics": {"key_present": present("NEXT_PUBLIC_POSTHOG_KEY"), "host_valid": browser_host in _HOSTS,
                              "delivery_verified": False},
        "ai_observability": {"enabled": env.get("POSTHOG_AI_ENABLED") == "1", "key_present": present("POSTHOG_PROJECT_KEY"),
                             "host_valid": host in _HOSTS, "delivery_verified": False},
        "sentry": {"dsn_present": present("SENTRY_DSN"), "release_present": present("SENTRY_RELEASE"),
                   "delivery_verified": False},
        "web_push": {"enabled": env.get("HSF_WEB_PUSH_ENABLED", "1").strip() != "0",
                     "private_key_present": present("VAPID_PRIVATE_KEY"), "contact_present": present("VAPID_SUBJECT"),
                     "delivery_verified": False},
    }


def public_probe(base: str, path: str) -> dict:
    parts = urlsplit(base)
    if parts.scheme != "https" or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("Probe URLs must be HTTPS origins without credentials, paths or query strings")
    import requests

    result: dict = {"reachable": False}
    try:
        response = requests.get(base.rstrip("/") + path, timeout=(3, 5), allow_redirects=False)
        result.update(reachable=True, status=response.status_code)
        if path == "/api/healthz":
            result["ok"] = response.status_code == 200 and response.json().get("ok") is True
            page = requests.get(base.rstrip("/"), timeout=(3, 5), allow_redirects=False)
            csp = page.headers.get("content-security-policy", "")
            connect = next((p.split()[1:] for p in csp.split(";") if p.strip().startswith("connect-src ")), [])
            result["analytics_origin_in_csp"] = any(h in connect for h in _HOSTS)
        else:
            result["ok"] = response.status_code == 200 and response.json().get("ok") is True
    except Exception:
        result["probe_failed"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-url")
    parser.add_argument("--api-url")
    args = parser.parse_args()
    report = configuration_report(os.environ)
    for name, url, path in (("web", args.web_url, "/api/healthz"), ("api", args.api_url, "/healthz")):
        if url:
            report[name + "_public_probe"] = public_probe(url, path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
