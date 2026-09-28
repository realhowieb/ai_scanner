"""Keep customer email addresses out of log output.

Scheduled jobs run in GitHub Actions on a public repository, so their logs are
public. Anything printed that may contain an address (a recipient, a username,
an SMTP error) goes through `mask_email` or `redact` first.
"""
from __future__ import annotations

import re
from typing import Any

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def _mask_local(local: str) -> str:
    return (local[:2] if len(local) > 2 else local[:1]) + "***"


def mask_email(value: Any) -> str:
    """'lovenatural4life@gmail.com' -> 'lo***@gmail.com'; a plain username -> 'ho***'."""
    text = str(value or "").strip()
    if not text:
        return ""
    if "@" in text:
        local, _, domain = text.rpartition("@")
        return f"{_mask_local(local)}@{domain}"
    return _mask_local(text)


def redact(value: Any) -> str:
    """Mask every email address inside free text (e.g. an exception message)."""
    return _EMAIL_RE.sub(lambda m: f"{_mask_local(m.group(1))}@{m.group(2)}", str(value))
