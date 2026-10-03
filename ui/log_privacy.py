"""Keep customer email addresses out of log output.

Scheduled jobs run in GitHub Actions on a public repository, so their logs are
public. Anything printed that may contain an address (a recipient, a username,
an SMTP error) goes through `log_id` or `redact` first.

P2-33: logs show no part of an address. `log_id` turns an email or username
into a stable pseudonym ("user#3fa9c2d1", the first 8 hex of its SHA-256), so
one person's lines can still be followed without revealing who they are; the
partial mask ("sa***@gmail.com") is only for screens the person themselves sees.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def _mask_local(local: str) -> str:
    return (local[:2] if len(local) > 2 else local[:1]) + "***"


def mask_email(value: Any) -> str:
    """'sample.customer@gmail.com' -> 'sa***@gmail.com'; a plain username -> 'ho***'."""
    text = str(value or "").strip()
    if not text:
        return ""
    if "@" in text:
        local, _, domain = text.rpartition("@")
        return f"{_mask_local(local)}@{domain}"
    return _mask_local(text)


def log_id(value: Any) -> str:
    """'Sample.Customer@gmail.com' -> 'user#<8 hex>'; same person -> same id."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return "user#" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def redact(value: Any) -> str:
    """Replace every email address inside free text (e.g. an exception message)."""
    return _EMAIL_RE.sub(lambda m: log_id(m.group(0)), str(value))
