"""P1-39 — describe an environment's email (SMTP) setup without exposing secrets.

The same settings live in three places (Streamlit Cloud, GitHub Actions, Render)
and drifted before (P1-33). `describe_smtp` summarises one environment: which
settings are missing, the sender's display name and domain, and whether the
domain is the one verified in Resend. It never returns the password, the SMTP
username, or the full sender address.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

VERIFIED_SENDER_DOMAIN = "ai.hsfinest.com"   # verified in Resend 2026-09-28 (P1-33)
DEFAULT_FROM_NAME = "HSF Alerts"


def describe_smtp(host: Any, user: Any, password: Any, smtp_from: Any,
                  from_name: Any = None, *, verified_domain: Optional[str] = None) -> Dict[str, Any]:
    from email.utils import parseaddr

    verified_domain = (verified_domain or VERIFIED_SENDER_DOMAIN).lower()
    missing = [name for name, val in (("SMTP_HOST", host), ("SMTP_USER", user), ("SMTP_PASS", password),
                                      ("SMTP_FROM", smtp_from)) if not str(val or "").strip()]
    name, addr = parseaddr(str(smtp_from or ""))
    domain = addr.rpartition("@")[2].lower() if "@" in addr else ""
    local = addr.rpartition("@")[0] if "@" in addr else ""
    result = {
        "configured": not missing,
        "missing": missing,
        "sender_domain": domain or None,
        "sender_masked": f"{local[:2]}***@{domain}" if domain else None,
        "display_name": name or str(from_name or "").strip() or DEFAULT_FROM_NAME,
        "domain_ok": domain == verified_domain,
        "verified_domain": verified_domain,
    }
    result["status"] = status_of(result)
    return result


def status_of(d: Dict[str, Any]) -> str:
    """OK / WARNING (wrong or missing sender domain) / MISSING (settings absent)."""
    if not d.get("configured"):
        return "MISSING"
    return "OK" if d.get("domain_ok") else "WARNING"


def describe_from_config() -> Dict[str, Any]:
    """This process's settings (the Streamlit app, or a GitHub Actions job)."""
    try:
        import config as c
    except Exception:
        return describe_smtp(None, None, None, None)
    return describe_smtp(getattr(c, "SMTP_HOST", ""), getattr(c, "SMTP_USER", ""), getattr(c, "SMTP_PASS", ""),
                         getattr(c, "SMTP_FROM", ""), getattr(c, "SMTP_FROM_NAME", ""))


def explain(d: Dict[str, Any]) -> str:
    """One line for the admin card."""
    if d.get("status") == "MISSING":
        return f"missing {', '.join(d.get('missing') or ['settings'])} — emails from here will not send"
    who = f"{d.get('display_name')} <{d.get('sender_masked')}>"
    if d.get("status") == "WARNING":
        return (f"sends as {who} — not the verified domain {d.get('verified_domain')}; "
                "Resend will reject emails to anyone but the account owner")
    return f"sends as {who} (verified domain)"
