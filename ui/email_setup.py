"""P1-39 — describe an environment's email (SMTP) setup without exposing secrets.

The same settings live in three places (Streamlit Cloud, GitHub Actions, Render)
and drifted before (P1-33). `describe_smtp` summarises one environment: which
settings are missing, the sender's display name and domain, and whether the
domain is the one verified in Resend. It never returns the password, the SMTP
username, or the full sender address.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

VERIFIED_SENDER_DOMAIN = "hsfinest.com"   # sender moved off ai.hsfinest.com 2026-11-16 (Resend un-verified it)
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


# ---- Is the sender domain actually verified in Resend? ------------------------------------------
# The domain check above only compares SMTP_FROM with a name written in code. On 2026-11-14
# Resend un-verified the sender domain and every email failed for three days while the
# health card still said OK. This asks Resend itself (GET /domains, needs a full-access key).
RESEND_API = "https://api.resend.com"
DOMAIN_CHECK_TTL_S = 600
_domain_cache: Dict[str, Any] = {}


def _resend_key() -> Optional[str]:
    try:
        from services.marketing_list import resend_key

        return resend_key()
    except Exception:
        return None


def resend_domain_status(domain: Optional[str], *, key: Optional[str] = None, session: Any = None,
                         now: Optional[float] = None) -> Dict[str, Any]:
    """{state, resend_status, domain}. state: verified / unverified (listed, not verified) /
    missing (not in the Resend account) / unknown (no key, no access, or Resend unreachable).
    Cached 10 minutes per domain. Never raises and never returns the key."""
    import time

    domain = str(domain or "").strip().lower()
    now = time.monotonic() if now is None else now
    hit = _domain_cache.get(domain)
    if hit and now - hit[0] < DOMAIN_CHECK_TTL_S:
        return hit[1]
    result: Dict[str, Any] = {"domain": domain or None, "state": "unknown", "resend_status": None}
    key = key or _resend_key()
    if not domain:
        result["why"] = "no sender address"
    elif not key:
        result["why"] = "no Resend API key"
    else:
        try:
            if session is None:
                import httpx as session  # in api/requirements.txt
            res = session.request("GET", f"{RESEND_API}/domains", timeout=10,
                                  headers={"Authorization": f"Bearer {key}"})
            code = int(res.status_code)
            if code != 200:
                result["why"] = f"Resend answered {code}"
            else:
                body = res.json()
                rows = body.get("data") if isinstance(body, dict) else body
                match = next((r for r in rows or [] if str(r.get("name") or "").lower() == domain), None)
                if match is None:
                    result["state"] = "missing"
                else:
                    status = str(match.get("status") or "").lower()
                    result["resend_status"] = status or None
                    result["state"] = "verified" if status == "verified" else "unverified"
        except Exception as e:
            result["why"] = f"Resend unreachable ({type(e).__name__})"
    _domain_cache[domain] = (now, result)
    return result


def sender_domain_row(d: Optional[Dict[str, Any]] = None, *, session: Any = None) -> Dict[str, Any]:
    """One row for the Admin > Email delivery card: is SMTP_FROM's domain verified in Resend?"""
    d = d or describe_from_config()
    if d.get("status") == "MISSING":
        return {"job": "sender", "label": "Sender domain", "status": "WARNING", "detail": explain(d)}
    s = resend_domain_status(d.get("sender_domain"), session=session)
    dom = s.get("domain") or "?"
    if s["state"] == "verified":
        return {"job": "sender", "label": "Sender domain", "status": "HEALTHY",
                "detail": f"{dom} is verified in Resend"}
    if s["state"] == "unverified":
        return {"job": "sender", "label": "Sender domain", "status": "WARNING",
                "detail": f"{dom} is '{s.get('resend_status')}' in Resend, not verified — every email will fail"}
    if s["state"] == "missing":
        return {"job": "sender", "label": "Sender domain", "status": "WARNING",
                "detail": f"{dom} is not in the Resend account — every email will fail"}
    return {"job": "sender", "label": "Sender domain", "status": "UNKNOWN",
            "detail": f"couldn't ask Resend about {dom} ({s.get('why') or 'unknown'})"}


def sender_domain_problem(*, session: Any = None) -> Optional[str]:
    """A one-line problem when Resend says the sender domain isn't verified, else None.
    'unknown' (no key / Resend down) is not a problem: it must never raise a false alarm."""
    row = sender_domain_row(session=session)
    return row["detail"] if row["status"] == "WARNING" else None
