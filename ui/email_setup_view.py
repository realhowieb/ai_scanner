"""P1-39 — Admin "Email setup" card: one row per environment.

Website (Streamlit Cloud) — read here, plus a "Send me a test email" button.
Scheduled jobs (GitHub Actions) — from the settings summary each email job
  records with its run (db.email_job_runs, P1-36).
Live alerts (Render billing service) — from its /health, on demand.
Never shows passwords, SMTP usernames or full sender addresses.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

ICONS = {"OK": "🟢", "WARNING": "🟠", "MISSING": "🔴", "UNKNOWN": "⚪"}


def latest_job_setup(runs) -> Optional[Dict[str, Any]]:
    """Newest recorded job run that carries a settings summary."""
    for r in runs or []:
        smtp = (r.get("stats") or {}).get("smtp")
        if isinstance(smtp, dict) and smtp.get("status"):
            return {**smtp, "at": r.get("at"), "job": r.get("job")}
    return None


def billing_setup(health: Dict[str, Any]) -> Dict[str, Any]:
    """Turn billing /health's `email` block into the same shape as describe_smtp."""
    from ui.email_setup import VERIFIED_SENDER_DOMAIN

    email = (health or {}).get("email")
    if not isinstance(email, dict):
        return {"status": "UNKNOWN", "detail": "billing service didn't report email settings (older version?)"}
    domain = email.get("sender_domain")
    if not email.get("configured"):
        return {"status": "MISSING",
                "detail": f"missing {', '.join(email.get('missing') or ['settings'])} — live alert emails will not send"}
    if domain != VERIFIED_SENDER_DOMAIN:
        return {"status": "WARNING", "detail": f"sender domain {domain or 'unknown'} is not {VERIFIED_SENDER_DOMAIN}"}
    return {"status": "OK", "detail": f"sends from *@{domain} (verified domain)"}


def _fetch_billing_health() -> Dict[str, Any]:
    import requests

    base = (os.getenv("BILLING_API_BASE") or "https://ai-scanner-h2c8.onrender.com").strip().rstrip("/")
    r = requests.get(f"{base}/health", timeout=float(os.getenv("BILLING_HEALTH_TIMEOUT", "5")))
    return r.json() if r.content else {}


def render_email_setup() -> None:
    """Never raises."""
    if st is None:
        return
    try:
        from ui.email_setup import describe_from_config, explain

        st.markdown("### 📮 Email setup")
        web = describe_from_config()
        st.markdown(f"- {ICONS[web['status']]} **Website (Streamlit Cloud)**: {explain(web)}")

        try:
            from db.email_job_runs import recent_email_runs

            job = latest_job_setup(recent_email_runs(days=7))
        except Exception:
            job = None
        if job:
            st.markdown(f"- {ICONS[job['status']]} **Scheduled emails (GitHub Actions)**: {explain(job)} "
                        f"· as of the {job.get('job')} run at {str(job.get('at'))[:16]} UTC")
        else:
            st.markdown(f"- {ICONS['UNKNOWN']} **Scheduled emails (GitHub Actions)**: no run recorded yet "
                        "(fills in after the next digest, wrap or alert run)")

        if st.button("Check live-alert email (Render)", key="admin_email_setup_billing"):
            try:
                b = billing_setup(_fetch_billing_health())
            except Exception:
                b = {"status": "UNKNOWN", "detail": "billing service didn't respond"}
            st.session_state["_email_setup_billing"] = b
        b = st.session_state.get("_email_setup_billing")
        if b:
            st.markdown(f"- {ICONS[b['status']]} **Live alerts (Render billing)**: {b['detail']}")

        _render_test_email()
        st.caption(f"Verified sender domain: {web['verified_domain']}. Passwords and SMTP usernames are never shown.")
    except Exception as exc:
        st.caption(f"Email setup unavailable: {type(exc).__name__}")


def _render_test_email() -> None:
    """Send one test email to the signed-in admin (the website's own email path)."""
    to = str(st.session_state.get("username") or "").strip().lower()
    if "@" not in to:
        st.caption("Test email needs an admin account that signs in with an email address.")
        return
    if st.button(f"Send me a test email ({to})", key="admin_email_setup_test"):
        from ui.email_utils import send_alert_email

        ok = send_alert_email(to, "Test email from HSF Admin",
                              "This is a test from the HSF Admin page (website email path). "
                              "If you received it, the website can send email.")
        if ok:
            st.success("Sent. Check your inbox (and spam) for 'Test email from HSF Admin'.")
        else:
            st.error("The email service refused or didn't respond. Check the website's SMTP settings above.")
