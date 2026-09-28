"""P1-36 — Admin "Email delivery" card (separate from the Run 59 health status)."""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

ICONS = {"HEALTHY": "🟢", "WAITING": "🟢", "UNKNOWN": "⚪", "WARNING": "🟠"}


def render_email_health(result: Optional[Dict[str, Any]] = None) -> None:
    """Never raises; shows nothing if Streamlit is unavailable."""
    if st is None:
        return
    try:
        if result is None:
            from analytics.email_health import evaluate
            from db.email_job_runs import recent_email_runs

            result = evaluate(recent_email_runs(days=5))
        st.markdown("### ✉️ Email delivery")
        st.markdown(f"{ICONS.get(result['status'], '⚪')} **{result['status'].title()}** · "
                    f"trading day {result['day']}")
        for job in ("digest", "evening", "alerts"):
            j = result["jobs"][job]
            st.markdown(f"- {ICONS.get(j['status'], '⚪')} **{j['label']}**: {j['detail']}")
        st.caption("Counts only (no addresses). Separate from the System Health status above. "
                   "Failed sends are also reported to Sentry.")
    except Exception as exc:
        st.caption(f"Email delivery status unavailable: {type(exc).__name__}")
