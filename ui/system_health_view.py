"""Run 59 — System Health admin view (read-only, anti-peeking).

Renders the SAME health model the System Health workflow writes
(`analytics.system_health`, persisted to `hsf_system_health`). It never computes
effectiveness statistics. The first screen answers "Do I need to do anything?".
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

_ICON = {"HEALTHY": "🟢", "WAITING": "🔵", "DEGRADED": "🟠", "ACTION_REQUIRED": "🔴", "UNKNOWN": "⚪"}
_HUMAN = {"NO_ACTION": "None", "WATCH": "Watch", "AUTOMATIC_RECOVERY_CANDIDATE": "None (automatable)",
          "HUMAN_ACTION_REQUIRED": "REQUIRED"}


def _load() -> Optional[Dict[str, Any]]:
    try:
        from db.system_health import load_latest
        return load_latest()
    except Exception:
        return None


def render_system_health(report: Optional[Dict[str, Any]] = None) -> None:
    from analytics import system_health as sh

    report = report or _load()
    st.markdown("### 🩺 System Health")
    if not report:
        st.info("No health snapshot yet. Run the **System Health** workflow (Actions → System Health).")
        return
    status = report["system_status"]
    c1, c2, c3 = st.columns(3)
    c1.metric("System status", f"{_ICON.get(status, '')} {status}")
    c2.metric("Health score", f"{report['health_score']}/100")
    c3.metric("Human action", _HUMAN.get(report["human_action"], report["human_action"]))
    st.caption(f"Generated {report['generated_at']} · next expected scan "
               f"{report['market'].get('next_expected_scan')} · autonomy "
               f"**{report['autonomy_readiness']['state']}**")

    cols = st.columns(5)
    for i, name in enumerate(sh.SUBSYSTEMS):
        s = report["subsystems"][name]
        with cols[i % 5]:
            st.markdown(f"{_ICON.get(s['status'], '')} **{sh._TITLES[name]}**  \n"
                        f"{sh.subsystem_label(s)}")
            st.caption(s["reason"])

    fe = report["subsystems"]["forward_evidence"]["metrics"] or {}
    st.markdown("#### Forward experiment progress")
    if fe:
        p1, p2, p3 = st.columns(3)
        p1.metric("Trading days", f"{fe.get('trading_days_collected')} / {fe.get('trading_days_preferred')}")
        p2.metric("Scan runs", f"{fe.get('scan_runs_collected')} / {fe.get('scan_runs_preferred')}")
        cov = fe.get("min_cohort_maturation_pct")
        p3.metric("Lowest cohort coverage", "—" if cov is None else f"{cov}%")
        st.caption(f"Status **{report.get('forward_evidence_status')}** · formal evaluation "
                   f"**{fe.get('formal_evaluation')}** (human approval required; never automatic)")
    else:
        st.caption("Readiness unavailable.")

    st.markdown("#### Active incidents")
    inc = report.get("incidents") or []
    if not inc:
        st.success("No incidents.")
    for i in inc:
        line = (f"**{i['severity']}** · {i['incident_id']} · {i['age_hours']}h — {i['summary']}  \n"
                f"Recommended: {i['recommended_action'] or '—'} · human action {i['human_action']}"
                f"{' · automation candidate' if i['automation_candidate'] else ''}")
        (st.error if i["severity"] == "CRITICAL" else st.warning if i["severity"] == "WARNING" else st.info)(line)

    with st.expander("Provider health"):
        st.dataframe(report["subsystems"]["market_data"]["metrics"].get("providers") or [], width="stretch")
    with st.expander("Workflow freshness"):
        st.dataframe(report["subsystems"]["workflows"]["metrics"].get("workflows") or [], width="stretch")
    with st.expander("Research data quality"):
        rc = report["subsystems"]["research_capture"]["metrics"] or {}
        st.json({k: rc.get(k) for k in ("observations", "by_cohort", "untagged", "duplicate_ids", "malformed",
                                        "metadata_block_pct", "metadata_scope_n")})
        st.json({"artifacts": report["subsystems"]["artifact_freshness"]["metrics"].get("artifacts")})
    st.caption("Operational observability only: no effectiveness statistics are computed or shown.")
