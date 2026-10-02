"""Read-only Admin control-center and research analytics views."""
from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from analytics import admin_research as ar
from db.admin_analytics import load_admin_data
from ui.arrow_safe import arrow_safe
from ui.observation_explorer import render_observation_explorer

DATE_WINDOWS = {"7 days": 7, "30 days": 30, "90 days": 90, "1 year": 365}


@st.cache_data(show_spinner=False, ttl=300)
def load_admin_data_cached(days: int, include_research: bool) -> dict[str, Any]:
    return load_admin_data(days=int(days), include_research=include_research)


def render_admin_analytics(
    *,
    username: str,
    db_status: str,
    admin_users: object,
    render_admin_users_panel: Any,
    render_system_health_panel: Any,
) -> dict[str, Any]:
    """Render the selected Admin view without eagerly running hidden sections."""
    if not bool(st.session_state.get("is_admin")):
        st.info("Admin analytics are only available to admin users.")
        return {"available": False, "errors": {"authorization": "denied"}}

    sections = (
        "Overview", "Product", "Signal Performance", "Stair-Stepper",
        "Research Evidence", "Observation Explorer", "System Health",
    )
    section = st.radio("Admin view", sections, horizontal=True, key="admin_analytics_view")
    period_label = st.selectbox(
        "Analytics period",
        list(DATE_WINDOWS),
        index=2,
        key="admin_analytics_period",
        disabled=section == "Observation Explorer",
    )
    days = DATE_WINDOWS[period_label]
    include_research = section in {"Signal Performance", "Stair-Stepper", "Research Evidence"}
    with st.spinner("Loading Admin intelligence..."):
        data = load_admin_data_cached(days, include_research)

    if not data.get("available"):
        st.warning("Admin analytics are temporarily unavailable. Existing Admin tools remain available below.")
        return data

    st.caption(f"Data updated: {_display_time(data.get('loaded_at'))} · bounded to the last {days} days")
    if data.get("errors"):
        missing = ", ".join(sorted(data["errors"]))
        st.info(f"Some sources are unavailable: {missing}. Available sections are still shown.")

    if section == "Overview":
        render_overview(data)
    elif section == "Product":
        render_product(data)
        st.markdown("#### Account administration")
        render_admin_users_panel(username=username, ADMIN_USERS=admin_users, db_status=db_status)
    elif section == "Signal Performance":
        render_signal_performance(data)
    elif section == "Stair-Stepper":
        render_stair_stepper(data)
    elif section == "Research Evidence":
        render_research_evidence(data)
    elif section == "Observation Explorer":
        render_observation_explorer(health=data.get("health"))
    else:
        render_operations(data, render_system_health_panel=render_system_health_panel)
    return data


def render_overview(data: Mapping[str, Any]) -> None:
    users = list(data.get("users") or [])
    events = list(data.get("events") or [])
    runs = list(data.get("runs") or [])
    observations = list(data.get("observations") or [])
    funnel = ar.evidence_funnel(observations)
    now = dt.datetime.now(dt.timezone.utc)

    active_hashes = {
        str(event.get("user_hash")) for event in events
        if event.get("user_hash") and event.get("event_name") in {"first_authenticated_session", "return_session"}
        and _within_days(event.get("occurred_at"), 30, now)
    }
    new_users = sum(_within_days(user.get("created_at"), 7, now) for user in users)
    scans_today = sum(
        (_parse_time(run.get("created_at")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)).date() == now.date()
        for run in runs
    )
    paying = sum(
        bool(user.get("is_active")) and bool(user.get("has_subscription"))
        and ar.normalize_plan(user.get("tier")) in {"pro", "premium"}
        for user in users
    )

    st.markdown("### HSF Control Center")
    for cards in (
        [("Total users", len(users)), ("Active users (30d)", len(active_hashes)),
         ("New users (7d)", new_users), ("Scans today", scans_today)],
        [("Research observations", funnel["captured"]), ("Matured observations", funnel["matured"]),
         ("Pending maturation", funnel["pending"]), ("Current paying users", paying)],
    ):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, cards):
            column.metric(label, f"{value:,}")

    st.markdown("#### HSF AI system status")
    statuses = _platform_statuses(data, funnel)
    status_df = pd.DataFrame(statuses)
    st.dataframe(arrow_safe(status_df), width="stretch", hide_index=True)
    with st.expander("Why these statuses?"):
        for item in statuses:
            st.markdown(f"**{item['System']} · {item['Status']}**")
            st.caption(item["Reason"])


def render_product(data: Mapping[str, Any]) -> None:
    users = list(data.get("users") or [])
    events = list(data.get("events") or [])
    runs = list(data.get("runs") or [])
    st.markdown("### Product analytics")

    range_days = int(data.get("days") or 90)
    daily_users = _daily_counts(users, "created_at", range_days)
    if not daily_users.empty:
        st.markdown("#### Users over time")
        st.line_chart(daily_users, height=260)
    else:
        st.info("No user signups are available for this period.")

    active_users = [user for user in users if user.get("is_active")]
    plans = Counter(ar.normalize_plan(user.get("tier")) for user in active_users)
    plan_df = pd.DataFrame([
        {"Plan": plan.title(), "Users": plans.get(plan, 0)} for plan in ar.CANONICAL_PLANS
    ])
    st.markdown("#### Current plan distribution")
    st.bar_chart(plan_df.set_index("Plan"), height=240)

    daily_runs = _daily_run_counts(runs, range_days)
    st.markdown("#### Scan activity")
    if daily_runs.empty:
        st.info("No scan runs are available for this period.")
    else:
        st.bar_chart(daily_runs, height=260)
        st.caption("Scheduled scans have no username; user scans are account-scoped runs.")

    signups = {str(event.get("user_hash")) for event in events
               if event.get("event_name") == "signup_completed" and event.get("user_hash")}
    paid_events = [event for event in events if event.get("event_name") == "successful_paid_conversion"]
    paid = {str(event.get("user_hash")) for event in paid_events if event.get("user_hash")}
    st.markdown("#### Recorded conversion events")
    cols = st.columns(3)
    cols[0].metric("Signup events", len(signups))
    cols[1].metric("Paid-conversion events", len(paid))
    conversion = (len(signups & paid) / len(signups)) if signups else None
    cols[2].metric("Recorded signup to paid", _pct(conversion))
    destination = Counter(ar.normalize_plan(event.get("plan")) for event in paid_events)
    if paid_events:
        st.dataframe(pd.DataFrame([
            {"Paid destination": plan.title(), "Recorded users": destination.get(plan, 0)}
            for plan in ("pro", "premium")
        ]), width="stretch", hide_index=True)
    st.caption(
        "Only recorded paid-conversion events are shown. Free-to-Pro, Free-to-Premium, and Pro-to-Premium "
        "history is not inferred from current account tiers because a complete transition ledger does not exist."
    )


def render_signal_performance(data: Mapping[str, Any]) -> None:
    observations = list(data.get("observations") or [])
    rows = ar.flatten_research(observations)
    if not rows:
        st.info("No scanner research observations are available for this period.")
        return

    st.markdown("### Signal performance")
    signals = sorted({str(row["signal"]) for row in rows})
    cohorts = sorted({str(row["cohort"]) for row in rows})
    horizons = sorted({str(row["horizon"]) for row in rows if row.get("horizon")})
    sessions = sorted({str(row["session"]) for row in rows})
    f1, f2, f3 = st.columns(3)
    signal = f1.selectbox("Signal", ["All"] + signals, key="admin_signal_filter")
    cohort = f2.selectbox("Research cohort", ["All"] + cohorts, key="admin_cohort_filter")
    horizon = f3.selectbox("Horizon", ["All"] + horizons, key="admin_horizon_filter")
    f4, f5, f6 = st.columns(3)
    session = f4.selectbox("Market session", ["All"] + sessions, key="admin_session_filter")
    min_n = int(f5.number_input("Minimum sample size", min_value=1, max_value=1000,
                                value=ar.DEFAULT_MIN_EVIDENCE_N, step=1, key="admin_min_evidence_n"))
    available_scores = [float(row["hsf_score"]) for row in rows if row.get("hsf_score") is not None]
    score_range = None
    if available_scores:
        score_low, score_high = int(min(available_scores)), int(max(available_scores))
        score_range = f6.slider("HSF Score range", min_value=score_low, max_value=score_high,
                                value=(score_low, score_high), key="admin_score_range")
    else:
        f6.caption("HSF Score filter unavailable: canonical score is not stored in these observations.")
    filtered = [row for row in rows if _matches(row, signal=signal, cohort=cohort,
                                                horizon=horizon, session=session,
                                                score_range=score_range)]
    summary = ar.performance_summary(filtered, min_n=min_n)
    unique_observations = len({row.get("observation_id") for row in filtered})
    columns = st.columns(4)
    columns[0].metric("Observations", f"{unique_observations:,}")
    columns[1].metric("Matured outcome rows", f"{summary['matured']:,}")
    columns[2].metric("Positive outcome", _pct(summary["positive_rate"]))
    columns[3].metric("Median forward return", _pct(summary["median_return"]))
    columns = st.columns(4)
    columns[0].metric("Average forward return", _pct(summary["average_return"]))
    columns[1].metric("Average MFE", _pct(summary["average_mfe"]))
    columns[2].metric("Average MAE", _pct(summary["average_mae"]))
    columns[3].metric("Evidence", "SUFFICIENT" if summary["sufficient"] else "INSUFFICIENT")
    st.caption("Returns are fractions displayed as percentages. Missing outcomes are excluded, never converted to zero.")

    grouped = ar.grouped_performance(rows, ("signal", "horizon"), min_n=min_n)
    metric_label = st.selectbox(
        "Heatmap metric",
        ["Median Return", "Average Return", "Positive Outcome %", "Observation Count", "MFE", "MAE"],
        key="admin_signal_heatmap_metric",
    )
    _render_heatmap(grouped, row_key="signal", col_key="horizon", metric_label=metric_label, min_n=min_n)

    st.markdown("#### HSF Score calibration")
    scored = [row for row in rows if row.get("hsf_score") is not None]
    if not scored:
        st.info(
            "Canonical HSF Score is not persisted in the current research observations. "
            "Score calibration and the Score × Horizon matrix are unavailable; Breakout Score was not substituted."
        )
    else:
        score_horizon = st.selectbox("Calibration horizon", horizons, key="admin_score_horizon")
        score_rows = [row for row in scored if row.get("horizon") == score_horizon]
        score_groups = ar.grouped_performance(score_rows, ("score_bucket",), min_n=min_n)
        st.dataframe(arrow_safe(pd.DataFrame(score_groups)), width="stretch", hide_index=True)
        score_matrix = ar.grouped_performance(scored, ("score_bucket", "horizon"), min_n=min_n)
        _render_heatmap(score_matrix, row_key="score_bucket", col_key="horizon",
                        metric_label=metric_label, min_n=min_n)

    export_df = pd.DataFrame(filtered)
    st.download_button(
        "Download filtered research CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="hsf_admin_research_filtered.csv",
        mime="text/csv",
        key="admin_research_export",
    )


def render_stair_stepper(data: Mapping[str, Any]) -> None:
    from analytics.stair_step_research import build_research_report

    observations = [row for row in data.get("observations") or [] if ar.is_stair_stepper(row)]
    report = build_research_report(observations)
    st.markdown("### Stair-Stepper evidence")
    st.caption(
        f"Latest observation: {_latest_timestamp(observations)} · "
        "kept separate from scheduled scanner research"
    )
    if not observations:
        st.info("No Stair-Stepper outcomes have matured for this period.")
        return
    horizons = list((report.get("methodology") or {}).get("outcome_horizons_minutes") or {})
    horizon = st.selectbox("Outcome horizon", horizons, index=min(1, len(horizons) - 1),
                           key="admin_stair_horizon")
    table = []
    for item in report.get("window_comparison") or []:
        outcome = (item.get("horizons") or {}).get(horizon) or {}
        table.append({
            "Window": item.get("window"),
            "Observations": item.get("observations"),
            "Matured": outcome.get("n", 0),
            "Positive %": _pct_number(outcome.get("directional_win_rate")),
            "Average return": _pct_number(outcome.get("mean_directional_return")),
            "Median return": _pct_number(outcome.get("median_directional_return")),
            "MFE": _pct_number(outcome.get("mean_mfe")),
            "MAE": _pct_number(outcome.get("mean_mae")),
        })
    table_df = pd.DataFrame(table)
    st.dataframe(arrow_safe(table_df), width="stretch", hide_index=True)
    if not table_df.empty:
        chart = table_df[["Window", "Median return", "Average return"]].set_index("Window")
        st.bar_chart(chart, height=280)

    verdict = report.get("best_window_verdict") or {}
    if verdict.get("status") == "EVIDENCE_AVAILABLE" and verdict.get("best_window") is not None:
        best_window = verdict["best_window"]
        best = next((row for row in table if row["Window"] == best_window), {})
        st.success(
            f"Best-supported window: {best_window} bars · N {best.get('Matured', 0)} · "
            f"Positive {best.get('Positive %', '—')} · Median {best.get('Median return', '—')}"
        )
        st.caption(verdict.get("reason"))
    else:
        st.info("NO WINDOW HAS SUFFICIENT EVIDENCE YET")
        st.caption(verdict.get("reason") or "The minimum sample and trading-day gates have not been met.")


def render_research_evidence(data: Mapping[str, Any]) -> None:
    observations = list(data.get("observations") or [])
    funnel = ar.evidence_funnel(observations)
    health = data.get("health") or {}
    st.markdown("### Research evidence")
    st.caption(
        f"Latest observation: {_display_time(funnel['latest_observation'])} · "
        f"Latest matured observation: {_display_time(funnel['latest_matured_observation'])} · "
        f"Last maturation run: {_last_maturation(health)}"
    )
    labels = ["Captured", "Valid metadata", "Matured", "Usable outcome", "Included in research"]
    values = [funnel["captured"], funnel["valid_metadata"], funnel["matured"],
              funnel["usable_outcome"], funnel["included_in_research"]]
    funnel_df = pd.DataFrame({"Stage": labels, "Count": values})
    base = funnel["captured"] or 0
    funnel_df["% of captured"] = [round(value / base * 100, 1) if base else None for value in values]
    st.bar_chart(funnel_df.set_index("Stage")[["Count"]], height=280)
    st.dataframe(arrow_safe(funnel_df), width="stretch", hide_index=True)

    columns = st.columns(4)
    columns[0].metric("Pending", f"{funnel['pending']:,}")
    columns[1].metric("Excluded", f"{funnel['excluded']:,}")
    columns[2].metric("Metadata complete", _pct(funnel["metadata_completeness"]))
    columns[3].metric("Oldest pending", _display_time(funnel["oldest_pending_observation"]))
    st.markdown("#### Cohort coverage")
    cohort_df = pd.DataFrame([{"Cohort": key, "Observations": value}
                              for key, value in sorted(funnel["cohorts"].items())])
    if cohort_df.empty:
        st.info("No research cohorts are available for this period.")
    else:
        st.bar_chart(cohort_df.set_index("Cohort"), height=240)

    forward = ((health.get("subsystems") or {}).get("forward_evidence") or {})
    metrics = forward.get("metrics") or {}
    st.markdown("#### Evidence readiness")
    st.write(f"**{forward.get('detail_state') or health.get('forward_evidence_status') or 'UNKNOWN'}**")
    st.caption(forward.get("reason") or "No persisted readiness explanation is available.")
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Trading days", f"{metrics.get('trading_days_collected', '—')} / "
                  f"{metrics.get('trading_days_preferred', '—')}")
        c2.metric("Scan runs", f"{metrics.get('scan_runs_collected', '—')} / "
                  f"{metrics.get('scan_runs_preferred', '—')}")
        c3.metric("Formal evaluation", metrics.get("formal_evaluation") or "UNKNOWN")


def render_operations(data: Mapping[str, Any], *, render_system_health_panel: Any) -> None:
    try:
        render_system_health_panel(data.get("health"))
    except Exception:
        st.info("No System Health snapshot is available.")
    st.markdown("#### Autonomous Recovery")
    recovery = list(data.get("recovery") or [])
    if not recovery:
        st.info("No Autonomous Recovery events are available for this period.")
        return
    rows = [{
        "Started": item.get("started_at"),
        "Incident": item.get("incident_id"),
        "Action": item.get("action"),
        "Result": item.get("result"),
    } for item in recovery]
    st.dataframe(arrow_safe(pd.DataFrame(rows)), width="stretch", hide_index=True, height=300)


def _platform_statuses(data: Mapping[str, Any], funnel: Mapping[str, Any]) -> list[dict[str, str]]:
    health = data.get("health") or {}
    subsystems = health.get("subsystems") or {}

    def subsystem(name: str, fallback: str = "UNKNOWN") -> tuple[str, str]:
        item = subsystems.get(name) or {}
        return str(item.get("status") or fallback), str(item.get("reason") or "No persisted status is available.")

    product_status = "HEALTHY" if "users" not in (data.get("errors") or {}) else "UNKNOWN"
    product_reason = "User and plan data loaded successfully." if product_status == "HEALTHY" else "User data unavailable."
    research_status, research_reason = subsystem("research_capture", "COLLECTING" if funnel.get("captured") else "UNKNOWN")
    scanner_status, scanner_reason = subsystem("scanner")
    maturation_status, maturation_reason = subsystem("maturation")
    evidence_item = subsystems.get("forward_evidence") or {}
    evidence_status = str(evidence_item.get("detail_state") or health.get("forward_evidence_status") or "UNKNOWN")
    evidence_reason = str(evidence_item.get("reason") or "No persisted evidence-readiness state is available.")
    billing_status = "UNKNOWN"
    billing_reason = "Billing reachability is not part of the persisted health snapshot; use the manual billing check below."
    return [
        {"System": "Product", "Status": product_status, "Reason": product_reason},
        {"System": "Scanner", "Status": scanner_status, "Reason": scanner_reason},
        {"System": "Research", "Status": research_status, "Reason": research_reason},
        {"System": "Maturation", "Status": maturation_status, "Reason": maturation_reason},
        {"System": "Evidence", "Status": evidence_status, "Reason": evidence_reason},
        {"System": "Billing", "Status": billing_status, "Reason": billing_reason},
    ]


def _render_heatmap(groups: Sequence[Mapping[str, Any]], *, row_key: str, col_key: str,
                    metric_label: str, min_n: int) -> None:
    if not groups:
        st.info("No matured observations are available for this matrix.")
        return
    metric_map = {
        "Median Return": "median_return", "Average Return": "average_return",
        "Positive Outcome %": "positive_rate", "Observation Count": "n",
        "MFE": "average_mfe", "MAE": "average_mae",
    }
    metric = metric_map[metric_label]
    display_rows = []
    for group in groups:
        value = group.get(metric)
        sufficient = int(group.get("n") or 0) >= min_n
        if metric == "n":
            shown = str(int(value or 0))
        elif not sufficient:
            shown = f"INSUFFICIENT (n={int(group.get('n') or 0)})"
        elif value is None:
            shown = "—"
        else:
            shown = _pct(value)
        display_rows.append({row_key: group.get(row_key), col_key: group.get(col_key), "value": shown})
    frame = pd.DataFrame(display_rows).pivot(index=row_key, columns=col_key, values="value")
    st.dataframe(arrow_safe(frame), width="stretch")
    st.caption(f"Evidence threshold: n ≥ {min_n}. Every insufficient cell retains its sample count.")


def _matches(row: Mapping[str, Any], *, signal: str, cohort: str, horizon: str, session: str,
             score_range: tuple[int, int] | None = None) -> bool:
    score = ar.number(row.get("hsf_score"))
    score_matches = score_range is None or (score is not None and score_range[0] <= score <= score_range[1])
    return ((signal == "All" or row.get("signal") == signal)
            and (cohort == "All" or row.get("cohort") == cohort)
            and (horizon == "All" or row.get("horizon") == horizon)
            and (session == "All" or row.get("session") == session)
            and score_matches)


def _parse_time(value: Any) -> dt.datetime | None:
    return ar.parse_timestamp(value)


def _within_days(value: Any, days: int, now: dt.datetime) -> bool:
    parsed = _parse_time(value)
    return bool(parsed and now - dt.timedelta(days=days) <= parsed <= now + dt.timedelta(minutes=5))


def _daily_counts(users: Sequence[Mapping[str, Any]], key: str, days: int) -> pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    start = now.date() - dt.timedelta(days=max(0, days - 1))
    counts = Counter(
        parsed.date() for user in users
        if (parsed := _parse_time(user.get(key))) is not None and parsed.date() >= start
    )
    index = pd.date_range(start, now.date(), freq="D")
    new_users = [counts.get(day.date(), 0) for day in index]
    prior_users = sum(
        1 for user in users
        if (parsed := _parse_time(user.get(key))) is not None and parsed.date() < start
    )
    cumulative = []
    running = prior_users
    for count in new_users:
        running += count
        cumulative.append(running)
    return pd.DataFrame({"New users": new_users, "Cumulative users": cumulative}, index=index)


def _daily_run_counts(runs: Sequence[Mapping[str, Any]], days: int) -> pd.DataFrame:
    now = dt.datetime.now(dt.timezone.utc)
    start = now.date() - dt.timedelta(days=max(0, days - 1))
    scheduled, user = Counter(), Counter()
    for run in runs:
        parsed = _parse_time(run.get("created_at"))
        if parsed is None or parsed.date() < start:
            continue
        (user if run.get("username") else scheduled)[parsed.date()] += 1
    index = pd.date_range(start, now.date(), freq="D")
    return pd.DataFrame({
        "Scheduled scans": [scheduled.get(day.date(), 0) for day in index],
        "User scans": [user.get(day.date(), 0) for day in index],
    }, index=index)


def _pct(value: Any) -> str:
    number = ar.number(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def _pct_number(value: Any) -> float | None:
    number = ar.number(value)
    return round(number * 100, 3) if number is not None else None


def _display_time(value: Any) -> str:
    parsed = _parse_time(value)
    return parsed.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if parsed else "unavailable"


def _latest_timestamp(observations: Sequence[Mapping[str, Any]]) -> str:
    values = [value for row in observations if (value := _parse_time(row.get("timestamp"))) is not None]
    return _display_time(max(values)) if values else "unavailable"


def _last_maturation(health: Mapping[str, Any]) -> str:
    maturation = ((health.get("subsystems") or {}).get("maturation") or {})
    return _display_time((maturation.get("metrics") or {}).get("last_success") or maturation.get("last_updated"))
