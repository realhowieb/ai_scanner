"""Kibana-style, read-only Admin Observation Explorer."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from analytics import admin_research as ar
from analytics import observation_explorer as oe
from db import observation_explorer as store
from ui.arrow_safe import arrow_safe


@st.cache_data(show_spinner=False, ttl=60)
def _options(filters: dict[str, Any]) -> dict[str, Any]:
    return store.query_options(filters)


@st.cache_data(show_spinner=False, ttl=60)
def _summary(filters: dict[str, Any]) -> dict[str, Any]:
    return store.query_summary(filters)


@st.cache_data(show_spinner=False, ttl=60)
def _timeline(filters: dict[str, Any], group_by: str) -> dict[str, Any]:
    return store.query_timeline(filters, group_by=group_by)


@st.cache_data(show_spinner=False, ttl=60)
def _page(filters: dict[str, Any], page: int, page_size: int) -> dict[str, Any]:
    return store.query_page(filters, page=page, page_size=page_size)


@st.cache_data(show_spinner=False, ttl=60)
def _comparison(filters: dict[str, Any]) -> dict[str, Any]:
    return store.query_cohort_comparison(filters)


@st.cache_data(show_spinner=False, ttl=60)
def _integrity(filters: dict[str, Any]) -> dict[str, Any]:
    return store.query_integrity(filters)


@st.cache_data(show_spinner=False, ttl=60)
def _histogram(filters: dict[str, Any]) -> dict[str, Any]:
    return store.query_histogram(filters)


@st.cache_data(show_spinner=False, ttl=60)
def _score_analysis(filters: dict[str, Any]) -> dict[str, Any]:
    return store.query_score_analysis(filters)


@st.cache_data(show_spinner=False, ttl=60)
def _detail(observation_id: str) -> dict[str, Any]:
    return store.query_detail(observation_id)


def render_observation_explorer(*, health: Mapping[str, Any] | None = None) -> None:
    if not bool(st.session_state.get("is_admin")):
        st.info("Observation Explorer is only available to admin users.")
        return

    st.markdown("### Observation Explorer")
    st.caption("Filter, summarize and inspect immutable HSF research observations. Read-only.")
    _render_saved_views()

    dataset_label = st.radio(
        "Dataset", ["Scanner Research", "Stair-Stepper"], horizontal=True,
        key="oe_dataset_label",
    )
    dataset = oe.DATASET_SCANNER if dataset_label == "Scanner Research" else oe.DATASET_STAIR_STEPPER
    epoch_spec, mixed = _render_epoch_and_time(dataset)
    start, end = _resolve_bounds(epoch_spec)
    option_filters = {
        "dataset": dataset, "start": start, "end": end,
        "compatible_control_design": epoch_spec.get("control_design") if epoch_spec else None,
    }
    options = _options(option_filters)
    if options.get("errors"):
        st.info("Some filter values could not be loaded. Typed filters and available values still work.")

    filters = _render_filter_bar(dataset, epoch_spec, start, end, options)
    if mixed:
        st.warning("MIXED RESEARCH DESIGNS — DIAGNOSTIC VIEW. Effectiveness comparisons are not promoted as evidence.")
    elif epoch_spec:
        st.caption(
            f"Research Epoch: {epoch_spec['label']} · start {epoch_spec['start']} · "
            f"control design {epoch_spec.get('control_design') or 'legacy / untagged'}"
        )

    with st.spinner("Querying observations..."):
        summary = _summary(filters)
    if summary.get("error"):
        st.warning("Observation data is temporarily unavailable.")
        st.caption(f"Query status: {summary['error']}")
        return

    _render_query_summary(summary, dataset)
    if int(summary.get("observations") or 0) == 0:
        if epoch_spec and epoch_spec.get("current"):
            st.info("No observations exist for the current research epoch yet.")
        else:
            st.info("No observations match these filters.")
        _render_freshness(summary, health, query_ms=summary.get("duration_ms"))
        return

    group_by = st.selectbox("Timeline grouping", ["total", "cohort", "signal", "status"],
                            key="oe_timeline_group")
    timeline = _timeline(filters, group_by)
    _render_timeline(timeline)
    _render_cohort_distribution(summary, dataset)
    _render_outcomes(filters, summary)
    _render_score_views(summary, filters)
    _render_comparison(filters, mixed=mixed)
    _render_integrity(filters, epoch_spec, dataset)
    _render_observation_table(filters)
    _render_drilldowns(filters, summary)
    _render_export(filters, int(summary.get("observations") or 0))
    query_ms = sum(float(item or 0) for item in (
        summary.get("duration_ms"), timeline.get("duration_ms"), options.get("duration_ms")
    ))
    _render_freshness(summary, health, query_ms=query_ms)


def _render_saved_views() -> None:
    st.caption("Saved views")
    labels = [
        "Current Evidence", "High Score Candidates", "Near Miss Analysis",
        "Controls", "Pending Maturation", "Research Integrity",
    ]
    columns = st.columns(len(labels))
    for column, label in zip(columns, labels):
        if column.button(label, key=f"oe_preset_{label}", width="stretch"):
            _apply_preset(label)
            st.rerun()


def _apply_preset(label: str) -> None:
    st.session_state["oe_dataset_label"] = "Scanner Research"
    st.session_state["oe_epoch_label"] = "Current — Run 59B"
    st.session_state["oe_time"] = "Current Forward Epoch"
    st.session_state["oe_cohorts"] = []
    st.session_state["oe_status"] = "All"
    if label == "Current Evidence":
        st.session_state["oe_status"] = "Matured"
    elif label == "High Score Candidates":
        st.session_state["oe_cohorts"] = ["CANDIDATE"]
        st.session_state["oe_score_min"] = 80.0
    elif label == "Near Miss Analysis":
        st.session_state["oe_cohorts"] = ["NEAR_MISS"]
    elif label == "Controls":
        st.session_state["oe_cohorts"] = ["CONTROL"]
    elif label == "Pending Maturation":
        st.session_state["oe_status"] = "Pending"


def _render_epoch_and_time(dataset: str) -> tuple[dict[str, Any] | None, bool]:
    if dataset == oe.DATASET_STAIR_STEPPER:
        time_value = st.selectbox("Time", ["Today", "7D", "30D", "90D", "Custom"], index=2, key="oe_time_stair")
        st.session_state["oe_effective_time"] = time_value
        return None, False
    specs = oe.epoch_specs()
    labels = [spec["label"] for spec in reversed(specs)] + ["All epochs — diagnostic"]
    selected = st.selectbox("Research epoch", labels, key="oe_epoch_label")
    time_value = st.selectbox(
        "Time", ["Today", "7D", "30D", "90D", "Current Forward Epoch", "Previous Forward Epoch", "Custom"],
        index=4, key="oe_time",
    )
    st.session_state["oe_effective_time"] = time_value
    if time_value == "Current Forward Epoch":
        return specs[-1], False
    if time_value == "Previous Forward Epoch":
        return specs[-2], False
    if selected == "All epochs — diagnostic":
        return None, True
    spec = next(item for item in specs if item["label"] == selected)
    return spec, False


def _resolve_bounds(epoch_spec: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    now = dt.datetime.now(dt.timezone.utc)
    time_value = st.session_state.get("oe_effective_time") or "30D"
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    if time_value == "Today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_value in {"7D", "30D", "90D"}:
        start = now - dt.timedelta(days=int(time_value[:-1]))
    elif time_value == "Current Forward Epoch":
        current = oe.epoch_specs()[-1]
        start = ar.parse_timestamp(current["start"])
    elif time_value == "Previous Forward Epoch":
        previous = oe.epoch_specs()[-2]
        start, end = ar.parse_timestamp(previous["start"]), ar.parse_timestamp(previous["end"])
    elif time_value == "Custom":
        default_start = (now - dt.timedelta(days=7)).date()
        custom = st.date_input("Custom dates", value=(default_start, now.date()), key="oe_custom_dates")
        if isinstance(custom, (list, tuple)) and custom:
            start = dt.datetime.combine(custom[0], dt.time.min, tzinfo=dt.timezone.utc)
            if len(custom) > 1:
                end = dt.datetime.combine(custom[1] + dt.timedelta(days=1), dt.time.min, tzinfo=dt.timezone.utc)
    if epoch_spec:
        epoch_start = ar.parse_timestamp(epoch_spec.get("start"))
        epoch_end = ar.parse_timestamp(epoch_spec.get("end"))
        start = max(value for value in (start, epoch_start) if value is not None)
        ends = [value for value in (end, epoch_end) if value is not None]
        end = min(ends) if ends else None
    return start.isoformat() if start else None, end.isoformat() if end else None


def _render_filter_bar(dataset: str, epoch_spec: Mapping[str, Any] | None, start: str | None,
                       end: str | None, options: Mapping[str, Any]) -> dict[str, Any]:
    st.markdown("#### Filters")
    row1 = st.columns(4)
    status = row1[0].selectbox(
        "Observation status", ["All", "Pending", "Matured", "Usable", "Excluded"], key="oe_status",
    )
    cohort_options = options.get("cohorts") or list(oe.CANONICAL_COHORTS)
    cohorts = row1[1].multiselect("Research cohort", cohort_options, key="oe_cohorts",
                                 disabled=dataset == oe.DATASET_STAIR_STEPPER)
    symbols = oe.normalize_symbols(row1[2].text_input("Symbols", key="oe_symbols", placeholder="AAPL, MSFT"))
    signal_options = ["All"] + list(options.get("signals") or [])
    signal = row1[3].selectbox("Signal / classification", signal_options, key="oe_signal")

    row2 = st.columns(4)
    horizons = ["All"] + list(options.get("horizons") or [])
    horizon = row2[0].selectbox(
        "Horizon", horizons, index=1 if len(horizons) > 1 else 0, key="oe_horizon",
    )
    scan_id = row2[1].text_input("Scan ID", key="oe_scan_id")
    sessions = ["All"] + list(options.get("sessions") or [])
    session = row2[2].selectbox("Session", sessions, key="oe_session")
    designs = ["All"] + list(options.get("control_designs") or [])
    control_design = row2[3].selectbox("Control design", designs, key="oe_control_design",
                                      disabled=dataset == oe.DATASET_STAIR_STEPPER)

    row3 = st.columns(3)
    score_enabled = bool(options.get("score_available"))
    score_min = row3[0].number_input("HSF Score minimum", min_value=0.0, max_value=100.0,
                                     value=0.0, step=1.0, key="oe_score_min", disabled=not score_enabled)
    score_max = row3[1].number_input("HSF Score maximum", min_value=0.0, max_value=100.0,
                                     value=100.0, step=1.0, key="oe_score_max", disabled=not score_enabled)
    if not score_enabled:
        row3[2].caption("Canonical HSF Score is not persisted for this selection; score filtering is unavailable.")
    compatible = epoch_spec.get("control_design") if epoch_spec and dataset == oe.DATASET_SCANNER else None
    return {
        "dataset": dataset, "start": start, "end": end,
        "status": status, "cohorts": cohorts, "symbols": symbols,
        "signal": None if signal == "All" else signal,
        "horizon": None if horizon == "All" else horizon,
        "scan_id": scan_id.strip() or None,
        "session": None if session == "All" else session,
        "control_design": None if control_design == "All" else control_design,
        "compatible_control_design": compatible,
        "score_min": score_min if score_enabled and score_min > 0 else None,
        "score_max": score_max if score_enabled and score_max < 100 else None,
    }


def _render_query_summary(summary: Mapping[str, Any], dataset: str) -> None:
    st.markdown("#### Matching observations")
    top = st.columns(5)
    top[0].metric("Observations", f"{int(summary.get('observations') or 0):,}")
    top[1].metric("Matured", f"{int(summary.get('matured') or 0):,}")
    top[2].metric("Usable", f"{int(summary.get('usable') or 0):,}")
    top[3].metric("Pending", f"{int(summary.get('pending') or 0):,}")
    top[4].metric("Excluded", f"{int(summary.get('excluded') or 0):,}")
    bottom = st.columns(4)
    if dataset == oe.DATASET_SCANNER:
        bottom[0].metric("Candidate", f"{int(summary.get('candidate') or 0):,}")
        bottom[1].metric("Near Miss", f"{int(summary.get('near_miss') or 0):,}")
        bottom[2].metric("Control", f"{int(summary.get('control') or 0):,}")
    bottom[3].metric("Positive", _pct(summary.get("positive_rate")))
    st.caption(
        f"Median score {_number(summary.get('median_score'), decimals=1)} · "
        f"Median return {_pct(summary.get('median_return'))} (usable N={int(summary.get('usable') or 0):,}) · "
        f"Mean return {_pct(summary.get('mean_return'))}"
    )


def _render_timeline(result: Mapping[str, Any]) -> None:
    st.markdown("#### Observations over time")
    rows = result.get("rows") or []
    if not rows:
        st.info("No timeline data is available for this selection.")
        return
    frame = pd.DataFrame(rows)
    chart = frame.pivot_table(index="bucket", columns="series", values="count", aggfunc="sum", fill_value=0)
    st.line_chart(chart, height=280)


def _render_cohort_distribution(summary: Mapping[str, Any], dataset: str) -> None:
    if dataset != oe.DATASET_SCANNER:
        return
    st.markdown("#### Cohort distribution")
    rows = [{"Cohort": label, "Observations": int(summary.get(key) or 0)} for label, key in (
        ("Candidate", "candidate"), ("Near Miss", "near_miss"), ("Control", "control"),
    )]
    st.bar_chart(pd.DataFrame(rows).set_index("Cohort"), height=240)
    drill = st.selectbox("Cohort drill-down", ["None"] + [row["Cohort"].upper().replace(" ", "_") for row in rows],
                         key="oe_cohort_drill")
    if drill != "None" and st.button("Apply cohort filter", key="oe_apply_cohort"):
        st.session_state["oe_cohorts"] = [drill]
        st.rerun()


def _render_outcomes(filters: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    st.markdown("#### Forward return distribution")
    if not filters.get("horizon"):
        st.info("Select a horizon to compare forward outcomes without mixing measurement windows.")
        return
    if not int(summary.get("matured") or 0):
        st.info("No matured observations exist for this selection.")
        return
    hist = _histogram(dict(filters))
    rows = hist.get("rows") or []
    if rows:
        frame = pd.DataFrame(rows)
        frame["Forward return %"] = frame["bucket"].astype(float) * 100
        st.bar_chart(frame.set_index("Forward return %")[["count"]], height=280)
    metrics = st.columns(6)
    metrics[0].metric("N", f"{int(summary.get('usable') or 0):,}")
    metrics[1].metric("Median", _pct(summary.get("median_return")))
    metrics[2].metric("Mean", _pct(summary.get("mean_return")))
    metrics[3].metric("Positive", _pct(summary.get("positive_rate")))
    metrics[4].metric("P25", _pct(summary.get("p25_return")))
    metrics[5].metric("P75", _pct(summary.get("p75_return")))
    excursion = st.columns(2)
    excursion[0].metric("Median MFE", _pct(summary.get("median_mfe")))
    excursion[1].metric("Median MAE", _pct(summary.get("median_mae")))


def _render_score_views(summary: Mapping[str, Any], filters: Mapping[str, Any] | None = None) -> None:
    st.markdown("#### HSF Score distribution and outcome relationship")
    if summary.get("median_score") is None:
        st.info(
            "Canonical HSF Score is not stored in these observations. Score distribution and "
            "HSF Score vs Forward Outcome are unavailable; scanner model scores are not substituted."
        )
        return
    result = _score_analysis(dict(filters or {}))
    rows = result.get("rows") or []
    if not rows:
        st.info("No scored observations match this selection.")
        return
    frame = pd.DataFrame(rows)
    st.bar_chart(frame.set_index("bucket")[["observations"]], height=240)
    table = frame.rename(columns={
        "bucket": "HSF Score", "observations": "N", "matured": "Matured",
        "mean_return": "Mean Return", "median_return": "Median Return",
        "positive_rate": "Positive %",
    })
    for column in ("Mean Return", "Median Return", "Positive %"):
        table[column] = table[column].map(_pct)
    st.dataframe(
        arrow_safe(table[["HSF Score", "N", "Matured", "Mean Return", "Median Return", "Positive %"]]),
        width="stretch", hide_index=True,
    )
    st.caption("Bucketed association only; this view does not establish causality.")


def _render_comparison(filters: Mapping[str, Any], *, mixed: bool) -> None:
    if filters.get("dataset") != oe.DATASET_SCANNER:
        return
    st.markdown("#### Candidate → Near Miss → Control")
    if mixed:
        st.info("Comparison is disabled for mixed research designs. Select one compatible epoch.")
        return
    comparison = _comparison(dict(filters))
    rows = comparison.get("rows") or []
    if not rows:
        st.info("No cohort comparison is available for this selection.")
        return
    table = []
    for row in rows:
        n = int(row.get("usable") or 0)
        sufficient = n >= ar.DEFAULT_MIN_EVIDENCE_N
        table.append({
            "Cohort": row.get("cohort"), "N": row.get("n"),
            "Matured": int(row.get("matured") or 0), "Usable": n,
            "Positive %": _pct(row.get("positive_rate")) if sufficient else f"INSUFFICIENT (N={n})",
            "Median Return": _pct(row.get("median_return")) if sufficient else "INSUFFICIENT",
            "Mean Return": _pct(row.get("mean_return")) if sufficient else "INSUFFICIENT",
            "Median MFE": _pct(row.get("median_mfe")) if sufficient else "INSUFFICIENT",
            "Median MAE": _pct(row.get("median_mae")) if sufficient else "INSUFFICIENT",
        })
    st.dataframe(arrow_safe(pd.DataFrame(table)), width="stretch", hide_index=True)
    st.caption(f"No winner is identified below the shared evidence threshold N={ar.DEFAULT_MIN_EVIDENCE_N}.")


def _render_integrity(filters: Mapping[str, Any], epoch_spec: Mapping[str, Any] | None, dataset: str) -> None:
    if dataset != oe.DATASET_SCANNER:
        return
    st.markdown("#### Research cohort integrity")
    result = _integrity(dict(filters))
    rows = pd.DataFrame([{
        "Candidate ↔ Near Miss": result.get("candidate_near_miss", 0),
        "Candidate ↔ Control": result.get("candidate_control", 0),
        "Near Miss ↔ Control": result.get("near_miss_control", 0),
        "Control Design": epoch_spec.get("control_design") if epoch_spec else "MIXED",
        "Epoch": epoch_spec.get("label") if epoch_spec else "All — diagnostic",
        "Status": result.get("status"),
    }])
    st.dataframe(arrow_safe(rows), width="stretch", hide_index=True)
    if result.get("status") == "OVERLAP_DETECTED":
        st.error("RESEARCH COHORT OVERLAP DETECTED")
    elif result.get("status") == "UNKNOWN":
        st.warning("Research cohort integrity could not be evaluated for this selection.")
    else:
        st.success("Research cohort integrity: PASS")


def _render_observation_table(filters: Mapping[str, Any]) -> None:
    st.markdown("#### Observations")
    p1, p2 = st.columns(2)
    page_size = int(p1.selectbox("Rows per page", [25, 50, 100], index=1, key="oe_page_size"))
    page_number = int(p2.number_input("Page", min_value=1, value=1, step=1, key="oe_page"))
    result = _page(dict(filters), page_number, page_size)
    rows = result.get("rows") or []
    if not rows:
        st.info("No observations are available on this page.")
        return
    summaries = [oe.observation_summary(row, horizon=filters.get("horizon")) for row in rows]
    display = pd.DataFrame([{
        "Timestamp": row["timestamp"], "Symbol": row["symbol"], "Cohort": row["cohort"],
        "Signal": row["signals"], "HSF Score": row["hsf_score"], "Rank": row["rank"],
        "Scan ID": row["scan_id"], "Research Epoch": row["research_epoch"],
        "Control Design": row["control_design"], "Status": row["status"], "Horizon": row["horizon"],
        "Forward Return": row["forward_return"], "MFE": row["mfe"], "MAE": row["mae"],
        "Matured At": row["matured_at"],
    } for row in summaries])
    st.dataframe(arrow_safe(display), width="stretch", hide_index=True, height=420)
    pages = max(1, math.ceil(int(result.get("total") or 0) / page_size))
    st.caption(f"Page {result.get('page')} of {pages} · {int(result.get('total') or 0):,} matching observations · "
               f"query {result.get('duration_ms') or '—'} ms")
    choices = {f"{row['symbol']} · {row['timestamp']} · {row['observation_id']}": row["observation_id"] for row in summaries}
    selected = st.selectbox("Inspect observation", ["None"] + list(choices), key="oe_detail_selection")
    if selected != "None":
        _render_detail(choices[selected])


def _render_detail(observation_id: str) -> None:
    detail = _detail(observation_id)
    observation = detail.get("observation")
    if not observation:
        st.info("Observation detail is unavailable.")
        return
    summary = oe.observation_summary(observation)
    with st.expander(f"Observation · {summary['symbol']} · {observation_id}", expanded=True):
        st.dataframe(arrow_safe(pd.DataFrame([{
            "Symbol": summary["symbol"], "Captured": summary["timestamp"], "Scan": summary["scan_id"],
            "Cohort": summary["cohort"], "Epoch": summary["research_epoch"],
            "HSF Score": summary["hsf_score"], "Rank": summary["rank"], "Signal": summary["signals"],
            "Status": summary["status"],
        }])), width="stretch", hide_index=True)
        st.markdown("**Capture context**")
        st.json({"market": observation.get("market") or {}, "indicators": observation.get("indicators") or {},
                 "session": observation.get("session"), "market_context": observation.get("market_context") or {}})
        st.markdown("**Research metadata**")
        st.json(observation.get("research_metadata") or {})
        outcomes = []
        for horizon, value in (observation.get("outcomes") or {}).items():
            if not isinstance(value, Mapping):
                continue
            outcomes.append({"Horizon": horizon, "Status": value.get("data_status"),
                             "Return": value.get("directional_return") if value.get("directional_return") is not None else value.get("raw_return"),
                             "MFE": value.get("mfe"), "MAE": value.get("mae"),
                             "Matured At": value.get("evaluation_time")})
        st.markdown("**Outcomes**")
        if outcomes:
            st.dataframe(arrow_safe(pd.DataFrame(outcomes)), width="stretch", hide_index=True)
        else:
            st.info("No outcomes have matured for this observation.")


def _render_drilldowns(filters: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    symbols = filters.get("symbols") or []
    if filters.get("scan_id"):
        st.markdown("#### Scan drill-down")
        st.write(f"**Scan:** {filters['scan_id']}")
        st.caption(f"{int(summary.get('observations') or 0):,} observations · "
                   f"{int(summary.get('matured') or 0):,} matured · {int(summary.get('pending') or 0):,} pending")
    if len(symbols) == 1:
        st.markdown("#### Symbol observation history")
        st.write(f"**{symbols[0]}** · {int(summary.get('observations') or 0):,} observations")
        st.caption(f"Candidate {int(summary.get('candidate') or 0):,} · Near Miss {int(summary.get('near_miss') or 0):,} · "
                   f"Control {int(summary.get('control') or 0):,} · Matured {int(summary.get('matured') or 0):,} · "
                   f"Pending {int(summary.get('pending') or 0):,}")


def _render_export(filters: Mapping[str, Any], total: int) -> None:
    st.markdown("#### Export")
    if st.button("Prepare filtered observations CSV", key="oe_prepare_export"):
        result = store.query_export(filters, limit=oe.MAX_EXPORT_ROWS)
        csv_result = oe.bounded_csv(result.get("rows") or [], maximum=oe.MAX_EXPORT_ROWS)
        st.session_state["oe_export"] = {**csv_result, "truncated": result.get("truncated") or csv_result["truncated"]}
    export = st.session_state.get("oe_export")
    if export:
        st.download_button("Download filtered observations CSV", export["csv"],
                           file_name="hsf_observation_explorer.csv", mime="text/csv", key="oe_download")
        if export.get("truncated") or total > oe.MAX_EXPORT_ROWS:
            st.warning(f"Export is capped at {oe.MAX_EXPORT_ROWS:,} rows and was truncated.")


def _render_freshness(summary: Mapping[str, Any], health: Mapping[str, Any] | None, *, query_ms: Any) -> None:
    health = health or {}
    maturation = ((health.get("subsystems") or {}).get("maturation") or {})
    last_maturation = (maturation.get("metrics") or {}).get("last_success") or maturation.get("last_updated")
    st.caption(
        f"Latest observation: {_time(summary.get('latest_observation'))} · "
        f"Latest matured observation: {_time(summary.get('latest_matured'))} · "
        f"Last maturation run: {_time(last_maturation)} · Query executed: {_time(dt.datetime.now(dt.timezone.utc))} · "
        f"measured query time {round(float(query_ms or 0), 1)} ms"
    )


def _pct(value: Any) -> str:
    number = ar.number(value)
    return "—" if number is None else f"{number * 100:+.2f}%"


def _number(value: Any, *, decimals: int = 0) -> str:
    number = ar.number(value)
    return "—" if number is None else f"{number:.{decimals}f}"


def _time(value: Any) -> str:
    parsed = ar.parse_timestamp(value)
    return parsed.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if parsed else "unavailable"
