from __future__ import annotations

from typing import Any, Callable

import streamlit as st


ADMIN_TAB_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    OSError,
    ImportError,
)


def render_admin_tab(
    *,
    tab_admin: Any,
    username: str,
    db_status: str,
    admin_users: object,
    render_admin_users_panel: Callable[..., Any],
    fetch_earnings_this_week: Callable[..., Any],
    get_db_conn: Callable[[], Any],
) -> None:
    with tab_admin:
        st.markdown("## 🛠 Admin Panel")

        try:
            render_admin_users_panel(
                username=username,
                ADMIN_USERS=admin_users,
                db_status=db_status,
            )
        except ADMIN_TAB_ERRORS as exc:
            st.error("Admin panel failed to render.")
            _show_exception(exc)

        st.markdown("---")
        st.markdown("### 🧰 Admin Tools")

        if not bool(st.session_state.get("is_admin")):
            st.info("🔒 Admin tools are only available to admin users.")
        else:
            c1, c2, c3 = st.columns(3)
            _render_force_tier_resync(c1)
            _render_earnings_refresh(c2, fetch_earnings_this_week, get_db_conn)
            _render_db_integrity_checks(c3, get_db_conn)

        st.caption(
            "Tip: Earnings refresh is intentionally admin-only and never runs automatically during scans."
        )


def _show_exception(exc: BaseException) -> None:
    try:
        st.exception(exc)
    except ADMIN_TAB_ERRORS:
        st.caption(f"{type(exc).__name__}: {exc}")


def _render_force_tier_resync(column: Any) -> None:
    with column:
        if st.button("🔄 Force tier resync", key="admin_force_tier_resync", width="stretch"):
            for key in (
                "entitlements",
                "tier",
                "tier_key",
                "profile_loaded_for_user",
                "pricing_focus",
            ):
                st.session_state.pop(key, None)
            st.success("Tier resync requested (session caches cleared). Reloading...")
            st.rerun()


def _render_earnings_refresh(
    column: Any,
    fetch_earnings_this_week: Callable[..., Any],
    get_db_conn: Callable[[], Any],
) -> None:
    with column:
        if not st.button("📅 Refresh earnings now", key="admin_refresh_earnings", width="stretch"):
            return

        with st.spinner("Refreshing earnings calendar (admin)..."):
            conn = None
            refreshed = 0
            try:
                earnings_list = _fetch_earnings_list(fetch_earnings_this_week)
                if not earnings_list:
                    st.info(
                        "Earnings refresh completed, but the upstream source returned 0 items. "
                        "(This can happen on weekends/holidays or if the provider is unavailable.)"
                    )
                    clear_earnings_result_cache()
                    st.stop()

                conn = get_db_conn()
                _ensure_earnings_table(conn)

                try:
                    _populate_earnings(conn, earnings_list)
                    refreshed = len(earnings_list)
                except ADMIN_TAB_ERRORS as exc:
                    st.warning(f"Earnings refresh ran, but upsert failed: {type(exc).__name__}: {exc}")

                if refreshed > 0:
                    st.success(f"✅ Earnings refreshed: upsert attempted for {refreshed} symbols.")
                else:
                    st.info(
                        "Earnings refresh completed, but no rows were upserted. "
                        "(This can happen if the DB helper is unavailable or rejected all symbols.)"
                    )

                clear_earnings_result_cache()

            except ADMIN_TAB_ERRORS as exc:
                st.error("❌ Earnings refresh failed.")
                _show_exception(exc)
            finally:
                _close_conn(conn)


def _fetch_earnings_list(fetch_earnings_this_week: Callable[..., Any]) -> list[Any]:
    try:
        return list(fetch_earnings_this_week() or [])
    except ADMIN_TAB_ERRORS:
        return []


def _ensure_earnings_table(conn: Any) -> None:
    try:
        from db.earnings import ensure_earnings_table  # type: ignore

        ensure_earnings_table(conn)
    except ADMIN_TAB_ERRORS:
        return


def _populate_earnings(conn: Any, earnings_list: list[Any]) -> None:
    import db.earnings as earn_mod  # type: ignore

    populate = getattr(earn_mod, "populate_earnings_calendar", None)
    if not callable(populate):
        raise TypeError("populate_earnings_calendar is not callable")

    try:
        populate(conn, earnings_list)
    except TypeError:
        try:
            populate(earnings_list)
        except TypeError:
            try:
                populate(conn=conn, earnings_info_list=earnings_list)
            except TypeError:
                populate(conn=conn, earnings_list=earnings_list)


def _render_db_integrity_checks(column: Any, get_db_conn: Callable[[], Any]) -> None:
    with column:
        if not st.button("🧪 DB integrity checks", key="admin_db_integrity", width="stretch"):
            return

        with st.spinner("Running DB checks..."):
            conn = None
            try:
                conn = get_db_conn()
                issues: list[str] = []
                earnings_count, bad_symbol_count = _collect_db_integrity(conn, issues)

                st.markdown("#### ✅ DB Check Report")
                if earnings_count is not None:
                    st.write(f"Earnings rows: **{int(earnings_count)}**")
                if bad_symbol_count is not None:
                    st.write(f"Symbol hygiene issues: **{int(bad_symbol_count)}**")

                if issues:
                    st.warning("Issues / notes:")
                    for msg in issues:
                        st.write(f"- {msg}")
                else:
                    st.success("All checks passed.")

            except ADMIN_TAB_ERRORS as exc:
                st.error("❌ DB checks failed.")
                _show_exception(exc)
            finally:
                _close_conn(conn)


def _collect_db_integrity(conn: Any, issues: list[str]) -> tuple[Any, Any]:
    earnings_count = None
    bad_symbol_count = None

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            _ = cur.fetchone()
    except ADMIN_TAB_ERRORS as exc:
        issues.append(f"DB connectivity failed: {type(exc).__name__}: {exc}")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = 'earnings_calendar';
                """
            )
            exists = (cur.fetchone() or [0])[0]

            if not exists:
                issues.append("earnings_calendar table is missing.")
            else:
                cur.execute("SELECT COUNT(*) FROM earnings_calendar;")
                earnings_count = (cur.fetchone() or [0])[0]

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM earnings_calendar
                    WHERE symbol IS NULL
                       OR symbol <> UPPER(TRIM(symbol))
                       OR symbol = '';
                    """
                )
                bad_symbol_count = (cur.fetchone() or [0])[0]

        if bad_symbol_count and int(bad_symbol_count) > 0:
            _normalize_earnings_symbols(conn, int(bad_symbol_count), issues)
    except ADMIN_TAB_ERRORS as exc:
        issues.append(f"Earnings table checks failed: {type(exc).__name__}: {exc}")

    return earnings_count, bad_symbol_count


def _normalize_earnings_symbols(conn: Any, bad_symbol_count: int, issues: list[str]) -> None:
    try:
        from db.earnings import ensure_earnings_table  # type: ignore

        ensure_earnings_table(conn)
        issues.append(
            f"Found {bad_symbol_count} non-normalized symbols; ran normalization UPDATE."
        )
    except ADMIN_TAB_ERRORS:
        issues.append(
            f"Found {bad_symbol_count} non-normalized symbols; normalization function unavailable."
        )


def _close_conn(conn: Any) -> None:
    try:
        if conn is not None:
            conn.close()
    except ADMIN_TAB_ERRORS:
        return


def clear_earnings_result_cache() -> None:
    st.session_state.pop("earnings_enriched_df", None)
    st.session_state.pop("earnings_enriched_signature", None)
    st.session_state.pop("earnings_enrichment_rerun_sig", None)
