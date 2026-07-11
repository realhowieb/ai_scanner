from __future__ import annotations

from pathlib import Path
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
APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EARNINGS_REFRESH_LIMIT = 250


def render_admin_tab(
    *,
    tab_admin: Any,
    username: str,
    db_status: str,
    admin_users: object,
    render_admin_users_panel: Callable[..., Any],
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
            _render_earnings_refresh(c2, get_db_conn)
            _render_db_integrity_checks(c3, get_db_conn)

        if bool(st.session_state.get("is_admin")):
            st.markdown("---")
            st.markdown("### 📊 Diagnostics")
            _render_billing_health_badge()
            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                _render_scan_errors_panel(get_db_conn)
            with diag_col2:
                _render_login_attempts_panel(get_db_conn)
            _render_ai_usage_panel()

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
    get_db_conn: Callable[[], Any],
) -> None:
    with column:
        refresh_limit = st.number_input(
            "Earnings refresh limit",
            min_value=25,
            max_value=12000,
            value=_earnings_refresh_limit_default(),
            step=25,
            key="admin_earnings_refresh_limit",
            help="Limits the admin refresh so flaky upstream earnings calls do not stall the app.",
        )
        if not st.button("📅 Refresh earnings now", key="admin_refresh_earnings", width="stretch"):
            return

        with st.spinner("Refreshing earnings calendar (admin)..."):
            conn = None
            try:
                symbols = _resolve_earnings_refresh_symbols(limit=int(refresh_limit))
                if not symbols:
                    st.warning(
                        "No symbols were available for earnings refresh. "
                        "Load a market universe or check the bundled universe files."
                    )
                    clear_earnings_result_cache()
                    return

                conn = get_db_conn()
                result = _populate_earnings(conn, symbols)
                attempted = _count_attempted_refreshes(result, fallback=len(symbols))
                dated = _count_refreshes_with_dates(result)

                if dated > 0:
                    st.success(
                        f"✅ Earnings refreshed: dates found for {dated} of {attempted} symbols."
                    )
                else:
                    st.warning(
                        f"Earnings refresh attempted {attempted} symbols, but the provider returned no dates. "
                        "Existing DB rows remain available; try a smaller limit later if Yahoo is rate-limiting."
                    )

                clear_earnings_result_cache()

            except ADMIN_TAB_ERRORS as exc:
                st.error("❌ Earnings refresh failed.")
                _show_exception(exc)
            finally:
                _close_conn(conn)


def _earnings_refresh_limit_default() -> int:
    try:
        raw_limit = int(st.session_state.get("admin_earnings_refresh_limit", DEFAULT_EARNINGS_REFRESH_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_EARNINGS_REFRESH_LIMIT
    return min(12000, max(25, raw_limit))


def _resolve_earnings_refresh_symbols(*, limit: int) -> list[str]:
    symbols: list[str] = []
    for key in ("sp500_universe", "nasdaq_universe", "combo_capped", "nasdaq_capped"):
        symbols.extend(_normalize_symbols(st.session_state.get(key)))

    symbols.extend(_read_symbols_file("sp500.txt"))
    symbols.extend(_read_symbols_file("nasdaq.txt"))
    symbols.extend(_normalize_symbols(st.session_state.get("active_watchlist_tickers")))
    return _dedupe_symbols(symbols)[: max(0, int(limit))]


def _normalize_symbols(raw_symbols: Any) -> list[str]:
    if not isinstance(raw_symbols, (list, tuple, set)):
        return []

    symbols: list[str] = []
    for raw in raw_symbols:
        sym = str(raw or "").strip().upper()
        if sym:
            symbols.append(sym)
    return symbols


def _read_symbols_file(filename: str) -> list[str]:
    try:
        text = (APP_ROOT / filename).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    return _normalize_symbols(text.splitlines())


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _count_attempted_refreshes(result: Any, *, fallback: int) -> int:
    if isinstance(result, dict):
        return len(result)
    try:
        return len(result)
    except TypeError:
        return int(fallback)


def _count_refreshes_with_dates(result: Any) -> int:
    values = result.values() if isinstance(result, dict) else result
    if not isinstance(values, (list, tuple, set)):
        try:
            values = list(values or [])
        except TypeError:
            return 0
    return sum(1 for item in values if _extract_earnings_date(item) is not None)


def _extract_earnings_date(item: Any) -> Any:
    if isinstance(item, dict):
        return item.get("earnings_date")
    return getattr(item, "earnings_date", None)


def _ensure_earnings_table(conn: Any) -> None:
    try:
        from db.earnings import ensure_earnings_table  # type: ignore

        ensure_earnings_table(conn)
    except ADMIN_TAB_ERRORS:
        return


def _populate_earnings(conn: Any, symbols: list[str]) -> Any:
    import db.earnings as earn_mod  # type: ignore

    populate = getattr(earn_mod, "populate_earnings_calendar", None)
    if not callable(populate):
        raise TypeError("populate_earnings_calendar is not callable")

    return populate(symbols, conn=conn)


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


def _render_ai_usage_panel() -> None:
    """Show which AI features are used most (last 30 days)."""
    st.markdown("#### AI feature usage (30d)")
    if not st.button("📈 Load AI usage", key="admin_load_ai_usage"):
        return
    try:
        from db.ai_usage import feature_usage_counts
        counts = feature_usage_counts(30)
        if not counts:
            st.info("No AI usage recorded yet.")
            return
        import pandas as pd
        df = pd.DataFrame(counts, columns=["feature", "calls"])
        st.dataframe(df)
        # Bar chart is best-effort: native charts pull in altair which can fail
        # on some runtimes. The table above already conveys the data.
        try:
            st.bar_chart(df.set_index("feature"))
        except ADMIN_TAB_ERRORS:
            pass
    except ADMIN_TAB_ERRORS as exc:
        st.error("Failed to load AI usage.")
        _show_exception(exc)


def _render_billing_health_badge() -> None:
    """Ping the billing service on demand and show a status badge."""
    if not st.button("🔌 Check billing service", key="admin_billing_health"):
        return
    try:
        import os

        import requests
        base = (os.getenv("BILLING_API_BASE") or "https://ai-scanner-h2c8.onrender.com").strip()
        timeout = float(os.getenv("BILLING_HEALTH_TIMEOUT", "3"))
        try:
            r = requests.get(f"{base}/health", timeout=timeout)
            ok = r.status_code == 200
        except ADMIN_TAB_ERRORS:
            ok = False
        if ok:
            st.success(f"✅ Billing service reachable ({base})")
        else:
            st.error(f"❌ Billing service unreachable — {base}/health did not respond. Subscriptions will not work.")
    except ADMIN_TAB_ERRORS:
        st.warning("⚠️ Could not check billing service health.")


def _render_scan_errors_panel(get_db_conn: Callable[[], Any]) -> None:
    st.markdown("#### Scan Errors (last 50)")
    if not st.button("🔍 Load scan errors", key="admin_load_scan_errors"):
        return
    conn = None
    try:
        conn = get_db_conn()
        from db.schema import ensure_neon_scan_errors_schema
        ensure_neon_scan_errors_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT occurred_at, context, username, ticker_count, error_type, message
                FROM scan_errors
                ORDER BY occurred_at DESC
                LIMIT 50
                """
            )
            rows = cur.fetchall()
        if not rows:
            st.success("No scan errors recorded.")
            return
        import pandas as pd
        df = pd.DataFrame(rows, columns=["occurred_at", "context", "username", "tickers", "error_type", "message"])
        st.dataframe(df)
        st.session_state["_admin_error_rows"] = rows
    except ADMIN_TAB_ERRORS as exc:
        st.error("Failed to load scan errors.")
        _show_exception(exc)
    finally:
        _close_conn(conn)

    # 🤖 AI triage of the loaded errors (admin-only; no per-user cap).
    rows_cached = st.session_state.get("_admin_error_rows")
    if rows_cached and st.button("🧠 AI triage these errors", key="admin_ai_triage"):
        try:
            from ui.ai import is_configured
            if not is_configured():
                st.info("AI triage needs ANTHROPIC_API_KEY configured.")
            else:
                from ui.ai_insights import triage_scan_errors
                with st.spinner("Triaging…"):
                    text, err = triage_scan_errors(rows_cached)
                if text:
                    st.markdown(text)
                else:
                    st.warning(err or "Could not triage errors.")
        except ADMIN_TAB_ERRORS as exc:
            _show_exception(exc)


def _render_login_attempts_panel(get_db_conn: Callable[[], Any]) -> None:
    st.markdown("#### Login Attempts (last 100)")
    if not st.button("🔍 Load login attempts", key="admin_load_login_attempts"):
        return
    conn = None
    try:
        conn = get_db_conn()
        from db.schema import ensure_neon_login_attempts_schema
        ensure_neon_login_attempts_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attempted_at, username, success, ip_address
                FROM login_attempts
                ORDER BY attempted_at DESC
                LIMIT 100
                """
            )
            rows = cur.fetchall()
        if not rows:
            st.success("No login attempts recorded.")
            return
        import pandas as pd
        df = pd.DataFrame(rows, columns=["attempted_at", "username", "success", "ip_address"])
        st.dataframe(df)

        failed = sum(1 for r in rows if not r[2])
        if failed:
            st.warning(f"{failed} failed login attempt(s) in the last 100 rows.")
    except ADMIN_TAB_ERRORS as exc:
        st.error("Failed to load login attempts.")
        _show_exception(exc)
    finally:
        _close_conn(conn)
