import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class CleanupSourceChecks(unittest.TestCase):
    def test_price_fetcher_logs_provider_summary(self):
        source = (ROOT / "data" / "prices.py").read_text()
        fetch_source = (ROOT / "data" / "fetch.py").read_text()
        helper_source = (ROOT / "data" / "price_utils.py").read_text()
        alpaca_source = (ROOT / "data" / "price_alpaca.py").read_text()

        self.assertIn("summarize_provider_skips", source)
        self.assertIn("[prices] {summary.message}", source)
        self.assertIn("_YFINANCE_BASE_ERRORS = (RuntimeError, TimeoutError, ConnectionError, OSError, ValueError)", source)
        self.assertIn('"YFRateLimitError"', source)
        self.assertIn("def _build_yfinance_errors", source)
        self.assertIn("except _YFINANCE_ERRORS as e", source)
        self.assertNotIn("except Exception", source)
        self.assertIn("from .price_utils import", source)
        self.assertIn("from .price_alpaca import", source)
        self.assertNotIn("def _normalize_df", source)
        self.assertNotIn("def _frame_fingerprint", source)
        self.assertNotIn("def _download_multi_alpaca", source)
        self.assertIn("_YAHOO_HTTP_ERRORS = (requests.RequestException, ValueError, KeyError, TypeError)", fetch_source)
        self.assertIn('"YFRateLimitError"', fetch_source)
        self.assertIn("def _build_yfinance_errors", fetch_source)
        self.assertIn("except _YAHOO_HTTP_ERRORS as e", fetch_source)
        self.assertIn("except ImportError", fetch_source)
        self.assertNotIn("except Exception", fetch_source)
        self.assertIn("def normalize_price_frame", helper_source)
        self.assertIn("def frame_fingerprint", helper_source)
        self.assertIn("def download_multi_alpaca", alpaca_source)
        self.assertIn("except requests_exc.RequestException as exc", alpaca_source)
        self.assertIn("except ValueError as exc", alpaca_source)
        self.assertIn("logger.warning", alpaca_source)

    def test_scan_engine_surfaces_provider_diagnostics(self):
        source = (ROOT / "scan" / "engine.py").read_text()

        self.assertIn("provider_skipped", source)
        self.assertIn("Provider skip sample", source)
        self.assertIn("Price provider warning", source)
        self.assertIn("_ENGINE_BOUNDARY_ERRORS = (", source)
        self.assertIn("_STREAMLIT_UI_ERRORS = (RuntimeError, TypeError, ValueError, AttributeError)", source)
        self.assertIn("except _ENGINE_BOUNDARY_ERRORS as e", source)
        self.assertIn("except _STREAMLIT_UI_ERRORS", source)
        self.assertEqual(source.count("except Exception"), 1)

    def test_ci_has_dependency_import_smoke_job(self):
        source = (ROOT / ".github" / "workflows" / "smoke.yml").read_text()
        pyproject_source = (ROOT / "pyproject.toml").read_text()
        dev_requirements = (ROOT / "requirements-dev.txt").read_text()

        self.assertIn("core-dependency-import-smoke", source)
        self.assertIn("full-dependency-import-smoke", source)
        self.assertIn("ruff check .", source)
        self.assertIn("python -m pip install -r requirements-dev.txt", source)
        self.assertIn("ruff", dev_requirements)
        self.assertIn("[tool.ruff.lint]", pyproject_source)
        self.assertIn("python -m pip install --prefer-binary -r requirements-core.txt", source)
        self.assertIn("Run dependency-backed unit tests", source)
        self.assertIn("python -m pip install --prefer-binary -r requirements.txt", source)
        self.assertIn("import scan.engine", source)
        self.assertIn("Live Streamlit startup smoke", source)
        self.assertIn("Deployment readiness doctor", source)
        self.assertIn("python scripts/streamlit_smoke.py --timeout 60", source)

    def test_scheduled_scan_workflow_uses_core_deps_and_runtime_secret_names(self):
        source = (ROOT / ".github" / "workflows" / "scheduled-scans.yml").read_text()

        self.assertIn("python -m pip install --prefer-binary -r requirements-core.txt", source)
        self.assertNotIn("pip install -r requirements.txt", source)
        self.assertIn("ALPACA_API_KEY_ID", source)
        self.assertIn("ALPACA_API_SECRET_KEY", source)
        self.assertIn("NEON_DATABASE_URL", source)
        self.assertIn('AI_SCANNER_SQLITE_FALLBACK: "false"', source)
        self.assertIn("Upload scheduled scan summary", source)
        self.assertIn("scheduled_scan_summary.json", source)
        self.assertNotIn("ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}", source)
        self.assertNotIn("ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}", source)

    def test_streamlit_smoke_script_starts_live_server(self):
        source = (ROOT / "scripts" / "streamlit_smoke.py").read_text()

        self.assertIn("streamlit", source)
        self.assertIn("_stcore/health", source)
        self.assertIn("def _validate_browser_shell", source)
        self.assertIn("static/js", source)
        self.assertIn("subprocess.Popen", source)
        self.assertIn("STREAMLIT_SERVER_HEADLESS", source)

    def test_streamlit_ui_uses_width_instead_of_use_container_width(self):
        allowed = {
            ROOT / "ui" / "app_boot.py",
        }
        offenders = []
        for path in (ROOT / "ui").rglob("*.py"):
            if path in allowed:
                continue
            if "use_container_width" in path.read_text():
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])

    def test_streamlit_browser_flow_script_exercises_login_shell(self):
        source = (ROOT / "scripts" / "streamlit_browser_flow.py").read_text()
        browser_requirements = (ROOT / "requirements-browser.txt").read_text()
        readme_source = (ROOT / "README.md").read_text()

        self.assertIn("playwright", browser_requirements)
        self.assertIn("import contextlib", source)
        self.assertIn("sync_playwright", source)
        self.assertIn("_stcore/health", source)
        self.assertIn("LOGIN_MARKERS", source)
        self.assertIn("AI_SCANNER_SMOKE_TEST", source)
        self.assertIn("ERROR_MARKERS", source)
        self.assertIn("Browser flow found app error marker", source)
        self.assertIn("StreamlitSecretNotFoundError", source)
        self.assertIn("page.goto", source)
        self.assertIn("def _wait_for_app_markers", source)
        self.assertIn("page.wait_for_function", source)
        self.assertIn("def _stop_streamlit", source)
        self.assertIn("proc.communicate(timeout=10)", source)
        self.assertIn("python scripts/streamlit_browser_flow.py --timeout 60", readme_source)

    def test_deployment_doctor_exists(self):
        source = (ROOT / "scripts" / "deployment_doctor.py").read_text()

        self.assertIn("def run_checks", source)
        self.assertIn("NEON_DATABASE_URL", source)
        self.assertIn("ALPACA_API_KEY_ID", source)
        self.assertIn("scheduler.cron_runner", source)

    def test_single_ticker_tools_are_extracted_from_scan_ui(self):
        scans_source = (ROOT / "ui" / "scans.py").read_text()
        single_source = (ROOT / "ui" / "single_ticker.py").read_text()

        self.assertIn("render_single_ticker_panel", scans_source)
        self.assertIn("handle_single_ticker_actions", scans_source)
        self.assertNotIn("def _get_live_quote", scans_source)
        self.assertNotIn("def _render_single_symbol_chart", scans_source)
        self.assertIn("def get_live_quote", single_source)
        self.assertIn("def render_single_symbol_chart", single_source)

    def test_watchlist_scan_tools_are_extracted_from_scan_ui(self):
        scans_source = (ROOT / "ui" / "scans.py").read_text()
        watchlist_source = (ROOT / "ui" / "watchlists.py").read_text()

        self.assertIn("render_active_watchlist_tools", scans_source)
        self.assertIn("handle_active_watchlist_actions", scans_source)
        self.assertNotIn("def build_watchlist_df", scans_source)
        self.assertIn("def build_watchlist_df", watchlist_source)
        self.assertIn("def render_active_watchlist_tools", watchlist_source)
        self.assertIn("def handle_active_watchlist_actions", watchlist_source)

    def test_scan_provider_helpers_are_extracted_from_scan_ui(self):
        scans_source = (ROOT / "ui" / "scans.py").read_text()
        provider_source = (ROOT / "ui" / "scan_providers.py").read_text()
        diagnostics_source = (ROOT / "ui" / "scan_diagnostics.py").read_text()
        three_step_source = (ROOT / "ui" / "three_step_scanner.py").read_text()

        self.assertLess(len(scans_source.splitlines()), 550)
        self.assertIn("from ui.scan_providers import", scans_source)
        self.assertIn("from ui.scan_diagnostics import render_data_provider_diagnostics", scans_source)
        self.assertIn("from ui.three_step_scanner import render_three_step_scanner", scans_source)
        self.assertIn("apply_alpaca_extended_prices", scans_source)
        self.assertNotIn("import requests", scans_source)
        self.assertNotIn("ALPACA_MAX_SNAPSHOT_BATCH", scans_source)
        self.assertNotIn("def render_three_step_scanner", scans_source)
        self.assertNotIn("def run_scan_engine", scans_source)
        self.assertNotIn("def _get_alpaca_headers", scans_source)
        self.assertNotIn("def _get_alpaca_extended_last_prices", scans_source)
        self.assertNotIn("def _apply_alpaca_extended_prices", scans_source)
        self.assertNotIn("def render_data_provider_diagnostics", scans_source)
        self.assertIn("def render_three_step_scanner", three_step_source)
        self.assertIn("def run_scan_engine", three_step_source)
        self.assertNotIn(":blue_circle:", three_step_source)
        self.assertNotIn(":green_circle:", three_step_source)
        self.assertIn('"active": "🔵"', three_step_source)
        self.assertIn('"done": "🟢"', three_step_source)
        self.assertIn("time.perf_counter()", three_step_source)
        self.assertIn("duration_sec=round(float(duration_sec), 2)", three_step_source)
        self.assertNotIn("duration_sec=0.0", three_step_source)
        self.assertIn("except (RuntimeError, TypeError, ValueError, OSError)", three_step_source)
        self.assertIn("def sanitize_universe_symbols", provider_source)
        self.assertIn("def get_alpaca_extended_last_prices", provider_source)
        self.assertIn("except (ValueError, requests_exc.RequestException)", provider_source)
        self.assertIn("def render_data_provider_diagnostics", diagnostics_source)
        self.assertIn("except (RuntimeError, OSError, ValueError)", diagnostics_source)

    def test_app_boot_helpers_are_extracted_from_app(self):
        app_source = (ROOT / "app.py").read_text()
        boot_source = (ROOT / "ui" / "app_boot.py").read_text()

        self.assertLess(app_source.index("sys.path.insert"), app_source.index("from ui.app_boot import"))
        self.assertIn("install_streamlit_compat", app_source)
        self.assertIn("configure_page()", app_source)
        self.assertIn("quiet_external_calls as _quiet_external_calls", app_source)
        self.assertNotIn("class _FilteredStderr", app_source)
        self.assertNotIn("def _patch_use_container_width", app_source)
        self.assertIn("class FilteredStderr", boot_source)
        self.assertIn("def patch_use_container_width", boot_source)
        self.assertIn("def install_warning_filters", boot_source)
        self.assertIn("warnings.filterwarnings", boot_source)
        self.assertIn("yfinance\\.utils", boot_source)
        self.assertIn("generic' unit for NumPy timedelta", boot_source)
        self.assertIn("def configure_page", boot_source)
        self.assertNotIn("st.cache =", boot_source)
        self.assertIn("`st.cache` is deprecated", boot_source)

    def test_app_session_helpers_are_extracted_from_app(self):
        app_source = (ROOT / "app.py").read_text()
        session_source = (ROOT / "ui" / "app_session.py").read_text()

        self.assertIn("from ui.app_session import", app_source)
        self.assertIn("normalize_admin_users", app_source)
        self.assertNotIn("def _norm_str", app_source)
        self.assertNotIn("FEATURE_MIN_TIER: dict", app_source)
        self.assertIn("def normalize_admin_users", session_source)
        self.assertIn("def is_admin_user", session_source)
        self.assertIn("def compute_entitlements", session_source)

    def test_app_runtime_helpers_are_extracted_from_app(self):
        app_source = (ROOT / "app.py").read_text()
        runtime_source = (ROOT / "ui" / "app_runtime.py").read_text()

        self.assertLess(len(app_source.splitlines()), 680)
        self.assertIn("from ui.app_runtime import", app_source)
        self.assertNotIn("def get_market_session", app_source)
        self.assertNotIn("def _normalize_results_to_df", app_source)
        self.assertNotIn("def render_active_filters_summary", app_source)
        self.assertNotIn("def render_onboarding_hint", app_source)
        self.assertIn("def get_market_session", runtime_source)
        self.assertIn("def normalize_results_to_df", runtime_source)
        self.assertIn("except (JSONDecodeError, TypeError)", runtime_source)
        self.assertIn("def render_sidebar_upgrade_card", runtime_source)

    def test_app_profile_helpers_are_extracted_from_app(self):
        app_source = (ROOT / "app.py").read_text()
        profile_source = (ROOT / "ui" / "app_user_profile.py").read_text()

        self.assertIn("from ui.app_user_profile import", app_source)
        self.assertIn("render_account_sidebar(", app_source)
        self.assertIn("load_saved_user_settings(", app_source)
        self.assertIn("apply_admin_scan_caps(", app_source)
        self.assertIn("load_latest_results_snapshot(", app_source)
        self.assertNotIn("def _account_label", app_source)
        self.assertNotIn("ADMIN_SCAN_CAP = 100_000", app_source)
        self.assertIn("def render_account_sidebar", profile_source)
        self.assertIn("def load_saved_user_settings", profile_source)
        self.assertIn("def apply_admin_scan_caps", profile_source)
        self.assertIn("APP_PROFILE_ERRORS = (", profile_source)

    def test_universe_db_helpers_are_extracted_from_universe_ui(self):
        universe_source = (ROOT / "ui" / "universe.py").read_text()
        universe_db_source = (ROOT / "ui" / "universe_db.py").read_text()

        self.assertLess(len(universe_source.splitlines()), 500)
        self.assertIn("from ui.universe_db import", universe_source)
        self.assertIn("db_get_universe(", universe_source)
        self.assertIn("db_upsert_universe(", universe_source)
        self.assertNotIn("def _get_db_conn", universe_source)
        self.assertNotIn("def _db_get_universe", universe_source)
        self.assertNotIn("def _db_upsert_universe", universe_source)
        self.assertNotIn("except Exception", universe_source)
        self.assertIn("def db_get_universe", universe_db_source)
        self.assertIn("def db_upsert_universe", universe_db_source)
        self.assertIn("def try_import", universe_db_source)
        self.assertIn("UNIVERSE_DB_ERRORS = (", universe_db_source)

    def test_results_tabs_are_extracted_from_app(self):
        app_source = (ROOT / "app.py").read_text()
        tabs_source = (ROOT / "ui" / "results_tabs.py").read_text()
        admin_tabs_source = (ROOT / "ui" / "admin_results_tab.py").read_text()

        self.assertIn("from ui.results_tabs import render_results_tabs", app_source)
        self.assertIn("render_results_tabs(", app_source)
        self.assertNotIn("tab_names = [f\"📊 Latest scan results", app_source)
        self.assertNotIn("def _render_admin_tab", app_source)
        self.assertLess(len(tabs_source.splitlines()), 350)
        self.assertIn("def render_results_tabs", tabs_source)
        self.assertIn("def _render_scan_history_tab", tabs_source)
        self.assertIn("key_prefix=f\"history_results_{_safe_widget_key(picked)}\"", tabs_source)
        self.assertIn("from ui.admin_results_tab import render_admin_tab", tabs_source)
        self.assertNotIn("def _render_admin_tab", tabs_source)
        self.assertNotIn("except Exception", tabs_source)
        self.assertIn("def render_admin_tab", admin_tabs_source)
        self.assertIn("ADMIN_TAB_ERRORS = (", admin_tabs_source)
        self.assertIn("populate_earnings_calendar", admin_tabs_source)
        self.assertIn("def _resolve_earnings_refresh_symbols", admin_tabs_source)
        self.assertNotIn("_fetch_earnings_list", admin_tabs_source)
        self.assertNotIn("except Exception", admin_tabs_source)
        self.assertIn("def _render_scan_errors_panel", admin_tabs_source)
        self.assertIn("def _render_login_attempts_panel", admin_tabs_source)
        self.assertIn("def _render_billing_health_badge", admin_tabs_source)
        self.assertIn("ensure_neon_scan_errors_schema", admin_tabs_source)
        self.assertIn("ensure_neon_login_attempts_schema", admin_tabs_source)

    def test_db_and_user_settings_helpers_are_extracted_from_app(self):
        app_source = (ROOT / "app.py").read_text()
        db_source = (ROOT / "db" / "core.py").read_text()
        settings_source = (ROOT / "ui" / "user_settings.py").read_text()

        self.assertIn("from db.core import get_conn as _get_db_conn_for_app", app_source)
        self.assertIn("from ui.user_settings import render_user_settings_footer", app_source)
        self.assertNotIn("def _get_database_url", app_source)
        self.assertNotIn("def _get_db_conn_for_app", app_source)
        self.assertNotIn("def render_user_settings_footer", app_source)
        self.assertIn("def _get_database_url", db_source)
        self.assertIn("NEON_DATABASE_URL", db_source)
        self.assertIn("def render_user_settings_footer", settings_source)

    def test_earnings_results_helpers_are_extracted_from_app(self):
        app_source = (ROOT / "app.py").read_text()
        earnings_source = (ROOT / "ui" / "earnings_results.py").read_text()
        earnings_panel_source = (ROOT / "ui" / "earnings.py").read_text()

        self.assertIn("from ui.earnings_results import", app_source)
        self.assertIn("render_earnings_controls(", app_source)
        self.assertIn("prepare_results_with_earnings(", app_source)
        self.assertNotIn("def _apply_earnings_enrichment", app_source)
        self.assertNotIn("def _canonical_symbol_series", app_source)
        self.assertNotIn("earnings_enriched_df\"] = enriched_df", app_source)
        self.assertIn("def render_earnings_controls", earnings_source)
        self.assertIn("def prepare_results_with_earnings", earnings_source)
        self.assertIn("def _apply_earnings_enrichment", earnings_source)
        self.assertIn('["ticker", "earnings_date", EARN_COL_DAYS]', earnings_panel_source)
        self.assertNotIn('"earnings_time", EARN_COL_DAYS', earnings_panel_source)

    def test_market_header_helpers_are_not_duplicated_in_app(self):
        app_source = (ROOT / "app.py").read_text()
        header_source = (ROOT / "ui" / "header.py").read_text()

        self.assertIn("from ui.header import render_header, render_market_snapshot, render_price_ticker", app_source)
        self.assertIn("render_price_ticker()", app_source)
        self.assertIn("render_market_snapshot(results_df=_snapshot_df)", app_source)
        self.assertNotIn("def _fetch_index_snapshot", app_source)
        self.assertNotIn("def _render_market_snapshot_legacy", app_source)
        self.assertNotIn("def _fetch_ticker_quotes", app_source)
        self.assertNotIn("def _render_price_ticker_legacy", app_source)
        self.assertIn("def _fetch_index_snapshot", header_source)
        self.assertIn("def render_market_snapshot", header_source)
        self.assertIn("def render_price_ticker", header_source)
        self.assertIn("from market_data import get_latest_quotes", header_source)
        self.assertIn("Header rendering must never call yfinance directly", header_source)
        self.assertNotIn("import yfinance", header_source)
        self.assertNotIn("yf.download", header_source)
        self.assertNotIn("yf.Ticker", header_source)

    def test_result_helpers_are_extracted_from_results_ui(self):
        results_path = ROOT / "ui" / "results.py"
        results_source = results_path.read_text()
        helper_source = (ROOT / "ui" / "result_helpers.py").read_text()
        table_source = (ROOT / "ui" / "result_tables.py").read_text()
        watchlist_source = (ROOT / "ui" / "result_watchlist.py").read_text()
        charts_source = (ROOT / "ui" / "charts.py").read_text()
        single_ticker_source = (ROOT / "ui" / "single_ticker.py").read_text()

        self.assertLess(len(results_source.splitlines()), 720)
        self.assertIn("from ui.result_helpers import", results_source)
        self.assertIn("from ui.result_tables import render_static_results_table", results_source)
        self.assertIn("from ui.result_watchlist import render_watchlist_action", results_source)
        self.assertNotIn("def _sync_selected_ticker_from_table", results_source)
        self.assertNotIn("def _find_row_for_ticker", results_source)
        self.assertNotIn("def _render_watchlist_action", results_source)
        self.assertNotIn(".basic-results-wrap {", results_source)
        self.assertNotIn("except Exception:\n                styles.append", results_source)
        self.assertIn("def sync_selected_ticker_from_table", helper_source)
        self.assertIn("def find_row_for_ticker", helper_source)
        self.assertIn("except (TypeError, ValueError, IndexError)", helper_source)
        self.assertIn("def render_static_results_table", table_source)
        self.assertIn("BASIC_RESULTS_TABLE_CSS", table_source)
        self.assertIn("def render_watchlist_action", watchlist_source)
        self.assertIn("except ImportError", watchlist_source)
        self.assertIn("except (RuntimeError, TypeError, ValueError, OSError)", watchlist_source)
        self.assertIn("except (RuntimeError, TypeError, ValueError)", results_source)
        self.assertNotIn("except Exception:\n            st.caption(\"AI notes are unavailable", results_source)
        self.assertIn("st.plotly_chart(fig, width=\"stretch\", key=key)", charts_source)
        self.assertIn("key=f\"single_ticker_chart_{_safe_chart_key(sym)}\"", single_ticker_source)
        self.assertIn("key=f\"{key_prefix}_chart_{st.session_state[picker_key]}_fast\"", results_source)
        self.assertIn("key=f\"{key_prefix}_chart_{st.session_state[picker_key]}_styled\"", results_source)

    def test_plaintext_demo_auth_config_is_removed(self):
        self.assertFalse((ROOT / "ui" / "config.yaml").exists())
        users_source = (ROOT / "db" / "users.py").read_text()

        self.assertIn("ENABLE_DEMO_USERS", users_source)
        self.assertIn("DEMO_BASIC_PASSWORD", users_source)

    def test_scheduler_ui_placeholder_has_real_renderer(self):
        source = (ROOT / "scheduler" / "ui.py").read_text()

        self.assertIn("def render_scheduler", source)
        self.assertIn("AI_SCANNER_SQLITE_FALLBACK", source)

    def test_market_heat_helpers_are_extracted_from_pages_main(self):
        pages_source = (ROOT / "ui" / "pages_main.py").read_text()
        market_heat_source = (ROOT / "ui" / "market_heat.py").read_text()
        runner_source = (ROOT / "ui" / "page_runners.py").read_text()

        self.assertLess(len(pages_source.splitlines()), 500)
        self.assertIn("from ui.market_heat import", pages_source)
        self.assertIn("from ui.page_runners import", pages_source)
        self.assertNotIn("def _fetch_predefined_screener", pages_source)
        self.assertNotIn("def _bind_session_args", pages_source)
        self.assertNotIn("def _call_with_overrides", pages_source)
        self.assertNotIn("def _optional_attr", pages_source)
        self.assertNotIn("import requests as _requests", pages_source)
        self.assertIn("def _fetch_predefined_screener", market_heat_source)
        self.assertIn("except (_requests_exc.RequestException, ValueError)", market_heat_source)
        self.assertIn("def _bind_session_args", runner_source)
        self.assertIn("def _call_with_overrides", runner_source)
        self.assertIn("except (ImportError, AttributeError)", runner_source)

    def test_auth_lockout_helpers_are_extracted_from_auth_ui(self):
        auth_source = (ROOT / "ui" / "auth.py").read_text()
        lockout_source = (ROOT / "ui" / "auth_lockout.py").read_text()
        sessions_source = (ROOT / "ui" / "auth_sessions.py").read_text()

        self.assertLess(len(auth_source.splitlines()), 690)
        self.assertIn("from ui.auth_lockout import", auth_source)
        self.assertIn("from ui.auth_sessions import", auth_source)
        self.assertNotIn("def _is_login_locked", auth_source)
        self.assertNotIn("def _register_failed_login_attempt", auth_source)
        self.assertNotIn("def _cookies_ready_or_stop", auth_source)
        self.assertNotIn("def _create_session", auth_source)
        self.assertNotIn("def _get_username_for_session", auth_source)
        self.assertNotIn("def _delete_session", auth_source)
        self.assertNotIn("COOKIE_PASSWORD", auth_source)
        self.assertNotIn("except Exception", auth_source)
        self.assertIn("COOKIE_MANAGER_STATE_KEY", auth_source)
        self.assertIn("def is_login_locked", lockout_source)
        self.assertIn("def register_failed_login_attempt", lockout_source)
        self.assertIn("except (TypeError, ValueError)", lockout_source)
        self.assertIn("def cookies_ready_or_stop", sessions_source)
        self.assertIn("COOKIE_MANAGER_STATE_KEY", sessions_source)
        self.assertIn("st.session_state.get(COOKIE_MANAGER_STATE_KEY)", sessions_source)
        self.assertIn("st.session_state[COOKIE_MANAGER_STATE_KEY] = cookies", sessions_source)
        self.assertIn("def create_session", sessions_source)
        self.assertIn("def get_username_for_session", sessions_source)
        self.assertIn("def delete_session", sessions_source)
        self.assertIn("COOKIE_PASSWORD", sessions_source)
        self.assertNotIn("except Exception", sessions_source)

    def test_headless_scan_helpers_are_shared(self):
        pre_post_source = (ROOT / "scan" / "pre_post.py").read_text()
        session_source = (ROOT / "scan" / "session.py").read_text()
        common_source = (ROOT / "scan" / "headless_common.py").read_text()

        self.assertIn("def run_headless_pipeline", common_source)
        self.assertIn("def fetch_headless_prices", common_source)
        self.assertIn("HEADLESS_BOUNDARY_ERRORS = (", common_source)
        self.assertIn("run_headless_pipeline", pre_post_source)
        self.assertIn("run_headless_pipeline", session_source)
        self.assertNotIn("fetch_price_data_parallel", pre_post_source)
        self.assertNotIn("fetch_price_data_batch", session_source)
        self.assertNotIn("except Exception", pre_post_source)
        self.assertNotIn("except Exception", session_source)

    def test_ui_package_uses_lazy_submodule_imports(self):
        source = (ROOT / "ui" / "__init__.py").read_text()

        self.assertIn("def __getattr__", source)
        self.assertIn("importlib.import_module", source)
        self.assertNotIn("pages = _try_import", source)
        self.assertNotIn("universe = _try_import", source)


if __name__ == "__main__":
    unittest.main()
