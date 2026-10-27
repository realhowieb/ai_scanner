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
        # requests import is optional (import-safety); the Yahoo HTTP error tuple
        # still wires RequestException when available plus the JSON parse errors.
        self.assertIn("_YAHOO_HTTP_ERRORS = (", fetch_source)
        self.assertIn("requests.RequestException", fetch_source)
        self.assertIn("(ValueError, KeyError, TypeError)", fetch_source)
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
        self.assertNotIn("streamlit_smoke", source)  # the Streamlit app was retired
        self.assertIn("Deployment readiness doctor", source)

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


    def test_streamlit_ui_uses_width_instead_of_use_container_width(self):
        allowed = {
            ROOT / "ui" / "app_boot.py",
        }
        offenders = []
        from tests.repo_files import python_files

        for path in python_files(ROOT):
            if not path.is_relative_to(ROOT / "ui"):
                continue
            if path in allowed:
                continue
            if "use_container_width" in path.read_text():
                offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])


    def test_deployment_doctor_exists(self):
        source = (ROOT / "scripts" / "deployment_doctor.py").read_text()

        self.assertIn("def run_checks", source)
        self.assertIn("NEON_DATABASE_URL", source)
        self.assertIn("ALPACA_API_KEY_ID", source)
        self.assertIn("scheduler.cron_runner", source)


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


    def test_plaintext_demo_auth_config_is_removed(self):
        self.assertFalse((ROOT / "ui" / "config.yaml").exists())
        users_source = (ROOT / "db" / "users.py").read_text()

        self.assertIn("ENABLE_DEMO_USERS", users_source)
        self.assertIn("DEMO_BASIC_PASSWORD", users_source)


    def test_auth_lockout_helpers_are_extracted_from_auth_ui(self):
        auth_source = (ROOT / "ui" / "auth.py").read_text()
        lockout_source = (ROOT / "ui" / "auth_lockout.py").read_text()
        sessions_source = (ROOT / "ui" / "auth_sessions.py").read_text()

        self.assertLess(len(auth_source.splitlines()), 740)
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
