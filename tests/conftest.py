"""Keep the test run away from a developer's real secrets.

A local .streamlit/secrets.toml (P2-37 points it at the Neon branch dev-local)
would otherwise be read by config and st.secrets during tests. CI has no such
file; this makes local runs behave the same way.
"""
try:
    from streamlit import config as _st_config

    _st_config.set_option("secrets.files", ["/nonexistent/hsf-tests-have-no-secrets.toml"])
except Exception:  # streamlit not installed (lightweight suites)
    pass


import sys as _sys

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _fresh_api_account_cache():
    """The API reuses account rows for a few seconds per process; tests swap fake
    accounts under the same email, so each test starts with nothing cached."""
    main = _sys.modules.get("api.main")
    if main is not None:
        main._account_cache.clear()
    yield
    main = _sys.modules.get("api.main")
    if main is not None:
        main._account_cache.clear()
