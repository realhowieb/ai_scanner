"""Run 75: one onboarding path, one scan entry and Today-first navigation."""
from pathlib import Path

from ui.app_session import TODAY_LANDING_KEY, should_land_on_today

ROOT = Path(__file__).resolve().parents[1]


def test_today_landing_happens_once_per_authenticated_user():
    state = {}
    assert should_land_on_today(state, " User@Example.com ") is True
    assert state[TODAY_LANDING_KEY] == "user@example.com"
    assert should_land_on_today(state, "user@example.com") is False
    assert should_land_on_today(state, "other@example.com") is True


def test_scanner_has_one_custom_scan_entry():
    app = (ROOT / "app.py").read_text()
    assert app.count('st.expander("Custom scan"') == 1
    assert 'st.markdown("## Run your own scan")' not in app
    assert "container=custom_scan_box" in app


def test_tour_is_only_onboarding_renderer():
    app = (ROOT / "app.py").read_text()
    onboarding = (ROOT / "ui" / "onboarding.py").read_text()
    assert 'render_tour("scanner")' in app
    assert "render_hsf_onboarding_entry" not in app
    assert "render_scanner_orientation" not in app
    assert "_render_dismissible_orientation" not in onboarding


def test_shared_link_redirect_precedes_today_landing():
    app = (ROOT / "app.py").read_text()
    shared = app.index('st.session_state.get("hsf_after_login_page")')
    today = app.index("should_land_on_today(st.session_state, username)")
    assert shared < today


def test_dead_ui_modules_are_removed():
    assert not (ROOT / "ui" / "components.py").exists()
    assert not (ROOT / "ui" / "watchlist_alerts.py").exists()
    assert not (ROOT / "ui" / "history.py").exists()
