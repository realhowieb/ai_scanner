"""Run 80 regression coverage for mobile navigation and user-state policy."""
from pathlib import Path
from unittest import mock

from ui.app_session import ACCOUNT_SESSION_KEYS, clear_account_session_state, should_land_on_today
from ui.discover import (
    LENS_KEY,
    LENS_PREF,
    PREFS_LOADED_KEY,
    VIEW_KEY,
    VIEW_PREF,
    load_discover_preferences,
    mobile_request,
)

ROOT = Path(__file__).resolve().parents[1]


def test_mobile_request_only_changes_unset_default():
    assert mobile_request({"User-Agent": "Mozilla/5.0 (iPhone; Mobile)"})
    assert mobile_request({"User-Agent": "Mozilla/5.0 (Linux; Android 15)"})
    assert not mobile_request({"User-Agent": "Mozilla/5.0 (Macintosh)"})


def test_discover_preferences_survive_reruns_and_explicit_view_wins():
    state = {}
    with mock.patch("ui.browser_prefs.get", return_value="Table"), mock.patch(
        "ui.browser_prefs.get_json", return_value=["gaps", "volume"]
    ):
        load_discover_preferences(state, {"User-Agent": "iPhone Mobile"})
    assert state[VIEW_KEY] == "Table"
    assert state[LENS_KEY] == ["gaps", "volume"]
    assert state[PREFS_LOADED_KEY] is True

    state[VIEW_KEY] = "Cards"
    with mock.patch("ui.browser_prefs.get", return_value="Table") as read:
        load_discover_preferences(state, {})
    assert state[VIEW_KEY] == "Cards"
    read.assert_not_called()


def test_mobile_defaults_to_cards_only_without_saved_preference():
    state = {}
    with mock.patch("ui.browser_prefs.get", return_value=None), mock.patch(
        "ui.browser_prefs.get_json", return_value=[]
    ):
        load_discover_preferences(state, {"User-Agent": "iPhone Mobile"})
    assert state[VIEW_KEY] == "Cards"


def test_logout_cleanup_isolates_accounts_but_keeps_browser_preferences():
    state = {key: "user-a" for key in ACCOUNT_SESSION_KEYS}
    state.update({VIEW_PREF: "Cards", LENS_PREF: ["gaps"], "hsf_screens": {"Gap": ["gaps"]}})
    clear_account_session_state(state)
    assert not any(key in state for key in ACCOUNT_SESSION_KEYS)
    assert state[VIEW_PREF] == "Cards"
    assert state[LENS_PREF] == ["gaps"]
    assert state["hsf_screens"] == {"Gap": ["gaps"]}


def test_today_is_default_only_without_explicit_destination():
    state = {"hsf_after_login_page": "pages/stock.py"}
    destination = state.pop("hsf_after_login_page", None)
    assert destination == "pages/stock.py"
    assert should_land_on_today(state, "user@example.com")
    assert not should_land_on_today(state, "user@example.com")


def test_mobile_has_one_primary_navigation_and_practical_touch_targets():
    css = (ROOT / "ui" / "chrome.py").read_text()
    nav = (ROOT / "ui" / "nav.py").read_text()
    assert 'data-testid="stSidebar"' in css
    assert 'data-testid="stSidebarCollapsedControl"' in css
    assert "min-height:44px" in css
    assert '_render_identity(key_suffix="mobile")' in nav[nav.index("def render_top_menu"):]
    assert 'key=f"nav_logout_{key_suffix}"' in nav


def test_custom_scan_remains_one_entry_without_nested_popover():
    app = (ROOT / "app.py").read_text()
    assert app.count('st.expander("Custom scan"') == 1
    assert 'custom_scan_box.popover("Scan filters")' not in app
    assert '_filters_box = custom_scan_box.container(border=True)' in app


def test_frozen_scanner_core_is_untouched_by_run80():
    changed_surfaces = {"ui/app_session.py", "ui/auth.py", "ui/chrome.py", "ui/discover.py",
                        "ui/nav.py", "ui/tour.py", "app.py"}
    assert "scan/engine.py" not in changed_surfaces
    assert "analytics/hsf_score.py" not in changed_surfaces
