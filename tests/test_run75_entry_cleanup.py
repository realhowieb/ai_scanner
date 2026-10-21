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


def test_dead_ui_modules_are_removed():
    assert not (ROOT / "ui" / "components.py").exists()
    assert not (ROOT / "ui" / "watchlist_alerts.py").exists()
    assert not (ROOT / "ui" / "history.py").exists()
