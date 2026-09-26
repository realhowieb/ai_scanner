"""P1-12: Breakout alert threshold is visible and unambiguous."""
from pathlib import Path

from ui.alerts import (
    BREAKOUT_ALERT_DEFAULT,
    BREAKOUT_ALERT_SCALE_COPY,
    BREAKOUT_ALERT_SCALE_LEGEND,
    _fmt_alert,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scale_distinguishes_breakout_score_from_hsf_score():
    assert "Breakout Score threshold" in BREAKOUT_ALERT_SCALE_COPY
    assert "not the 0-100 HSF Score" in BREAKOUT_ALERT_SCALE_COPY
    assert "Lower thresholds fire more often" in BREAKOUT_ALERT_SCALE_COPY
    assert "higher thresholds" in BREAKOUT_ALERT_SCALE_COPY


def test_visible_legend_names_default_and_direction():
    assert BREAKOUT_ALERT_DEFAULT == 8.0
    assert "8.0 default" in BREAKOUT_ALERT_SCALE_LEGEND
    assert "more frequent" in BREAKOUT_ALERT_SCALE_LEGEND
    assert "more selective" in BREAKOUT_ALERT_SCALE_LEGEND


def test_saved_alert_label_names_the_actual_score():
    text = _fmt_alert({"alert_type": "breakout", "threshold": 8, "watchlist_only": False})
    assert text == "🚀 Breakout Score ≥ 8 (all tickers)"


def test_ui_keeps_existing_threshold_semantics_and_empirical_preview():
    source = (ROOT / "ui" / "alerts.py").read_text()
    runner = (ROOT / "scheduler" / "alert_runner.py").read_text()
    assert 'min_value=0.0' in source
    assert 'step=0.5' in source
    assert "Show threshold history and observed scale" in source
    assert "render_breakout_threshold_insight(float(thr))" in source
    assert "if score >= threshold:" in runner


def test_alert_runner_and_storage_are_not_redefined_in_ui():
    source = (ROOT / "ui" / "alerts.py").read_text()
    assert "def _evaluate(" not in source
    assert "CREATE TABLE" not in source
