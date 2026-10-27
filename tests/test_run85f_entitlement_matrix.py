"""Run 85F: cross-product entitlement and alternate-render-path regression tests."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ui.app_session import (
    ALERT_LIMIT_BY_TIER,
    FEATURE_MIN_TIER,
    TIER_ORDER,
    clear_account_session_state,
    clear_entitlement_sensitive_state,
    compute_entitlements,
    enforce_account_boundary,
)

try:  # the Streamlit app was retired; its modules are gone
    from ui.discover import entitled_lenses
except ImportError:
    entitled_lenses = None
from ui.entitlement_view import (
    redact_prebreakout_frame,
    redact_prebreakout_opportunity,
)

try:  # the Streamlit app was retired; its modules are gone
    from ui.smart_alerts import build_smart_alert_suggestions
except ImportError:
    build_smart_alert_suggestions = None

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("basic", "pro", "premium", "admin")


def flags(tier: str) -> dict[str, bool]:
    return compute_entitlements(
        tier_obj=SimpleNamespace(key=tier, name=tier.upper()),
        is_admin=tier == "admin",
    )


def test_every_tier_capability_is_derived_from_the_canonical_map():
    for tier in TIERS:
        actual = flags(tier)
        assert set(actual) == set(FEATURE_MIN_TIER)
        for capability, required in FEATURE_MIN_TIER.items():
            expected = tier == "admin" or (
                required != "admin" and TIER_ORDER[tier] >= TIER_ORDER[required]
            )
            assert actual[capability] is expected, (tier, capability, required)


def test_alert_limits_are_exact_for_all_four_tiers():
    assert {tier: ALERT_LIMIT_BY_TIER[tier] for tier in TIERS} == {
        "basic": 1, "pro": 5, "premium": 25, "admin": 25,
    }


def test_dynamic_premium_state_is_cleared_on_logout_and_downgrade():
    premium_keys = {
        "_ai_summary_scan-1": "summary",
        "_ai_ticker_NVDA": "analysis",
        "_ai_chat_history": ["message"],
        "brief_narrative_2026-09-27": "brief",
        "opp_ai_NVDA": "note",
        "aic_explain_NVDA": "explanation",
        "ai_notes": "legacy",
    }
    state = {**premium_keys, "scanner_view": "cards"}
    clear_entitlement_sensitive_state(state)
    assert not any(key in state for key in premium_keys)
    assert state["scanner_view"] == "cards"

    state = {**premium_keys, "scanner_view": "table"}
    clear_account_session_state(state)
    assert not any(key in state for key in premium_keys)
    assert state["scanner_view"] == "table"


def test_dynamic_premium_state_is_cleared_across_accounts():
    state = {
        "_hsf_account_owner": "not-the-new-owner",
        "username": "new@example.com",
        "_ai_ticker_NVDA": "old account analysis",
        "brief_narrative_old": "old account brief",
        "scanner_view": "cards",
    }
    assert enforce_account_boundary(state, "new@example.com") is True
    assert "_ai_ticker_NVDA" not in state
    assert "brief_narrative_old" not in state
    assert state["username"] == "new@example.com"
    assert state["scanner_view"] == "cards"


def test_prebreakout_redaction_preserves_hsf_score_and_nonpremium_evidence():
    opportunity = {
        "ticker": "XYZ", "score": 77, "status": "STRONG",
        "signals": ["prebreakout", "gapper"], "n_signals": 2,
        "primary_setup": "PreBreakout", "prob": 72.0, "prob_raw": 0.4, "prob_rank": 3,
        "reasons": ["PreBreakout setup probability 72%", "Gap +3%"],
        "model": {"breakout_score": 12.0, "prebreakout_prob": 72.0},
    }
    hidden = redact_prebreakout_opportunity(opportunity, allowed=False)
    shown = redact_prebreakout_opportunity(opportunity, allowed=True)
    assert hidden["score"] == shown["score"] == 77
    assert hidden["signals"] == ["gapper"]
    assert hidden["primary_setup"] == "Gapper"
    assert hidden["prob"] is None
    assert hidden["prob_raw"] is None and hidden["prob_rank"] is None
    assert hidden["model"] == {"breakout_score": 12.0, "prebreakout_prob": None}
    assert hidden["reasons"] == ["Gap +3%"]
    assert shown == opportunity


def test_prebreakout_columns_are_premium_only_without_mutating_source():
    source = pd.DataFrame([{
        "Ticker": "XYZ", "HSF Score": 77, "BreakoutScore": 12,
        "PreBreakoutProb%": 72, "PreBreakoutProbRaw": 0.42,
    }])
    hidden = redact_prebreakout_frame(source, allowed=False)
    shown = redact_prebreakout_frame(source, allowed=True)
    assert "PreBreakoutProb%" not in hidden.columns
    assert "PreBreakoutProbRaw" not in hidden.columns
    assert list(hidden["HSF Score"]) == [77]
    assert shown is source
    assert "PreBreakoutProb%" in source.columns


def test_background_email_and_alert_workers_enforce_tiers():
    digest = (ROOT / "scheduler" / "morning_digest.py").read_text()
    realtime = (ROOT / "billing_service" / "realtime_alerts.py").read_text()
    scheduled = (ROOT / "scheduler" / "alert_runner.py").read_text()
    assert "tier_key = _email_tier_key(email, users.get(username), users, get_user_tier)" in digest
    assert 'has_min_tier(tier_key, "pro")' in digest
    assert 'has_min_tier(tier_key, "premium")' in digest
    assert "ALERT_LIMITS" in realtime and "_apply_plan_limits(alerts)" in realtime
    assert "_alert_limit_for_user" in scheduled and "_email_allowed_for_tier" in scheduled


def test_frozen_scoring_and_research_core_not_part_of_run85f_surface_changes():
    guarded = {
        "scan/engine.py", "ui/opportunities.py", "scan/breakout.py",
        "ml_prebreakout.py", "analytics/forward_research.py",
    }
    touched = {
        "ui/entitlement_view.py", "ui/headline_score.py", "ui/result_cards.py",
        "ui/results_intelligence.py", "ui/today.py", "ui/stock_intelligence.py",
        "ui/smart_alerts.py", "ui/watchlist_intelligence_feed.py", "ui/watchlists.py",
        "ui/alerts.py", "ui/market_brief.py", "ui/methodology.py", "pages/journal.py",
        "billing_service/realtime_alerts.py",
    }
    assert guarded.isdisjoint(touched)
