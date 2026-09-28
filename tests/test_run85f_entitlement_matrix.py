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
from ui.discover import entitled_lenses
from ui.entitlement_view import (
    redact_prebreakout_frame,
    redact_prebreakout_opportunity,
)
from ui.smart_alerts import build_smart_alert_suggestions

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
        "primary_setup": "PreBreakout", "prob": 72.0,
        "reasons": ["PreBreakout setup probability 72%", "Gap +3%"],
        "model": {"breakout_score": 12.0, "prebreakout_prob": 72.0},
    }
    hidden = redact_prebreakout_opportunity(opportunity, allowed=False)
    shown = redact_prebreakout_opportunity(opportunity, allowed=True)
    assert hidden["score"] == shown["score"] == 77
    assert hidden["signals"] == ["gapper"]
    assert hidden["primary_setup"] == "Gapper"
    assert hidden["prob"] is None
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


def test_smart_alert_prebreakout_evidence_requires_premium_entitlement():
    df = pd.DataFrame([{"Ticker": "XYZ", "PreBreakoutProb": 0.72}])
    assert build_smart_alert_suggestions(df, can_early_breakout=False) == []
    assert len(build_smart_alert_suggestions(df, can_early_breakout=True)) == 1


def test_early_breakout_lens_is_premium_only():
    counts = {"early": 4, "gaps": 2}
    assert "early" not in entitled_lenses(counts, can_early_breakout=False)
    assert "early" in entitled_lenses(counts, can_early_breakout=True)
    assert "gaps" in entitled_lenses(counts, can_early_breakout=False)


def test_verified_alternate_paths_use_canonical_capability_names():
    checks = {
        "ui/watchlists.py": "can_export_csv",
        "ui/watchlist_intelligence_feed.py": "can_track_record",
        "ui/alerts.py": "can_track_record",
        "ui/results.py": "can_export_csv",
        "ui/results_tabs.py": "can_early_breakout",
        "ui/stock_intelligence.py": "can_early_breakout",
        "ui/market_brief.py": "can_early_breakout",
        "pages/journal.py": "can_paper_trade",
    }
    for rel, capability in checks.items():
        assert capability in (ROOT / rel).read_text(), rel


def test_pro_csv_export_redacts_premium_model_columns():
    source = (ROOT / "ui" / "results.py").read_text()
    assert source.count("redact_prebreakout_frame(") >= 2
    assert source.count('ent.get("can_early_breakout")') >= 2


def test_background_email_and_alert_workers_enforce_tiers():
    digest = (ROOT / "scheduler" / "morning_digest.py").read_text()
    realtime = (ROOT / "billing_service" / "realtime_alerts.py").read_text()
    scheduled = (ROOT / "scheduler" / "alert_runner.py").read_text()
    assert "tier_key = _email_tier_key(email, users.get(username), users, get_user_tier)" in digest
    assert 'has_min_tier(tier_key, "pro")' in digest
    assert 'has_min_tier(tier_key, "premium")' in digest
    assert "ALERT_LIMITS" in realtime and "_apply_plan_limits(alerts)" in realtime
    assert "_alert_limit_for_user" in scheduled and "_email_allowed_for_tier" in scheduled


def test_customer_plan_name_never_uses_basic_label():
    from ui.plan_labels import plan_label

    assert plan_label("basic") == "Free"
    for rel in ("ui/account_card.py", "ui/pricing.py", "pages/billing.py"):
        assert "You're on Basic" not in (ROOT / rel).read_text()


def test_labs_is_authenticated_but_has_no_invented_customer_entitlement():
    source = (ROOT / "pages" / "kalshi.py").read_text()
    assert 'if not _username:' in source
    assert not any("kalshi" in capability for capability in FEATURE_MIN_TIER)


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
