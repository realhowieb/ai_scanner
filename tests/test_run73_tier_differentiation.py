"""Run 73: customer plan promises must match enforced entitlements."""
from pathlib import Path

from ui.app_session import ALERT_LIMIT_BY_TIER, FEATURE_MIN_TIER, compute_entitlements
from ui.pricing import (
    ALERTS,
    ROWS,
    TAGLINES,
    alert_upgrade_message,
    included,
    plans_html,
    pricing_markdown,
    required_tier,
    upgrade_message,
)

ROOT = Path(__file__).resolve().parents[1]


def test_free_retains_core_discovery_and_basic_intelligence():
    free_rows = {label for label, flag in ROWS if flag is None}
    assert "Latest full-market ranking (HSF Score), Today and scheduled market opportunities" in free_rows
    assert "HSF Score and basic Stock Intelligence" in free_rows
    assert all(included(flag, "basic") for _label, flag in ROWS if flag is None)
    flags = compute_entitlements(tier_obj="basic", is_admin=False)
    assert flags["can_scan_sp500"] is True


def test_pro_is_monitor_and_investigate_plan():
    expected = {
        "can_email_alerts", "can_export_csv", "can_scan_nasdaq", "can_premarket",
        "can_afterhours", "can_unusual_volume", "can_earnings", "can_scan_history",
        "can_track_record",
    }
    assert all(FEATURE_MIN_TIER[feature] == "pro" for feature in expected)
    flags = compute_entitlements(tier_obj="pro", is_admin=False)
    assert all(flags[feature] for feature in expected)
    assert "Monitor and investigate" in TAGLINES["pro"]


def test_premium_promises_are_real_and_exclusive_to_premium():
    premium = {"can_ai_notes", "can_early_breakout", "can_full_universe", "can_paper_trade"}
    assert all(FEATURE_MIN_TIER[feature] == "premium" for feature in premium)
    pro = compute_entitlements(tier_obj="pro", is_admin=False)
    paid = compute_entitlements(tier_obj="premium", is_admin=False)
    assert all(not pro[feature] and paid[feature] for feature in premium)


def test_alert_limits_and_upgrade_copy_match_plans():
    assert ALERT_LIMIT_BY_TIER == {"basic": 1, "pro": 5, "premium": 25, "admin": 25}
    alert_row = next(line for line in pricing_markdown().splitlines() if line.startswith("| Alerts"))
    assert "| 1 | 5 | 25 |" in alert_row
    assert "Pro" in alert_upgrade_message("basic") and "5" in alert_upgrade_message("basic")
    assert "Premium" in alert_upgrade_message("pro") and "25" in alert_upgrade_message("pro")


def test_upgrade_prompts_name_correct_plan_and_user_value():
    assert required_tier("can_export_csv") == "pro"
    assert "Pro" in upgrade_message("can_export_csv")
    assert "interactive results" in upgrade_message("can_export_csv")
    assert required_tier("can_ai_notes") == "premium"
    assert "Premium" in upgrade_message("can_ai_notes")
    assert "research workflow" in upgrade_message("can_ai_notes")


def test_admin_only_capabilities_are_not_customer_benefits():
    customer_flags = {flag for _label, flag in ROWS if flag not in (None, ALERTS)}
    assert "can_diagnostics" not in customer_flags
    assert "can_admin_panel" not in customer_flags
    text = pricing_markdown().lower()
    assert "diagnostics" not in text
    assert "retrain" not in text


def test_signed_out_and_billing_use_same_shared_plan_renderer():
    landing = (ROOT / "ui" / "landing.py").read_text()
    billing = (ROOT / "pages" / "billing.py").read_text()
    assert "plans_html()" in landing
    assert "from ui.pricing import pricing_markdown" in billing
    assert "from ui.pricing import benefits_markdown" in billing


def test_pro_is_the_only_featured_plan_and_free_copy_is_truthful():
    html = plans_html()
    assert html.count('class="hsf-plan featured"') == 1
    assert "Discover what matters in the market" in html
    auth = (ROOT / "ui" / "auth.py").read_text()
    assert "basic Stock Intelligence" in auth          # Run 85B: lowercase so it never reads as a plan name
    assert "Interactive charts" not in auth
    sidebar = (ROOT / "ui" / "app_runtime.py").read_text()
    assert "You're seeing a limited scan" not in sidebar


def test_no_prohibited_or_misleading_plan_claims():
    customer_copy = (pricing_markdown() + plans_html() + " ".join(upgrade_message(k) for k in FEATURE_MIN_TIER)).lower()
    prohibited = (
        "ai-powered rankings", "full-universe mode", "guaranteed", "guarantee",
        "probability of profit", "win rate", "risk-free", "can't lose",
    )
    assert all(claim not in customer_copy for claim in prohibited)
