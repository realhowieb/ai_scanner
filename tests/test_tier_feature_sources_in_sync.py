"""The two plan-rule sources must agree.

config.TIERS_CONFIG "features" drive the Tier flags (auth.tiering.get_user_tier,
used by ui/filters.py); ui.app_session.FEATURE_MIN_TIER drives entitlements,
the pricing page and gating. A plan change made in one place only would show a
feature on one screen and lock it on another.
"""
import unittest

from auth.tiering import get_user_tier
from ui.app_session import FEATURE_MIN_TIER, TIER_ORDER

# Tier attribute -> FEATURE_MIN_TIER flag
TIER_FLAG = {
    "can_premarket": "can_premarket",
    "can_afterhours": "can_afterhours",
    "can_unusual_volume": "can_unusual_volume",
    "can_export_csv": "can_export_csv",
    "can_ai_notes": "can_ai_notes",
}


def _entitled(tier: str, flag: str) -> bool:
    return TIER_ORDER[tier] >= TIER_ORDER[FEATURE_MIN_TIER[flag]]


class TierFeatureSourcesInSync(unittest.TestCase):
    def test_tier_flags_match_feature_min_tier(self):
        for tier in ("basic", "pro", "premium", "admin"):
            t = get_user_tier("x", users={"x": {"tier": tier}})
            self.assertEqual(t.key, tier)
            for attr, flag in TIER_FLAG.items():
                with self.subTest(tier=tier, feature=attr):
                    self.assertEqual(getattr(t, attr), _entitled(tier, flag))

    def test_nasdaq_feature_matches(self):
        for tier in ("basic", "pro", "premium", "admin"):
            t = get_user_tier("x", users={"x": {"tier": tier}})
            with self.subTest(tier=tier):
                self.assertEqual("NASDAQ" in t.features, _entitled(tier, "can_scan_nasdaq"))


if __name__ == "__main__":
    unittest.main()
