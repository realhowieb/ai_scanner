import unittest


class AlertTierLimitTest(unittest.TestCase):
    def test_limits_by_tier(self):
        from ui.app_session import alert_limit_for_tier

        self.assertEqual(alert_limit_for_tier("basic"), 1)
        self.assertEqual(alert_limit_for_tier("pro"), 5)
        self.assertEqual(alert_limit_for_tier("premium"), 25)
        self.assertEqual(alert_limit_for_tier("admin"), 25)

    def test_case_insensitive_and_default(self):
        from ui.app_session import alert_limit_for_tier

        self.assertEqual(alert_limit_for_tier("PRO"), 5)
        self.assertEqual(alert_limit_for_tier(None), 1)
        self.assertEqual(alert_limit_for_tier("nonsense"), 1)

    def test_email_alerts_entitlement_is_pro(self):
        from ui.app_session import FEATURE_MIN_TIER

        self.assertEqual(FEATURE_MIN_TIER.get("can_email_alerts"), "pro")


if __name__ == "__main__":
    unittest.main()
