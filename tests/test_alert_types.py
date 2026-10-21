import unittest


class AlertTypesTests(unittest.TestCase):
    def test_ema_cross_is_allowed_alert_type(self):
        from db.alerts import ALERT_TYPES

        self.assertIn("ema_cross", ALERT_TYPES)


    def test_ewo_cross_is_allowed_alert_type(self):
        from db.alerts import ALERT_TYPES

        self.assertIn("ewo_cross", ALERT_TYPES)


if __name__ == "__main__":
    unittest.main()
