"""User-facing error contracts and owned paper-order status."""
from unittest import mock

from tests.test_api_paid_features import PaidApiTestCase


class ExperienceTests(PaidApiTestCase):
    def test_validation_is_actionable_without_echoing_paper_secrets(self):
        secret = "sensitive-secret-that-is-too-long-" * 20
        response = self.client.post("/v1/paper/account", headers={**self.h("prem@example.com"),
                                    "X-Request-ID": "experience-test-id"},
                                    json={"api_key": "valid-key", "api_secret": secret})
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["code"], "validation_error")
        self.assertIn("api_secret", body["fields"])
        self.assertEqual(body["request_id"], response.headers["X-Request-ID"])
        self.assertNotIn(secret, response.text)
        self.assertNotIn("input", body["detail"][0])

    def test_zero_share_trade_is_rejected_before_writing(self):
        with mock.patch("api.trading.log") as write:
            response = self.client.post("/v1/journal", headers=self.h("pro@example.com"),
                                        json={"ticker": "AAPL", "entry_price": 100, "shares": 0})
        self.assertEqual(response.status_code, 422)
        self.assertIn("shares", response.json()["fields"])
        write.assert_not_called()

    def test_status_requires_premium_and_only_passes_the_current_owner(self):
        receipt = {"state": "submitted", "status": "accepted", "ticker": "AAPL", "qty": 2,
                   "order_id": "o1", "filled_qty": 0, "journal_synced": False}
        with mock.patch("services.paper_orders.status", return_value=receipt) as lookup:
            denied = self.get("pro@example.com", "/v1/paper/orders/confirmation-123")
            self.assertEqual(denied.status_code, 403)
            lookup.assert_not_called()
            response = self.get("prem@example.com", "/v1/paper/orders/confirmation-123")
        self.assertEqual(response.status_code, 200)
        lookup.assert_called_once_with("prem@example.com", "confirmation-123")
        self.assertEqual(response.json()["state"], "submitted")

    def test_unknown_confirmation_does_not_disclose_another_users_order(self):
        with mock.patch("services.paper_orders.status", return_value=None):
            self.assertEqual(self.get("prem@example.com", "/v1/paper/orders/confirmation-123").status_code, 404)
