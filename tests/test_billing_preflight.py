"""Go-live billing pre-flight (/debug/status -> billing_preflight).

Catches the launch mistakes that break a real checkout: a test price with a
live key (or the reverse), a wrong amount or interval, an archived price, only
one yearly price, and a webhook endpoint that isn't at /webhook or misses events.
"""
import unittest
from unittest.mock import MagicMock, patch

from tests.test_billing_service import _FASTAPI_AVAILABLE, _client, _load_billing_module

PRICES = {
    "price_pro": {"livemode": False, "active": True, "unit_amount": 2500, "currency": "usd", "recurring": {"interval": "month"}},
    "price_prem": {"livemode": False, "active": True, "unit_amount": 4000, "currency": "usd", "recurring": {"interval": "month"}},
    "price_pro_y": {"livemode": False, "active": True, "unit_amount": 25000, "currency": "usd", "recurring": {"interval": "year"}},
    "price_prem_y": {"livemode": False, "active": True, "unit_amount": 40000, "currency": "usd", "recurring": {"interval": "year"}},
    "price_old": {"livemode": False, "active": False, "unit_amount": 1900, "currency": "usd", "recurring": {"interval": "month"}},
}
ENV = {
    "STRIPE_PRICE_PRO": "price_pro",
    "STRIPE_PRICE_PREMIUM": "price_prem",
    "STRIPE_PRICE_PRO_YEARLY": "price_pro_y",
    "STRIPE_PRICE_PREMIUM_YEARLY": "price_prem_y",
    "STRIPE_PRICE_PRO_LEGACY": "price_old",
    "STRIPE_PRICE_PREMIUM_LEGACY": "",
}
GOOD_HOOK = {"url": "https://ai-scanner-h2c8.onrender.com/webhook", "status": "enabled",
             "enabled_events": ["checkout.session.completed", "customer.subscription.updated",
                                "customer.subscription.deleted"]}


def _module(key="sk_test_fake", prices=None, hooks=(GOOD_HOOK,)):
    bm = _load_billing_module({"STRIPE_SECRET_KEY": key})
    table = dict(PRICES if prices is None else prices)

    def retrieve(price_id):
        if price_id not in table:
            raise RuntimeError("No such price")
        return table[price_id]

    bm.stripe.Price = MagicMock()
    bm.stripe.Price.retrieve.side_effect = retrieve
    bm.stripe.WebhookEndpoint = MagicMock()
    bm.stripe.WebhookEndpoint.list.return_value.auto_paging_iter.return_value = list(hooks)
    return bm


def _preflight(bm, env=None):
    merged = dict(ENV)
    merged.update(env or {})
    with patch.dict("os.environ", merged):
        return bm._billing_preflight()


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed in this environment")
class BillingPreflightTest(unittest.TestCase):
    def test_all_good_in_test_mode(self):
        r = _preflight(_module())
        self.assertEqual(r["problems"], [])
        self.assertTrue(r["ok"])
        self.assertEqual(r["stripe_mode"], "test")

    def test_live_key_with_test_prices_is_flagged(self):
        r = _preflight(_module(key="sk_live_fake"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["stripe_mode"], "live")
        self.assertIn("STRIPE_PRICE_PRO: price is test but key is live", r["problems"])
        self.assertIn("STRIPE_PRICE_PRO_LEGACY[0]: price is test but key is live", r["problems"])

    def test_restricted_live_key_is_live(self):
        self.assertEqual(_module(key="rk_live_x")._stripe_key_mode("rk_live_x"), "live")

    def test_unknown_price_wrong_amount_and_interval(self):
        prices = dict(PRICES)
        del prices["price_prem"]
        prices["price_pro"] = dict(PRICES["price_pro"], unit_amount=1900)
        prices["price_pro_y"] = dict(PRICES["price_pro_y"], recurring={"interval": "month"})
        r = _preflight(_module(prices=prices))
        self.assertFalse(r["ok"])
        self.assertTrue(any(p.startswith("STRIPE_PRICE_PREMIUM: not found") for p in r["problems"]))
        self.assertIn("STRIPE_PRICE_PRO: amount 1900, expected 2500", r["problems"])
        self.assertIn("STRIPE_PRICE_PRO_YEARLY: interval 'month', expected 'year'", r["problems"])

    def test_archived_current_price_fails_but_archived_legacy_is_fine(self):
        prices = dict(PRICES)
        prices["price_prem"] = dict(PRICES["price_prem"], active=False)
        r = _preflight(_module(prices=prices))
        self.assertEqual(r["problems"], ["STRIPE_PRICE_PREMIUM: price is archived"])

    def test_yearly_optional_but_not_half_set(self):
        self.assertTrue(_preflight(_module(), {"STRIPE_PRICE_PRO_YEARLY": "", "STRIPE_PRICE_PREMIUM_YEARLY": ""})["ok"])
        r = _preflight(_module(), {"STRIPE_PRICE_PREMIUM_YEARLY": ""})
        self.assertIn("only one yearly price is set; set both or neither", r["problems"])

    def test_monthly_price_missing(self):
        r = _preflight(_module(), {"STRIPE_PRICE_PRO": ""})
        self.assertIn("STRIPE_PRICE_PRO not set", r["problems"])

    def test_webhook_wrong_path_and_missing_events(self):
        wrong = dict(GOOD_HOOK, url="https://ai-scanner-h2c8.onrender.com/stripe/webhook")
        r = _preflight(_module(hooks=[wrong]))
        self.assertIn("webhook: no enabled endpoint at path /webhook", r["problems"])
        partial = dict(GOOD_HOOK, enabled_events=["checkout.session.completed"])
        r = _preflight(_module(hooks=[partial]))
        self.assertIn("webhook: missing events: customer.subscription.updated, customer.subscription.deleted",
                      r["problems"])
        self.assertTrue(_preflight(_module(hooks=[dict(GOOD_HOOK, enabled_events=["*"])]))["ok"])

    def test_webhook_list_not_permitted_is_a_warning(self):
        bm = _module()
        bm.stripe.WebhookEndpoint.list.side_effect = RuntimeError("permission")
        r = _preflight(bm)
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["warnings"]), 1)

    def test_no_secret_values_in_output(self):
        r = _preflight(_module())
        text = repr(r)
        for value in ("sk_test_fake", "whsec_fake", "price_pro", "price_old"):
            self.assertNotIn(value, text)

    def test_debug_status_includes_preflight(self):
        bm = _module()
        with patch.dict("os.environ", dict(ENV, BILLING_DEBUG_STATUS="1")):
            body = _client(bm).get("/debug/status").json()
        self.assertIn("billing_preflight", body)
        self.assertTrue(body["billing_preflight"]["ok"])

    def test_expected_amounts_match_config(self):
        import config

        bm = _module()
        expected = {name: cents for name, _i, cents in bm._EXPECTED_PRICES}
        self.assertEqual(expected["STRIPE_PRICE_PRO"], config.TIERS_CONFIG["pro"]["price_monthly"] * 100)
        self.assertEqual(expected["STRIPE_PRICE_PREMIUM"], config.TIERS_CONFIG["premium"]["price_monthly"] * 100)
        self.assertEqual(expected["STRIPE_PRICE_PRO_YEARLY"], config.TIERS_CONFIG["pro"]["price_yearly"] * 100)
        self.assertEqual(expected["STRIPE_PRICE_PREMIUM_YEARLY"], config.TIERS_CONFIG["premium"]["price_yearly"] * 100)


if __name__ == "__main__":
    unittest.main()
