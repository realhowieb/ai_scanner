"""Admin sign-up funnel (api.admin_analytics.signup_funnel): counts per step for the
period's new accounts, overall and by sign-up source. Pure function, no database."""
import unittest

from api.admin_analytics import signup_funnel
from ui.acquisition import user_hash


def acct(user, *, verified=False, engaged=False, tier="basic", sub=False, trial=False):
    return {"user": user, "verified": verified, "engaged": engaged, "tier": tier,
            "has_subscription": sub, "trial": trial}


class SignupFunnelTests(unittest.TestCase):
    def test_steps_rates_and_sources(self):
        data = {"available": True, "errors": {},
                "accounts": [acct("a@x.com", verified=True, engaged=True, tier="pro", sub=True),
                             acct("b@x.com", verified=True, engaged=True, trial=True, tier="basic"),
                             acct("c@x.com", verified=True),
                             acct("d@x.com")],
                "sources": {user_hash("a@x.com"): "reddit", user_hash("b@x.com"): "reddit",
                            user_hash("c@x.com"): "direct"}}
        f = signup_funnel(data)
        self.assertEqual([s["users"] for s in f["steps"]], [4, 3, 2, 1])
        self.assertEqual(f["steps"][1]["of_signups"], 0.75)
        self.assertEqual(f["invite_trials"], 1)
        self.assertEqual([(r["source"], r["signed_up"], r["paid"]) for r in f["by_source"]],
                         [("Reddit", 2, 1), ("Direct", 1, 0), ("Not recorded", 1, 0)])

    def test_pro_tier_without_a_subscription_is_not_paid(self):
        f = signup_funnel({"available": True, "accounts": [acct("a@x.com", tier="pro")], "sources": {}})
        self.assertEqual(f["steps"][3]["users"], 0)

    def test_empty_period_and_unavailable(self):
        f = signup_funnel({"available": True, "accounts": [], "sources": {}, "errors": {"alerts": "X"}})
        self.assertEqual([s["of_signups"] for s in f["steps"]], [None] * 4)
        self.assertEqual(f["unavailable"], ["alerts"])
        self.assertIsNone(signup_funnel({"available": False}))


if __name__ == "__main__":
    unittest.main()
