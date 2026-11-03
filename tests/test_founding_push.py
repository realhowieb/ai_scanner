"""Founding 100 push: the launch and reminder emails, the weekly recap card and the sale
alert to the admins (services/founding_push.py)."""
import datetime as dt
import importlib.util as _ilu
import unittest
from unittest import mock

from services import founding_push as push
from services.social_posts import check_claims

TIERS = [
    {"tier": "founder_pro", "label": "Founder Pro", "price_cents": 49900, "capacity": 75, "remaining": 73, "on_sale": True,
     "five_year": {"monthly_cents": 2500, "five_years_monthly_cents": 150000, "price_cents": 49900, "saved_cents": 100100}},
    {"tier": "founder_elite", "label": "Founder Elite", "price_cents": 99900, "capacity": 25, "remaining": 24, "on_sale": True,
     "five_year": {"monthly_cents": 4000, "five_years_monthly_cents": 240000, "price_cents": 99900, "saved_cents": 140100}},
]
OPEN = {"open": True, "checkout_open": True, "remaining": 97, "tiers": TIERS}
BASE = "https://app.hsfinest.com"
# The API and billing webhook need fastapi; the core-only CI jobs skip those tests.
_NEEDS_API = unittest.skipUnless(_ilu.find_spec("fastapi"), "needs fastapi")


def sales_copy(text: str) -> str:
    """The email minus the fixed lines that use claim words in their plain sense: the founder
    rate ("50% off"), the refund guarantee and the no-returns disclaimer."""
    for safe in (push.EXTRAS, "Every founding payment has a 14-day full refund guarantee.",
                 "It makes no promise of returns."):
        text = text.replace(safe, "")
    return text


class OfferTests(unittest.TestCase):
    def test_open_closed_and_sold_out(self):
        self.assertEqual(len(push.offer(OPEN)["tiers"]), 2)
        self.assertIsNone(push.offer({"open": False, "tiers": []}))
        sold = {**OPEN, "tiers": [{**t, "remaining": 0} for t in TIERS]}
        self.assertIsNone(push.offer(sold))
        one = {**OPEN, "tiers": [TIERS[0], {**TIERS[1], "on_sale": False}]}
        self.assertEqual([t["tier"] for t in push.offer(one)["tiers"]], ["founder_pro"])

    def test_page_url_is_tagged_for_the_marketing_report(self):
        self.assertEqual(push.page_url(BASE + "/", "launch"),
                         BASE + "/founding-100?utm_source=newsletter&utm_medium=email&utm_campaign=founding-launch")

    def test_unknown_membership_counts_as_founder(self):
        with mock.patch("db.founding.membership", side_effect=RuntimeError("db down")):
            self.assertTrue(push.is_founder("a@x.com"))
        with mock.patch("db.founding.membership", return_value={"status": "active"}):
            self.assertTrue(push.is_founder("a@x.com"))
        with mock.patch("db.founding.membership", return_value={"status": "refunded"}):
            self.assertFalse(push.is_founder("a@x.com"))
        with mock.patch("db.founding.membership", return_value=None):
            self.assertFalse(push.is_founder("a@x.com"))


class ComposeTests(unittest.TestCase):
    def test_launch_email(self):
        m = push.compose("launch", push.offer(OPEN), base=BASE, note="Hi, it's Howard.\n\nThanks for trying HSF.")
        self.assertIn("Founding 100 is open", m["subject"])
        for s in ("Founder Pro: $499 once, 73 of 75 seats left.", "Five years of Pro paying monthly would be $1,500.",
                  "Five years of Premium (what Elite includes) paying monthly would be $2,400.",
                  "Founder Elite: $999 once, 24 of 25 seats left.", "14-day full refund guarantee",
                  "Hi, it's Howard.", "Thanks for trying HSF.", "utm_campaign=founding-launch"):
            self.assertIn(s, m["text"])
        self.assertIn("See the Founding 100", m["html"])
        self.assertNotIn("<script", m["html"])
        check_claims(sales_copy(m["text"]))

    def test_reminder_email_leads_with_seats_left(self):
        m = push.compose("reminder", push.offer(OPEN), base=BASE)
        self.assertEqual(m["subject"], "97 of 100 Founding seats left")
        self.assertIn("utm_campaign=founding-reminder", m["text"])
        check_claims(sales_copy(m["text"]))

    def test_note_is_escaped(self):
        m = push.compose("launch", push.offer(OPEN), base=BASE, note="<b>hi</b>")
        self.assertNotIn("<b>hi</b>", m["html"])

    def test_weekly_card(self):
        text, html = push.weekly_card(BASE, OPEN)
        self.assertIn("97 of 100 seats left", text)
        self.assertIn("utm_campaign=founding-weekly-recap", text)
        self.assertIn("Founding 100", html)
        check_claims(text)
        self.assertIsNone(push.weekly_card(BASE, {"open": False, "tiers": []}))

    def test_weekly_recap_email_carries_the_card(self):
        from services import weekly_recap

        card = push.weekly_card(BASE, OPEN)
        args = dict(pro=False, picks={"n": 0}, movers=[], alerts=[], earnings=[], monday=dt.date(2026, 11, 2), base_url=BASE)
        _s, text, html = weekly_recap.compose(**args, founding=card)
        self.assertIn("97 of 100 seats left", text)
        self.assertIn("See the Founding 100", html)
        _s, text, _h = weekly_recap.compose(**args)
        self.assertNotIn("Founding", text)


class RecipientTests(unittest.TestCase):
    USERS = {"a@x.com": {}, "b@x.com": {}, "c@x.com": {}, "d@x.com": {"is_active": False}, "e@x.com": {},
             "f@x.com": {}, "nope": {}}

    def _run(self, campaign, sent=None, launch=None):
        sent = sent or {}
        launch = launch or {}

        def emails_sent(kind, ref):
            self.assertEqual(kind, push.KIND)
            return sent if ref == campaign else launch

        with mock.patch("db.users.load_users", return_value=self.USERS), \
                mock.patch("db.referrals.emails_sent", side_effect=emails_sent), \
                mock.patch("db.email_verification.is_email_verified", side_effect=lambda e: e != "c@x.com"), \
                mock.patch("db.email_prefs.wants_email", side_effect=lambda e, k: (k == "tips" and e != "e@x.com")), \
                mock.patch.object(push, "is_founder", side_effect=lambda e: e == "f@x.com"):
            return push.recipients(campaign)

    def test_launch_skips_inactive_unverified_unsubscribed_founders_and_repeats(self):
        out = self._run("launch", sent={"b@x.com": "t"})
        self.assertEqual(out["to"], ["a@x.com"])
        self.assertEqual(out["skipped"], {"already_sent": 1, "unverified": 1, "not_eligible": 2, "unsubscribed": 1,
                                          "founder": 1})

    def test_reminder_only_to_people_who_got_the_launch(self):
        out = self._run("reminder", launch={"b@x.com": "t", "f@x.com": "t"})
        self.assertEqual(out["to"], ["b@x.com"])
        self.assertEqual(out["skipped"]["no_launch_email"], 3)

    def test_send_claims_each_account_and_releases_failures(self):
        claims, released = [], []
        with mock.patch.object(push, "recipients", return_value={"to": ["a@x.com", "b@x.com", "c@x.com"], "skipped": {}}), \
                mock.patch.object(push, "_base", return_value=BASE), \
                mock.patch("db.referrals.claim_email", side_effect=lambda u, k, r: (claims.append((u, k, r)), u != "c@x.com")[1]), \
                mock.patch("db.referrals.release_email", side_effect=lambda u, k, r: released.append(u)), \
                mock.patch("db.email_prefs.unsubscribe_url", return_value="https://u"), \
                mock.patch("ui.email_utils.send_digest_email", side_effect=lambda to, *a, **k: to == "a@x.com") as sender:
            out = push.send("launch", status=OPEN)
        self.assertEqual((out["sent"], out["failed"], out["skipped"]), (1, 1, {"already_sent": 1}))
        self.assertEqual(released, ["b@x.com"])
        self.assertEqual(claims[0], ("a@x.com", "founding_push", "launch"))
        self.assertEqual(sender.call_args.kwargs["unsubscribe_url"], "https://u")

    def test_send_does_nothing_when_closed(self):
        with mock.patch.object(push, "recipients") as who:
            self.assertEqual(push.send("launch", status={"open": False})["sent"], 0)
        who.assert_not_called()


class SaleAlertTests(unittest.TestCase):
    def test_alert_names_tier_number_amount_and_seats(self):
        avail = {"tiers": {"founder_pro": {"capacity": 75, "remaining": 74}, "founder_elite": {"capacity": 25, "remaining": 25}},
                 "sold": 1, "capacity": 100}
        with mock.patch("db.founding.membership", return_value={"tier": "founder_pro", "membership_number": 1, "status": "active"}), \
                mock.patch("db.founding_refunds.payments", return_value=[{"kind": "purchase", "amount_cents": 49900}]), \
                mock.patch("db.founding.availability", return_value=avail), \
                mock.patch.object(push, "_base", return_value=BASE), \
                mock.patch("services.ops_notify.notify_ops") as notify:
            self.assertTrue(push.sale_alert("buyer@x.com"))
        subject, text = notify.call_args.args
        self.assertEqual(subject, "Founding sale: Founder Pro #001, $499.00")
        self.assertIn("Founder Pro 74 of 75 left", text)
        self.assertIn("1 of 100 sold", text)
        self.assertIn("buyer@x.com", text)
        self.assertTrue(notify.call_args.kwargs["email"])

    def test_alert_never_raises(self):
        with mock.patch("db.founding.membership", side_effect=RuntimeError("down")):
            self.assertFalse(push.sale_alert("buyer@x.com"))

    @_NEEDS_API
    def test_webhook_sends_the_alert_after_the_confirmation(self):
        from billing_service import engine

        with mock.patch("services.founding_push.sale_alert") as alert:
            engine._founding_sale_alert("buyer@x.com", upgrade=True)
        alert.assert_called_once_with("buyer@x.com", upgrade=True)
        with mock.patch("services.founding_push.sale_alert", side_effect=RuntimeError("boom")):
            engine._founding_sale_alert("buyer@x.com")  # never raises


class SocialPostTests(unittest.TestCase):
    def test_friday_post_links_the_public_page(self):
        from services import social_posts

        self.assertEqual(social_posts.compose_founding("2026-W45", status=OPEN)["page"], "/founding-100")


if __name__ == "__main__":
    unittest.main()


@_NEEDS_API
class AdminApiTests(unittest.TestCase):
    ADMIN = {"username": "admin@x.com"}

    def test_send_refuses_when_closed_or_reminder_first(self):
        from api import founding as svc
        from api.errors import CodedError

        with mock.patch.object(push, "offer", return_value=None):
            with self.assertRaises(CodedError) as e:
                svc.push_send(self.ADMIN, "launch")
            self.assertEqual(e.exception.code, "push_closed")
        with mock.patch.object(push, "offer", return_value=push.offer(OPEN)), \
                mock.patch.object(push, "campaign_state", return_value={"sent": 0, "first_sent_at": None}):
            with self.assertRaises(CodedError) as e:
                svc.push_send(self.ADMIN, "reminder")
            self.assertEqual(e.exception.code, "push_no_launch")
        with self.assertRaises(CodedError):
            svc.push_send(self.ADMIN, "spam")

    def test_preview_sends_nothing(self):
        from api import founding as svc

        with mock.patch.object(push, "offer", return_value=push.offer(OPEN)), \
                mock.patch.object(push, "_base", return_value=BASE), \
                mock.patch.object(push, "recipients", return_value={"to": ["a@x.com"], "skipped": {"founder": 2}}), \
                mock.patch.object(push, "campaign_state", return_value={"sent": 0, "first_sent_at": None}), \
                mock.patch.object(push, "send") as send:
            out = svc.push_preview("launch", "Hello")
        send.assert_not_called()
        self.assertEqual((out["recipients"], out["skipped"], out["on_sale"]), (1, {"founder": 2}, True))
        self.assertIn("Hello", out["html"])
        self.assertIn("HSFinest.com", out["html"])  # the full branded shell, as it lands in an inbox
