"""Admin accounts receive the morning digest and alert emails; the digest logs
why it skipped each account and counts only accepted sends; the Admin Users
page never silently downgrades an account's stored tier.

Live finding (2026-09-28): the digest counted a stored tier of "admin" as Free
and ignored the DB is_admin flag, so an admin account got no morning email.
"""
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

USERS = {
    "admin-tier@example.com": {"tier": "admin"},
    "admin-flag@example.com": {"tier": "basic"},     # is_admin flag set in DB
    "pro@example.com": {"tier": "pro"},
    "free@example.com": {"tier": "basic"},
    "unverified@example.com": {"tier": "premium"},
    "nowatch@example.com": {"tier": "premium"},
    "bounce@example.com": {"tier": "premium"},
    "localuser": {"tier": "premium"},
}


def run_digest(send_result=lambda to: to != "bounce@example.com"):
    import scheduler.morning_digest as md

    sends, composed = [], {}

    def fake_send(to_address, **kw):
        sends.append(to_address)
        return send_result(to_address)

    def fake_compose(email, watch_rows, gappers, earnings_hits, picks, **kw):
        composed[email] = picks
        return "h", "t"

    def tickers(wl_id, email):
        return [] if email == "nowatch@example.com" else ["AAPL"]

    out = io.StringIO()
    with mock.patch("config.MORNING_DIGEST_ENABLED", True), \
         mock.patch("config.MORNING_DIGEST_MAX_USERS", 500), \
         mock.patch.object(md, "_latest_snapshot_df", return_value=object()), \
         mock.patch.object(md, "_todays_setups", return_value=([], [])), \
         mock.patch.object(md, "_market_gappers", return_value=[]), \
         mock.patch.object(md, "_earnings_today", return_value=set()), \
         mock.patch.object(md, "_earnings_days_map", return_value={}), \
         mock.patch.object(md, "_watchlist_notes", return_value=[]), \
         mock.patch.object(md, "_compose", side_effect=fake_compose), \
         mock.patch.object(md, "_prebreakout_picks", return_value=[{"symbol": "X"}]), \
         mock.patch("db.users.load_users", return_value=dict(USERS)), \
         mock.patch("db.users.is_admin_from_db", side_effect=lambda u: u in (
             "admin-tier@example.com", "admin-flag@example.com")), \
         mock.patch("db.email_verification.is_email_verified",
                    side_effect=lambda u: u != "unverified@example.com"), \
         mock.patch("db.watchlists.list_watchlists", return_value=[{"id": 1}]), \
         mock.patch("db.watchlists.get_watchlist_tickers", side_effect=tickers), \
         mock.patch("market_data.build_day_trader_metrics", return_value=[]), \
         mock.patch("ui.email_utils.send_digest_email", side_effect=fake_send), \
         mock.patch("db.earnings.mark_earnings_refreshed_today") as mark, \
         redirect_stdout(out):
        md.run_morning_digest(force=True)
    return sends, composed, out.getvalue(), mark


class MorningDigestEligibilityTests(unittest.TestCase):
    def test_admin_accounts_receive_the_digest(self):
        sends, composed, _, _ = run_digest()
        self.assertIn("admin-tier@example.com", sends)
        self.assertIn("admin-flag@example.com", sends)
        self.assertIn("pro@example.com", sends)
        self.assertNotIn("free@example.com", sends)
        # Admins count as Premium for PreBreakout picks; Pro does not get them.
        self.assertEqual(composed["admin-flag@example.com"], [{"symbol": "X"}])
        self.assertEqual(composed["pro@example.com"], [])

    def test_skip_reasons_logged_and_only_accepted_sends_counted(self):
        sends, _, log, mark = run_digest()
        self.assertIn("bounce@example.com", sends)          # attempted...
        line = [l for l in log.splitlines() if l.startswith("[morning_digest] sent")][0]
        self.assertIn("sent 3 digest(s)", line)             # ...but not counted
        for reason in ("plan_below_pro=1", "unverified=1", "empty_watchlist=1",
                       "send_failed=1", "not_email=1"):
            self.assertIn(reason, line)
        mark.assert_called_once()

    def test_throttle_not_marked_when_every_send_fails(self):
        _, _, log, mark = run_digest(send_result=lambda to: False)
        self.assertIn("sent 0 digest(s)", log)
        mark.assert_not_called()


class AlertEmailEligibilityTests(unittest.TestCase):
    def test_admin_flag_allows_alert_email_whatever_plan_is_stored(self):
        import scheduler.alert_runner as ar

        with mock.patch.object(ar, "_tier_key_for_user", return_value="basic"), \
             mock.patch("db.users.is_admin_from_db", side_effect=lambda u: u == "a@example.com"):
            self.assertTrue(ar._email_allowed_for_tier("a@example.com"))
            self.assertFalse(ar._email_allowed_for_tier("f@example.com"))
        with mock.patch.object(ar, "_tier_key_for_user", return_value="pro"):
            self.assertTrue(ar._email_allowed_for_tier("p@example.com"))


class AdminUsersTierPickerTests(unittest.TestCase):
    def test_stored_tier_stays_selected(self):
        src = (ROOT / "ui" / "admin_users.py").read_text()
        self.assertIn("tier_options.index(current_tier)", src)
        self.assertNotIn('row["tier"] if row["tier"] in ["basic", "pro", "premium"] else "basic"', src)


if __name__ == "__main__":
    unittest.main()
