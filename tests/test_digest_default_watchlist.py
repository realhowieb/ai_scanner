"""The morning digest and evening wrap use the DEFAULT watchlist only.

2026-09-29: an account with 8 watchlists got a 40-ticker digest because every
list was merged. Today shows the default list, so the emails do too.
"""
import contextlib
import io
import unittest
from unittest import mock

from scheduler.morning_digest import default_watchlist_tickers

LISTS = [{"id": 2, "name": "WLIST", "is_default": False},
         {"id": 34, "name": "SEP 17", "is_default": True},
         {"id": 21, "name": "July 13", "is_default": False}]
ITEMS = {2: ["aapl", "MSFT"], 34: ["nvda", "SMTC", "NVDA", ""], 21: ["TSLA"] * 16}


def fetch(wid, email):
    return ITEMS.get(wid, [])


class DefaultWatchlistTests(unittest.TestCase):
    def test_only_the_default_list(self):
        self.assertEqual(default_watchlist_tickers("a@example.com", lambda e: LISTS, fetch), ["NVDA", "SMTC"])

    def test_first_list_when_none_is_marked_default(self):
        lists = [dict(w, is_default=False) for w in LISTS]
        self.assertEqual(default_watchlist_tickers("a@example.com", lambda e: lists, fetch), ["AAPL", "MSFT"])

    def test_no_lists(self):
        self.assertEqual(default_watchlist_tickers("a@example.com", lambda e: [], fetch), [])
        self.assertEqual(default_watchlist_tickers("a@example.com", lambda e: None, fetch), [])

    def test_digest_and_wrap_price_only_the_default_list(self):
        import scheduler.evening_wrap as ew
        import scheduler.morning_digest as md

        for job in ("digest", "evening"):
            asked = []
            patches = [
                mock.patch("config.MORNING_DIGEST_ENABLED", True), mock.patch("config.MORNING_DIGEST_MAX_USERS", 500),
                mock.patch("db.users.load_users", return_value={"a@example.com": {"tier": "pro"}}),
                mock.patch("db.users.is_admin_from_db", return_value=False),
                mock.patch("db.email_verification.is_email_verified", return_value=True),
                mock.patch("db.watchlists.list_watchlists", return_value=LISTS),
                mock.patch("db.watchlists.get_watchlist_tickers", side_effect=fetch),
                mock.patch("market_data.build_day_trader_metrics",
                           side_effect=lambda t, **k: asked.append(tuple(t)) or []),
                mock.patch("ui.email_utils.send_digest_email", return_value=True),
                mock.patch("db.email_prefs.wants_email", return_value=True),
                mock.patch("db.email_prefs.unsubscribe_url", return_value=None),
                mock.patch("db.email_job_runs.record_email_run"),
                mock.patch.object(md, "mark_sent"),
            ]
            if job == "digest":
                patches += [mock.patch.object(md, n, return_value=v) for n, v in (
                    ("_latest_snapshot_df", None), ("_market_gappers", []), ("_earnings_today", set()),
                    ("_earnings_days_map", {}), ("_watchlist_notes", []), ("_compose", ("h", "t")))]
                run = lambda: md.run_morning_digest(force=True)  # noqa: E731
            else:
                patches += [mock.patch.object(ew, n, return_value=v) for n, v in (
                    ("_market_close_context", {}), ("_day_movers", ([], [])), ("_tomorrow_setups", ([], [])),
                    ("_todays_events", []), ("_tomorrows_earnings", []), ("_compose_wrap", ("h", "t")))]
                patches.append(mock.patch.object(md, "_latest_snapshot_df", return_value=None))
                run = lambda: ew.run_evening_wrap(force=True)  # noqa: E731
            with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
                for p in patches:
                    stack.enter_context(p)
                run()
            self.assertEqual(asked, [("NVDA", "SMTC")], job)


if __name__ == "__main__":
    unittest.main()
