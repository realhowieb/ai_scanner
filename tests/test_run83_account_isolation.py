"""Run 83 (B2) — account-owned session state never crosses to another account.

Scenario from the Run 82 audit: user A signs out and user B signs in on the
same browser tab. A's watchlist quotes used to survive logout and appear on B's
Market Brief. These tests drive the app's own logout clearing
(clear_account_session_state), the identity boundary app.py runs after every
sign-in (enforce_account_boundary), and the real readers on Market Brief, Today
and Stock Intelligence.
"""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from ui.app_session import (
    ACCOUNT_OWNER_KEY,
    ACCOUNT_SESSION_KEYS,
    clear_account_session_state,
    enforce_account_boundary,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_ST = importlib.util.find_spec("streamlit") is not None

A, B = "alice@example.com", "bob@example.com"
A_ROWS = [{"ticker": "ALICEPICK", "last": 12.3, "chg_pct": 4.2}]
BROWSER_PREFS = {"hsf_results_view": "cards", "hsf_lens": ["momentum"], "hsf_tour": "done"}


class SS(dict):
    """Minimal st.session_state stand-in (mapping + attribute reads)."""
    __getattr__ = dict.get


def signed_in_as_a() -> SS:
    s = SS(username=A, display_name="Alice", tier="pro", tier_key="pro", user_id=A, is_admin=False,
           active_watchlist_tickers=["ALICEPICK"], active_watchlist_id=11,
           active_watchlist_quote_rows=list(A_ROWS), _watchlist_prior_rows={"ALICEPICK": {}},
           _loaded_user_settings=True, _portal_url="https://billing.stripe.test/p/ALICE",
           dt_watch_symbols="ALICEPICK", dt_watch_baseline={"ALICEPICK": 12.0},
           alert_price_tk="ALICEPICK", **BROWSER_PREFS)
    enforce_account_boundary(s, A)
    return s


ACCOUNT_KEYS_IN_FIXTURE = ("active_watchlist_tickers", "active_watchlist_id", "active_watchlist_quote_rows",
                           "_watchlist_prior_rows", "_loaded_user_settings", "_portal_url",
                           "dt_watch_symbols", "dt_watch_baseline", "alert_price_tk")


class BoundaryTests(unittest.TestCase):
    def test_same_user_reruns_keep_their_state(self):
        s = signed_in_as_a()
        for _ in range(3):
            self.assertFalse(enforce_account_boundary(s, A))
        self.assertEqual(s["active_watchlist_quote_rows"], A_ROWS)
        self.assertEqual(s["active_watchlist_tickers"], ["ALICEPICK"])

    def test_logout_clears_account_state_and_keeps_browser_prefs(self):
        s = signed_in_as_a()
        clear_account_session_state(s)                               # what logout runs
        for key in ACCOUNT_KEYS_IN_FIXTURE + ("username", "tier", "tier_key"):
            self.assertNotIn(key, s, key)
        for key, value in BROWSER_PREFS.items():
            self.assertEqual(s[key], value, key)                    # browser conveniences stay

    def test_identity_change_clears_state_even_without_logout(self):
        s = signed_in_as_a()
        # A different account signs in on this tab without the logout path having run
        # (session restore, a future auth flow): the identity keys are B's.
        s.update(username=B, display_name="Bob", tier="premium", user_id=B,
                 hsf_after_login_page="pages/stock.py")
        self.assertTrue(enforce_account_boundary(s, B))
        for key in ACCOUNT_KEYS_IN_FIXTURE + ("tier_key",):
            self.assertNotIn(key, s, key)
        self.assertEqual((s["username"], s["display_name"], s["tier"]), (B, "Bob", "premium"))
        self.assertEqual(s["hsf_after_login_page"], "pages/stock.py")   # B's shared link survives
        for key, value in BROWSER_PREFS.items():
            self.assertEqual(s[key], value, key)

    def test_owner_marker_survives_logout_and_is_not_the_raw_email(self):
        s = signed_in_as_a()
        clear_account_session_state(s)
        self.assertIn(ACCOUNT_OWNER_KEY, s)
        self.assertNotIn(ACCOUNT_OWNER_KEY, ACCOUNT_SESSION_KEYS)
        self.assertNotIn("alice", s[ACCOUNT_OWNER_KEY])
        s["username"] = B
        self.assertTrue(enforce_account_boundary(s, B))

    def test_every_audited_account_key_is_cleared(self):
        for key in ("active_watchlist_quote_rows", "_watchlist_prior_rows", "_loaded_user_settings",
                    "_portal_url", "dt_watch_baseline", "latest_results_df", "pt_secret"):
            self.assertIn(key, ACCOUNT_SESSION_KEYS, key)

    def test_app_enforces_the_boundary_before_any_page_switch(self):
        app = (ROOT / "app.py").read_text()
        boundary = app.index("enforce_account_boundary(st.session_state, username)")
        self.assertLess(app.index("if not authed:"), boundary)
        self.assertLess(boundary, app.index('st.session_state.get("hsf_after_login_page")'))
        self.assertLess(boundary, app.index("should_land_on_today(st.session_state, username)"))


def a_then_b(*, via_logout: bool) -> SS:
    s = signed_in_as_a()
    if via_logout:
        clear_account_session_state(s)
    s.update(username=B, tier="pro", user_id=B, display_name="Bob")
    enforce_account_boundary(s, B)
    return s


@unittest.skipUnless(HAS_ST, "needs streamlit")
class ReaderTests(unittest.TestCase):
    """The screens that showed A's data to B now read B's own account."""

    def brief_rows(self, s, b_tickers):
        import ui.market_brief as mb

        wls = [{"id": 22, "name": "Bob list"}] if b_tickers else []
        with mock.patch.object(mb, "st", mock.MagicMock(session_state=s)), \
             mock.patch("db.watchlists.list_watchlists", return_value=wls) as lw, \
             mock.patch("db.watchlists.get_watchlist_tickers", return_value=b_tickers), \
             mock.patch("market_data.build_day_trader_metrics",
                        side_effect=lambda t, **k: [{"ticker": x, "last": 1.0, "chg_pct": 0.0} for x in t]):
            rows = mb._watchlist_rows(B)
        return rows, lw

    def test_market_brief_shows_b_own_watchlist(self):
        for via_logout in (True, False):
            s = a_then_b(via_logout=via_logout)
            rows, lw = self.brief_rows(s, ["BOBPICK"])
            self.assertEqual([r["ticker"] for r in rows], ["BOBPICK"], via_logout)
            lw.assert_called_once_with(B)

    def test_market_brief_empty_b_watchlist_does_not_fall_back_to_a(self):
        for via_logout in (True, False):
            rows, _ = self.brief_rows(a_then_b(via_logout=via_logout), [])
            self.assertEqual(rows, [], via_logout)

    def test_today_watchlist_is_b_own(self):
        import ui.today as today

        s = a_then_b(via_logout=False)
        with mock.patch.object(today, "st", mock.MagicMock(session_state=s)), \
             mock.patch("ui.user_cache.get_default_watchlist_id", return_value=22) as did, \
             mock.patch("ui.user_cache.get_watchlist_tickers", return_value=["BOBPICK"]):
            self.assertEqual(today._user_watchlist(B), ["BOBPICK"])
        did.assert_called_once_with(B)

    def test_stock_intelligence_watch_status_does_not_carry_over(self):
        import ui.stock_intelligence as si

        s = a_then_b(via_logout=False)
        with mock.patch.object(si, "st", mock.MagicMock(session_state=s)):
            self.assertFalse(si._is_watched("ALICEPICK"))
            self.assertEqual(si._watch_label("ALICEPICK"), "☆ Watch")


if __name__ == "__main__":
    unittest.main()
