"""Day Trader -> watchlist action wiring (selector, routing, persistence, dedup)."""
import contextlib
import unittest
from unittest import mock

import ui.day_trader as dt


class FakeSt:
    """Minimal Streamlit stand-in for driving the action UI deterministically."""

    def __init__(self, *, buttons=None, selectbox=None, text=None, session=None):
        self.session_state = session if session is not None else {}
        self._buttons = buttons or {}
        self._selectbox = selectbox or {}
        self._text = text or {}
        self.messages = []  # (kind, text)

    def markdown(self, *a, **k):
        pass

    def caption(self, t="", *a, **k):
        self.messages.append(("caption", t))

    def success(self, t="", *a, **k):
        self.messages.append(("success", t))

    def info(self, t="", *a, **k):
        self.messages.append(("info", t))

    def button(self, label="", key=None, disabled=False, **k):
        return bool(self._buttons.get(key, False)) and not disabled

    def selectbox(self, label="", options=None, key=None, **k):
        options = list(options or [])
        if key in self._selectbox:
            return self._selectbox[key]
        v = self.session_state.get(key)
        if v in options:
            return v
        return options[0] if options else None

    def text_input(self, label="", key=None, **k):
        return self._text.get(key, "")

    def columns(self, n):
        return [self for _ in range(n)]

    def expander(self, *a, **k):
        return contextlib.nullcontext()

    def switch_page(self, *a, **k):
        raise RuntimeError("no navigation in test")


ROWS = [{"ticker": "NVDA", "last": 100.0}, {"ticker": "AMD", "last": 50.0}]
WLS = [
    {"id": 1, "name": "Day Trades", "is_default": True, "symbol_count": 0},
    {"id": 2, "name": "Momentum", "is_default": False, "symbol_count": 3},
]


class FeedbackTests(unittest.TestCase):
    def test_added(self):
        self.assertEqual(dt._watchlist_add_feedback("NVDA", "Momentum", {"added": ["NVDA"]}),
                         "NVDA added to Momentum.")

    def test_already_present(self):
        self.assertEqual(dt._watchlist_add_feedback("NVDA", "Momentum",
                                                    {"added": [], "already_present": ["NVDA"]}),
                         "NVDA is already in Momentum.")

    def test_failure(self):
        self.assertIn("Couldn't", dt._watchlist_add_feedback("NVDA", "Momentum", {}))


class SelectorRoutingTests(unittest.TestCase):
    def _run(self, buttons, session):
        fake = FakeSt(buttons=buttons, session=session)
        with (
            mock.patch.object(dt, "st", fake),
            mock.patch.object(dt, "_render_watchlist_action") as wl,
        ):
            dt._render_row_actions()
        return fake, wl

    def test_stale_selection_reset_to_valid(self):
        fake, _ = self._run({}, {"dt_rows": ROWS, "dt_action_ticker": "GONE"})
        # stale value cleared so the selectbox falls to a current result
        self.assertNotEqual(fake.session_state.get("dt_action_ticker"), "GONE")

    def test_chart_receives_pick(self):
        fake, _ = self._run({"dt_act_chart": True}, {"dt_rows": ROWS})
        self.assertEqual(fake.session_state["dt_show_chart"], "NVDA")

    def test_trade_plan_receives_pick(self):
        fake, _ = self._run({"dt_act_plan": True}, {"dt_rows": ROWS})
        self.assertEqual(fake.session_state["dt_show_plan"], "NVDA")

    def test_alert_receives_pick(self):
        fake, _ = self._run({"dt_act_alert": True}, {"dt_rows": ROWS})
        self.assertEqual(fake.session_state["alert_price_tk"], "NVDA")
        self.assertEqual(fake.session_state["alert_price_val"], 100.0)

    def test_watchlist_action_receives_pick(self):
        fake, wl = self._run({"dt_act_wl": True}, {"dt_rows": ROWS})
        self.assertEqual(fake.session_state["dt_show_wl"], "NVDA")
        wl.assert_called_once_with("NVDA")


class WatchlistAddTests(unittest.TestCase):
    def _run(self, *, buttons, selectbox=None, text=None, membership=None):
        fake = FakeSt(buttons=buttons, selectbox=selectbox, text=text,
                      session={"username": "tester"})
        add = mock.MagicMock(return_value={"added": ["NVDA"], "already_present": []})
        create = mock.MagicMock(return_value=99)
        member = membership or {}
        with (
            mock.patch.object(dt, "st", fake),
            mock.patch("db.watchlists.list_watchlists", return_value=WLS),
            mock.patch("db.watchlists.get_watchlist_tickers",
                       side_effect=lambda wid, u: member.get(wid, [])),
            mock.patch("db.watchlists.add_tickers_to_watchlist", add),
            mock.patch("db.watchlists.create_watchlist", create),
        ):
            dt._render_watchlist_action("NVDA")
        return fake, add, create

    def test_quick_add_targets_default(self):
        fake, add, _ = self._run(buttons={"dt_wl_quickadd": True})
        add.assert_called_once_with("tester", ["NVDA"], 1)  # default id
        self.assertTrue(any("added to Day Trades" in t for _, t in fake.messages))

    def test_add_to_non_default(self):
        fake, add, _ = self._run(buttons={"dt_wl_add": True},
                                 selectbox={"dt_wl_dest": "Momentum"})
        add.assert_called_once_with("tester", ["NVDA"], 2)  # chosen non-default id

    def test_duplicate_add_prevented(self):
        # NVDA already in the default watchlist -> quick-add disabled, not called
        fake, add, _ = self._run(buttons={"dt_wl_quickadd": True}, membership={1: ["NVDA"]})
        add.assert_not_called()
        self.assertTrue(any("Already in" in t for _, t in fake.messages))

    def test_create_new_and_add(self):
        fake, add, create = self._run(buttons={"dt_wl_create": True},
                                      text={"dt_wl_new_name": "Breakouts"})
        create.assert_called_once_with("tester", "Breakouts")
        add.assert_called_once_with("tester", ["NVDA"], 99)

    def test_requires_login(self):
        fake = FakeSt(session={})  # no username
        with mock.patch.object(dt, "st", fake):
            dt._render_watchlist_action("NVDA")
        self.assertTrue(any("Sign in" in t for _, t in fake.messages))

    def test_no_scan_triggered_on_add(self):
        # rendering the watchlist action must not invoke the day-trader screen
        with mock.patch.object(dt, "render_day_trader_panel") as scan:
            self._run(buttons={"dt_wl_quickadd": True})
        scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
