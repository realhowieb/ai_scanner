"""P2 backlog — design pass, tour, saved screens, share links, home screen."""
import importlib.util
import re
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ui import browser_prefs, discover, tour

ROOT = Path(__file__).resolve().parents[1]
GATED_PAGES = ("alerts", "brief", "day_trader", "journal", "kalshi", "settings", "stock", "today", "watchlists")


class DesignPassTests(unittest.TestCase):
    def test_every_page_hides_chrome_before_its_sign_in_gate(self):
        for name in GATED_PAGES + ("reset_password", "verify_email"):
            src = (ROOT / "pages" / f"{name}.py").read_text()
            with self.subTest(page=name):
                cfg = src.index("st.set_page_config(")
                chrome = src.index("hide_developer_chrome()")
                self.assertLess(cfg, chrome)
                gate = src.find("st.stop()")
                if gate != -1:
                    self.assertLess(chrome, gate)

    def test_page_titles_use_the_shared_header_without_emoji(self):
        for rel in ("pages/billing.py", "pages/journal.py", "pages/settings.py", "pages/reset_password.py",
                    "pages/verify_email.py", "pages/watchlists.py", "ui/today.py", "ui/methodology.py",
                    "ui/day_trader.py", "ui/kalshi_scanner.py"):
            src = (ROOT / rel).read_text()
            with self.subTest(rel=rel):
                self.assertIn("render_page_header(", src)
                self.assertNotRegex(src, r'st\.title\("')
                for title in re.findall(r'render_page_header\("([^"]+)"', src):
                    self.assertTrue(all(ord(ch) < 0x2190 for ch in title), title)   # no emoji in titles


class BrowserPrefsTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("streamlit"), "save_cookies lives in ui.auth_sessions (streamlit)")
    def test_get_put_json_roundtrip_with_fake_jar(self):
        jar = {}
        with mock.patch.object(browser_prefs, "_jar", return_value=jar), \
                mock.patch("ui.auth_sessions.save_cookies") as save:
            self.assertTrue(browser_prefs.put_json("k", {"a": ["x"]}))
            self.assertEqual(browser_prefs.get_json("k"), {"a": ["x"]})
            self.assertIsNone(browser_prefs.get("missing"))
        save.assert_called()

    def test_oversized_values_are_refused(self):
        with mock.patch.object(browser_prefs, "_jar", return_value={}):
            self.assertFalse(browser_prefs.put("k", "x" * (browser_prefs.MAX_VALUE_CHARS + 1)))

    def test_unavailable_cookies_never_raise(self):
        with mock.patch.object(browser_prefs, "_jar", side_effect=RuntimeError("no cookies")):
            self.assertIsNone(browser_prefs.get("k"))
            self.assertEqual(browser_prefs.get_json("k", default=[]), [])


class TourTests(unittest.TestCase):
    def test_five_steps_pointing_at_real_pages(self):
        self.assertEqual(len(tour.TOUR_STEPS), 5)
        for _title, body, page, _label in tour.TOUR_STEPS:
            self.assertTrue((ROOT / page).exists(), page)
            self.assertNotIn("win", body.lower().split())
        self.assertIn("not a probability of profit", tour.TOUR_STEPS[2][1])

    def test_step_clamping(self):
        self.assertEqual(tour.clamp_step(-3), 0)
        self.assertEqual(tour.clamp_step(99), len(tour.TOUR_STEPS) - 1)

    def test_done_is_read_from_browser_prefs(self):
        with mock.patch("ui.browser_prefs.get", return_value="done"):
            self.assertTrue(tour.tour_done())
        with mock.patch("ui.browser_prefs.get", return_value=None):
            self.assertFalse(tour.tour_done())

    def test_tour_is_placed_on_today_and_scanner(self):
        self.assertIn('render_tour("scanner")', (ROOT / "app.py").read_text())
        self.assertIn('render_tour("today")', (ROOT / "ui" / "today.py").read_text())


class SavedScreensTests(unittest.TestCase):
    def _df(self):
        return pd.DataFrame([
            {"Ticker": "AAA", "IsBreakout": True, "VolRel20": 2.0, "GapPct": 3.0},
            {"Ticker": "BBB", "IsBreakout": True, "VolRel20": 1.0, "GapPct": 0.0},
            {"Ticker": "CCC", "IsBreakout": False, "VolRel20": 2.0, "GapPct": 3.0},
        ])

    def test_lenses_combine_with_and_keeping_order(self):
        df = self._df()
        self.assertEqual(list(discover.apply_lenses(df, ["breakouts", "volume"])["Ticker"]), ["AAA"])
        self.assertEqual(list(discover.apply_lenses(df, ["volume", "gaps"])["Ticker"]), ["AAA", "CCC"])
        self.assertIs(discover.apply_lenses(df, []), df)
        self.assertIs(discover.apply_lenses(df, ["all"]), df)

    def test_screen_storage_is_normalized_and_bounded(self):
        raw = {" Breakout vol ": ["breakouts", "volume", "bogus"], "empty": [], "": ["gaps"], "bad": "x"}
        self.assertEqual(discover.normalize_screens(raw), {"Breakout vol": ["breakouts", "volume"]})
        screens = {}
        for i in range(discover.MAX_SCREENS + 3):
            screens = discover.save_screen(screens, f"s{i}", ["gaps"])
        self.assertEqual(len(screens), discover.MAX_SCREENS)
        self.assertIn(f"s{discover.MAX_SCREENS + 2}", screens)          # newest kept
        replaced = discover.save_screen({"a": ["gaps"]}, "a", ["volume"])
        self.assertEqual(replaced, {"a": ["volume"]})
        self.assertEqual(discover.save_screen({}, "x", ["all"]), {})      # nothing to save

    def test_apply_uses_the_pending_key_not_the_live_widget_key(self):
        src = (ROOT / "ui" / "discover.py").read_text()
        self.assertIn("st.session_state[PENDING_LENS_KEY] = list(screens[pick])", src)
        self.assertIn('selection_mode="multi"', src)
        self.assertIn('st.session_state["hsf_lens_pending"] = ["new"]', (ROOT / "ui" / "today.py").read_text())


class ShareAndHomeScreenTests(unittest.TestCase):
    def test_share_url(self):
        from ui.stock_intelligence import share_url

        with mock.patch("config.APP_BASE_URL", "https://example.test/"):
            self.assertEqual(share_url(" nvda "), "https://example.test/stock?ticker=NVDA")

    def test_shared_link_survives_sign_in(self):
        stock = (ROOT / "pages" / "stock.py").read_text()
        self.assertLess(stock.index('st.query_params.get("ticker")'), stock.index("st.stop()"))
        self.assertIn('st.session_state["hsf_after_login_page"] = "pages/stock.py"', stock)
        app = (ROOT / "app.py").read_text()
        self.assertIn('st.switch_page(st.session_state.pop("hsf_after_login_page"))', app)

    def test_home_screen_instructions_are_honest(self):
        src = (ROOT / "pages" / "settings.py").read_text()
        self.assertIn("Add to Home Screen", src)
        self.assertIn("not an offline app", src)


if __name__ == "__main__":
    unittest.main()
