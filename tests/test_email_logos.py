"""Company logo chips in the HSF emails (ui.email_kit)."""
import os
import unittest
from unittest import mock

from ui import email_kit as k


class EmailLogoTests(unittest.TestCase):
    def test_ticker_leaves_marker_and_shell_resolves(self):
        cell = k.ticker("AAPL")
        self.assertIn("<!--hsf-logo:AAPL-->", cell)
        with mock.patch.dict(os.environ, {"HSF_EMAIL_LOGOS": "1"}):
            out = k.resolve_logos(cell + k.ticker("ANET") + k.ticker("TSEM"), infos={
                "AAPL": {"kind": "logo", "light": False},
                "ANET": {"kind": "logo", "light": True},
                "TSEM": {"kind": "none"},
            })
        self.assertNotIn("hsf-logo", out)
        self.assertIn("image-stock/AAPL.png", out)
        self.assertIn("background:#1e293b'><img src='https://financialmodelingprep.com/image-stock/ANET.png'", out)
        self.assertNotIn("TSEM.png", out)       # a photo: letter tile instead
        self.assertIn(">T</span><strong>TSEM</strong>", out)

    def test_switch_off_strips_markers(self):
        with mock.patch.dict(os.environ, {"HSF_EMAIL_LOGOS": "0"}):
            html = k.shell(k.card("Movers", k.ticker("AAPL")))
        self.assertNotIn("hsf-logo", html)
        self.assertNotIn("image-stock", html)
        self.assertIn("<strong>AAPL</strong>", html)

    def test_check_failure_strips_markers(self):
        with mock.patch.dict(os.environ, {"HSF_EMAIL_LOGOS": "1"}), \
                mock.patch("api.logo_check.logo_infos", side_effect=RuntimeError("down")):
            self.assertEqual(k.resolve_logos(k.ticker("AAPL")), "<strong>AAPL</strong>")

    def test_odd_symbols_get_no_marker(self):
        self.assertEqual(k.ticker("—"), "<strong>—</strong>")
        self.assertEqual(k.ticker("AAPL", logo=False), "<strong>AAPL</strong>")


if __name__ == "__main__":
    unittest.main()


class WeeklyRecapCardTests(unittest.TestCase):
    def test_recap_uses_cards_with_logos(self):
        import datetime as dt

        from services.weekly_recap import compose, recap_glance

        picks = {"n": 5, "days": 1, "avg": 0.012, "spy": 0.003, "beat": 3, "with_spy": 5,
                 "best": [{"ticker": "AMD", "ret": 0.05}]}
        _, _text, inner = compose(pro=True, picks=picks, movers=[("AAPL", 0.031)], alerts=[("TSLA", -0.02)],
                                  earnings=[("MSFT", "Tue")], monday=dt.date(2026, 10, 19), base_url="https://app")
        self.assertTrue(k.is_kit(inner))
        for t in ("AMD", "AAPL", "TSLA", "MSFT"):
            self.assertIn(f"<!--hsf-logo:{t}-->", inner)
        self.assertNotIn("<ul", inner)        # old bullet layout gone
        glance = recap_glance(picks, [("AAPL", 0.031)])
        self.assertIn("+1.2%", glance)
        self.assertIn("AAPL", glance)


class CompanyNameTests(unittest.TestCase):
    def test_short_names_and_lookup(self):
        from api import symbol_search as ss

        rows = [{"ticker": "AAPL", "name": "Apple Inc. Common Stock"},
                {"ticker": "BRK.B", "name": "Berkshire Hathaway Inc. Class B"},
                {"ticker": "TSM", "name": "Taiwan Semiconductor Manufacturing Company Ltd. American Depositary Shares"},
                {"ticker": "ZZZ", "name": "ZZZ"}]
        with mock.patch.object(ss, "_rows", return_value=rows):
            out = ss.names(["aapl", "BRK-B", "TSM", "ZZZ", "NOPE"])
        self.assertEqual(out, {"AAPL": "Apple Inc.", "BRK-B": "Berkshire Hathaway Inc. Class B",
                               "TSM": "Taiwan Semiconductor Manufacturing Company Ltd."})
