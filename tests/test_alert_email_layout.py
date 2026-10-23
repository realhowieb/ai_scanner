"""Alert email layout (sections + ticker tables) and P2-50 overlap removal."""
import unittest

from scheduler.alert_email import compose, drop_covered_breakouts, make_section, split_line


def brk(thr, tickers, **kw):
    a = {"alert_type": "breakout", "threshold": thr, **kw}
    return make_section(a, [f"{t}: BreakoutScore {s} (≥ {thr:g})" for t, s in tickers])


WATCH = make_section({"alert_type": "watchlist"}, ["SMTC: in scan results (BreakoutScore 33.4)",
                                                   "AEHR: in scan results (BreakoutScore 26.8)"])
B30 = brk(30.0, [("IOVA", 118.9), ("KOD", 69.5), ("MINT", 44.9)])
B50 = brk(50.0, [("IOVA", 118.9), ("KOD", 69.5)])


class OverlapTests(unittest.TestCase):
    def test_narrower_breakout_section_is_dropped(self):
        self.assertEqual([s["heading"] for s in drop_covered_breakouts([WATCH, B30, B50])],
                         [WATCH["heading"], B30["heading"]])
        self.assertEqual(len(drop_covered_breakouts([B50, B30])), 1)        # order doesn't matter

    def test_non_overlapping_sections_stay(self):
        other = brk(50.0, [("ZZZ", 80.0)])
        self.assertEqual(len(drop_covered_breakouts([B30, other])), 2)

    def test_watchlist_only_is_never_dropped_for_a_breakout(self):
        w = make_section({"alert_type": "watchlist"}, ["IOVA: in scan results"])
        self.assertEqual(len(drop_covered_breakouts([B30, w])), 2)


class LayoutTests(unittest.TestCase):
    def test_split_line_moves_the_threshold_to_the_heading(self):
        self.assertEqual(split_line("IOVA: BreakoutScore 118.9 (≥ 30)"), ("IOVA", "BreakoutScore 118.9"))
        self.assertEqual(split_line("KMX: BreakoutScore 47.4 (≥ 30) ⚠️ earnings in 2d"),
                         ("KMX", "BreakoutScore 47.4 ⚠️ earnings in 2d"))
        self.assertEqual(split_line("SMTC: in scan results (BreakoutScore 33.4)"), ("SMTC", "BreakoutScore 33.4"))
        self.assertEqual(split_line("DAL: in scan results ⚠️ earnings in 1d"), ("DAL", "in the latest scan ⚠️ earnings in 1d"))

    def test_html_has_headings_and_rows_not_pre(self):
        subject, text, html = compose([WATCH, B30, B50])
        self.assertEqual(subject, "📈 2 alerts: SMTC, AEHR, IOVA…")
        self.assertNotIn("<pre>", html)
        self.assertIn("<h3", html)
        self.assertIn("Breakout alert · BreakoutScore ≥ 30</h3>", html)
        self.assertIn("Watchlist alert · your watchlist names in the latest scan</h3>", html)
        self.assertEqual(html.count("<tr>"), 5 + 2 + 2)   # 5 rows, 2 header rows, 2 cards
        self.assertNotIn("(≥ 30)", html)
        self.assertIn("  IOVA   BreakoutScore 118.9", text)

    def test_long_lists_are_capped(self):
        many = brk(10.0, [(f"T{i}", 20.0) for i in range(30)])
        _s, text, html = compose([many])
        self.assertIn("…and 20 more", text)
        self.assertEqual(html.count("<tr>"), 10 + 1 + 1)   # top 10, header row, card
        self.assertIn("+ 20 more", html)
        self.assertIn("See all 30 in the Scanner", html)

    def test_html_is_escaped(self):
        sec = make_section({"alert_type": "price", "ticker": "A<B"}, ["A<B: crossed <script>"])
        self.assertNotIn("<script>", compose([sec])[2])

    def test_watchlist_only_breakout_heading(self):
        self.assertEqual(brk(40.0, [("X", 50.0)], watchlist_only=True)["heading"],
                         "Breakout alert · BreakoutScore ≥ 40 · your watchlist")


class RedesignTests(unittest.TestCase):
    QUOTES = {"IOVA": {"score": 88, "last": 4.12, "chg_pct": 12.5, "setup": "Breakout"},
              "KOD": {"score": 71, "last": 9.3, "chg_pct": -1.2, "setup": None},
              "MINT": {"score": 40, "last": 50.0, "chg_pct": 0.1, "setup": None}}

    def test_rows_carry_price_move_score_and_setup(self):
        _s, _t, html = compose([B30], quotes=self.QUOTES, watched=["kod"])
        self.assertIn("4.12", html)
        self.assertIn("+12.50%", html)
        self.assertIn(">88</span>", html)
        self.assertIn("Breakout · BreakoutScore 118.9", html)
        self.assertIn(">Watchlist</span>", html)
        self.assertIn("3 stocks", html)

    def test_rows_without_quotes_show_dashes(self):
        _s, _t, html = compose([B30])
        self.assertIn("—", html)
        self.assertIn("IOVA", html)

    def test_earnings_flag_becomes_a_tag(self):
        sec = make_section({"alert_type": "breakout", "threshold": 30},
                           ["KMX: BreakoutScore 47.4 (≥ 30) ⚠️ earnings in 0d"])
        html = compose([sec])[2]
        self.assertIn("Earnings today", html)
        self.assertNotIn("⚠️ earnings", html)

    def test_single_ticker_alert_links_to_the_stock(self):
        sec = make_section({"alert_type": "price", "ticker": "AAPL"}, ["AAPL: 201.50 crossed above 200"])
        html = compose([sec])[2]
        self.assertIn("/stocks/AAPL", html)
        self.assertIn("Open AAPL in HSF", html)

    def test_glance_counts_strongest_and_watchlist(self):
        from scheduler.alert_email import glance

        g = glance([B30, B50], self.QUOTES, ["MINT"])
        self.assertIn("3 stocks</b> cleared BreakoutScore ≥ 30.", g)
        self.assertIn("Top HSF Scores: IOVA 88, KOD 71, MINT 40.", g)
        self.assertIn("On your watchlist: MINT.", g)
        self.assertIn("2 alerts", glance([WATCH, B30]))
        self.assertIn("Strongest: SMTC, AEHR, IOVA.", glance([WATCH, B30]))   # no scores: alert order

    def test_sent_email_uses_the_shell(self):
        from ui.email_kit import shell

        html = shell(compose([B30])[2])
        self.assertIn("HSFinest.AI", html)
        self.assertEqual(html.count("border-radius:12px;padding:16px 20px"), 1)   # not double-wrapped


class OtherEmailTests(unittest.TestCase):
    def test_live_alert_card(self):
        from billing_service.realtime_alerts import _live_alert_html

        html = _live_alert_html("Price alert: AAPL 201.50 is above your 200.00 target (live)",
                                {"last": 201.5, "prev_close": 196.0}, "AAPL", "Not advice.", "https://u/x")
        self.assertIn("Price alert · AAPL", html)
        self.assertIn("201.50 is above your 200.00 target", html)
        self.assertIn("+2.81%", html)
        self.assertIn("/stocks/AAPL", html)
        self.assertIn("Unsubscribe", html)
        self.assertNotIn("<pre>", html)

    def test_intelligence_alert_and_account_emails_use_the_shell(self):
        from ui.email_utils import _branded_card, alert_html

        html = alert_html("HSF Intelligence Alert", "line one\nline <two>")
        self.assertIn("⚡ HSFinest.AI", html)
        self.assertIn("line &lt;two&gt;", html)
        self.assertNotIn("<pre>", html)
        card = _branded_card("Reset", "<p>x</p>", "https://a/b?t=1&k=2", "Reset my password", "30 minutes")
        self.assertIn("Reset my password", card)
        self.assertIn("https://a/b?t=1&amp;k=2", card)


if __name__ == "__main__":
    unittest.main()
