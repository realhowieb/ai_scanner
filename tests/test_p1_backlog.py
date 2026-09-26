"""P1 backlog — lenses, cards, new-since-visit, recap, Today, My Stocks, nav,
Ticker Intelligence v2. Presentation only; no DB, no Streamlit runtime."""
import datetime as dt
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ui import discover, last_visit, market_scans, recap
from ui.product_copy import find_prohibited_claims
from ui.results_intelligence import consolidate_scanner_results

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
HAS_ST = importlib.util.find_spec("streamlit") is not None


def _df():
    return pd.DataFrame([
        {"Ticker": "AAA", "Why": "a", "BreakoutScore": 40, "IsBreakout": True, "VolRel20": 2.4,
         "GapPct": 4.1, "PctChange": 3.0, "Trend10D%": 12, "PreBreakoutProb%": 62, "Last": 10.0},
        {"Ticker": "BBB", "Why": "b", "BreakoutScore": 30, "IsBreakout": False, "BreakoutPos20D": 0.99,
         "VolRel20": 1.1, "GapPct": 0.5, "PctChange": 0.2, "Trend10D%": 1, "Last": 20.0},
        {"Ticker": "CCC", "Why": "c", "BreakoutScore": 20, "IsBreakout": False, "VolRel20": 1.6,
         "GapPct": -2.5, "PctChange": -2.4, "Trend10D%": -3, "Last": 30.0},
    ])


class LensTests(unittest.TestCase):
    def test_lenses_filter_without_reordering(self):
        df = _df()
        self.assertEqual(list(discover.apply_lens(df, "breakouts")["Ticker"]), ["AAA", "BBB"])
        self.assertEqual(list(discover.apply_lens(df, "volume")["Ticker"]), ["AAA", "CCC"])
        self.assertEqual(list(discover.apply_lens(df, "gaps")["Ticker"]), ["AAA", "CCC"])
        self.assertEqual(list(discover.apply_lens(df, "momentum")["Ticker"]), ["AAA"])
        self.assertEqual(list(discover.apply_lens(df, "early")["Ticker"]), ["AAA"])
        self.assertIs(discover.apply_lens(df, "all"), df)

    def test_new_lens_uses_the_new_set(self):
        self.assertEqual(list(discover.apply_lens(_df(), "new", {"CCC"})["Ticker"]), ["CCC"])

    def test_only_non_empty_lenses_are_offered(self):
        counts = discover.lens_counts(_df(), set())
        self.assertEqual(counts["all"], 3)
        opts = discover.available_lenses(counts)
        self.assertEqual(opts[0], "all")
        self.assertNotIn("new", opts)                 # no new names → lens hidden
        self.assertIn("gaps", opts)

    def test_thresholds_match_existing_definitions(self):
        from ui.results_intelligence import GAP_SIGNAL_MIN, PREBREAKOUT_SIGNAL_MIN

        self.assertEqual(GAP_SIGNAL_MIN, 2.0)
        self.assertEqual(PREBREAKOUT_SIGNAL_MIN, 50.0)
        self.assertEqual(discover.RVOL_MIN, 1.5)      # "N× avg volume" rule in ui.result_explain

    def test_mark_new_prefixes_why_only_for_new_names(self):
        out = discover.mark_new(_df(), {"BBB"})
        self.assertEqual(list(out["Why"]), ["a", f"{discover.NEW_MARK} · b", "c"])
        self.assertEqual(list(discover.mark_new(out, {"BBB"})["Why"])[1], f"{discover.NEW_MARK} · b")  # idempotent

    def test_card_view_wrapper_dispatch(self):
        calls = []
        fake_st = mock.MagicMock()
        fake_st.session_state = {discover.VIEW_KEY: "Cards"}
        with mock.patch.object(discover, "st", fake_st), \
                mock.patch("ui.result_cards.render_result_cards", side_effect=lambda df: calls.append("cards")):
            discover.with_card_view(lambda *a, **k: calls.append("table"))(_df())
        fake_st.session_state[discover.VIEW_KEY] = "Table"
        with mock.patch.object(discover, "st", fake_st):
            discover.with_card_view(lambda *a, **k: calls.append("table"))(_df())
        self.assertEqual(calls, ["cards", "table"])


class CardTests(unittest.TestCase):
    def test_card_model(self):
        from ui.result_cards import card_markdown, card_model

        row = dict(_df().iloc[0])
        row["HSF Score"] = 77
        row["Why"] = "2.4× avg volume · +4.1% gap"
        m = card_model(row)
        self.assertEqual(m["ticker"], "AAA")
        self.assertEqual(m["score"], 77)
        self.assertEqual(m["why"], ["2.4× avg volume", "+4.1% gap"])
        self.assertIn("$10.00", m["facts"])
        self.assertIn("RVOL 2.40×", m["facts"])
        self.assertTrue(card_markdown(m).startswith("**AAA** · HSF Score **77**"))


class MarketScansTests(unittest.TestCase):
    RUNS = [
        {"id": 1, "username": "cron", "label": "US_MARKET", "created_at": "2026-09-28T13:35:00+00:00"},
        {"id": 2, "username": "cron", "label": "US_MARKET", "created_at": "2026-09-28T19:35:00+00:00"},
        {"id": 3, "username": "cron", "label": "US_MARKET", "created_at": "2026-09-25T19:35:00+00:00"},
        {"id": 4, "username": "alice", "label": "US_MARKET", "created_at": "2026-09-28T20:00:00+00:00"},
        {"id": 5, "username": "cron", "label": "postmarket", "created_at": "2026-09-28T21:35:00+00:00"},
    ]

    def test_market_runs_filter_and_sort(self):
        runs = market_scans.market_runs(self.RUNS)
        self.assertEqual([r["id"] for r in runs], [2, 1, 3])
        self.assertEqual([r["id"] for r in market_scans.runs_on_day(runs, dt.date(2026, 9, 28))], [2, 1])

    def test_diff_and_tickers(self):
        self.assertEqual(market_scans.tickers_of(pd.DataFrame({"Ticker": ["a", "B", "a"]})), ["A", "B"])
        self.assertEqual(market_scans.diff_tickers(["A", "B"], ["B", "C"]), {"entered": ["C"], "left": ["A"]})

    def test_top_setups_is_the_canonical_hsf_ranking(self):
        df = _df()
        self.assertEqual([o["ticker"] for o in market_scans.top_setups(df, 2)],
                         [o["ticker"] for o in consolidate_scanner_results(df.to_dict("records"), top_n=2)])


class RecapTests(unittest.TestCase):
    def _runs(self):
        return market_scans.market_runs(MarketScansTests.RUNS)

    def test_recap_day(self):
        self.assertEqual(recap.recap_day(self._runs(), dt.datetime(2026, 9, 28, 22, 0, tzinfo=UTC)),
                         dt.date(2026, 9, 28))
        self.assertEqual(recap.recap_day(self._runs(), dt.datetime(2026, 9, 29, 12, 0, tzinfo=UTC)),
                         dt.date(2026, 9, 28))     # no scan yet today → last session
        self.assertIsNone(recap.recap_day([], dt.datetime(2026, 9, 29, tzinfo=UTC)))

    def test_titles_and_diff(self):
        day_runs = market_scans.runs_on_day(self._runs(), dt.date(2026, 9, 28))
        first = pd.DataFrame({"Ticker": ["AAA", "BBB"]})
        last = _df()
        after_close = recap.build_recap(day_runs, first, last, day=dt.date(2026, 9, 28),
                                        now=dt.datetime(2026, 9, 28, 22, 0, tzinfo=UTC))
        self.assertEqual(after_close["title"], "End-of-day recap")
        self.assertEqual(after_close["entered"], ["CCC"])
        self.assertEqual(after_close["left"], [])
        self.assertEqual(after_close["scans"], 2)
        midday = recap.build_recap(day_runs, first, last, day=dt.date(2026, 9, 28),
                                   now=dt.datetime(2026, 9, 28, 17, 0, tzinfo=UTC))
        self.assertEqual(midday["title"], "Today so far")
        prior = recap.build_recap(day_runs, first, last, day=dt.date(2026, 9, 28),
                                  now=dt.datetime(2026, 9, 29, 12, 0, tzinfo=UTC))
        self.assertEqual(prior["title"], "Last session recap · Mon Sep 28")

    def test_recap_is_descriptive_only(self):
        day_runs = market_scans.runs_on_day(self._runs(), dt.date(2026, 9, 28))
        r = recap.build_recap(day_runs, pd.DataFrame({"Ticker": ["ZZZ"]}), _df(), day=dt.date(2026, 9, 28),
                              now=dt.datetime(2026, 9, 28, 22, 0, tzinfo=UTC))
        text = "\n".join(recap.recap_lines(r))
        self.assertEqual(find_prohibited_claims(text), [])
        for word in ("return", "win", "hit rate", "profit", "gain of"):
            self.assertNotIn(word, text.lower())
        self.assertIn("Standouts by HSF Score", text)


class LastVisitAndTodayTests(unittest.TestCase):
    def test_new_tickers(self):
        cur = pd.DataFrame({"Ticker": ["A", "B", "C"]})
        self.assertEqual(last_visit.new_tickers(cur, pd.DataFrame({"Ticker": ["A"]})), {"B", "C"})
        self.assertEqual(last_visit.new_tickers(cur, None), set())      # first visit → nothing marked

    def test_watchlist_in_scan(self):
        from ui.today import watchlist_in_scan

        w = watchlist_in_scan(["AAA", "ZZZ"], _df())
        self.assertEqual([r["ticker"] for r in w["found"]], ["AAA"])
        self.assertIsInstance(w["found"][0]["score"], int)
        self.assertEqual(w["missing"], ["ZZZ"])


class PagesAndNavTests(unittest.TestCase):
    def test_today_page_exists_and_is_login_gated(self):
        src = (ROOT / "pages" / "today.py").read_text()
        self.assertIn('st.session_state.get("username")', src)
        self.assertIn("render_today(_username)", src)

    def test_my_stocks_merges_watchlists_and_alerts(self):
        src = (ROOT / "pages" / "watchlists.py").read_text()
        self.assertIn('st.tabs(["📋 Watchlist", "🔔 Alerts"])', src)
        self.assertIn("render_alerts_body(_username", src)
        self.assertIn("render_watchlists_panel", src)
        alerts = (ROOT / "pages" / "alerts.py").read_text()
        self.assertIn("render_alerts_body(_username", alerts)            # shortcuts still land here

    def test_app_wires_discover_bar_and_cards(self):
        app = (ROOT / "app.py").read_text()
        self.assertIn("df = render_discover_bar(df)", app)
        self.assertIn("render_results=with_card_view(render_results)", app)

    def test_mobile_css(self):
        from ui.chrome import CHROME_CSS

        self.assertIn("@media (max-width:640px)", CHROME_CSS)


class TickerIntelligenceV2Tests(unittest.TestCase):
    def _life(self):
        from ui.stock_intelligence import reconstruct_lifecycle

        return reconstruct_lifecycle([
            {"time": dt.datetime(2026, 9, 22, 14, 35, tzinfo=UTC), "score": 58, "status": "WATCH",
             "score_version": "1.0", "signals": ["breakout"]},
            {"time": dt.datetime(2026, 9, 25, 16, 35, tzinfo=UTC), "score": 71, "status": "STRONG",
             "score_version": "1.0", "signals": ["breakout", "gainer"]},
        ])

    def test_timeline_summary_and_lines(self):
        from ui.stock_intelligence import _timeline_line, timeline_summary

        life = self._life()
        self.assertEqual(timeline_summary(life),
                         "First recorded Sep 22, 10:35 AM ET at HSF 58 · now HSF 71 (+13) across 1 update")
        self.assertIn("First recorded by HSF", _timeline_line(life[0]))
        self.assertIn("WATCH → STRONG", _timeline_line(life[1]))
        self.assertIn("+gainer", _timeline_line(life[1]))

    def test_layout_order_and_headings(self):
        src = (ROOT / "ui" / "stock_intelligence.py").read_text()
        body = src[src.index("    _render_header(intel)\n"):]
        self.assertLess(body.index("_render_actions(intel"), body.index("_render_historical(intel)"))
        self.assertIn("#### Why HSF is showing this", src)
        self.assertIn("#### What changed since HSF first noticed it", src)
        self.assertNotIn("hsf_observations", src)                        # scanner history only

    def test_stock_page_deep_link(self):
        src = (ROOT / "pages" / "stock.py").read_text()
        self.assertIn('st.query_params.get("ticker")', src)
        self.assertIn('st.query_params.pop("ticker", None)', src)


if __name__ == "__main__":
    unittest.main()
