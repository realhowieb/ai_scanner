import os
import unittest


class EarningsSourcesTest(unittest.TestCase):
    def setUp(self):
        for k in ("FMP_API_KEY", "FINNHUB_API_KEY"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k in ("FMP_API_KEY", "FINNHUB_API_KEY"):
            os.environ.pop(k, None)

    def test_no_key_is_noop(self):
        from data import earnings_sources as es

        self.assertEqual(es.fetch_earnings_window_fmp("2026-06-28", "2026-10-28"), {})
        self.assertEqual(es.fetch_earnings_window("2026-06-28", "2026-10-28"), ({}, "none"))

    def test_fmp_parsing(self):
        from datetime import date, timedelta

        from data import earnings_sources as es

        # Relative to today — the source keeps only future (>= today) dates, so
        # hardcoded dates rot as the clock crosses them.
        today = date.today()
        aapl_early = today + timedelta(days=9)
        aapl_late = today + timedelta(days=100)   # later -> ignored (earliest wins)
        brkb_d = today + timedelta(days=12)
        notime_d = today + timedelta(days=40)
        old_d = today - timedelta(days=2000)       # past -> ignored
        win_start = (today - timedelta(days=20)).isoformat()
        win_end = (today + timedelta(days=150)).isoformat()

        os.environ["FMP_API_KEY"] = "test"
        es._get_json = lambda url, timeout=20.0: [
            {"symbol": "AAPL", "date": aapl_early.isoformat(), "time": "amc"},
            {"symbol": "AAPL", "date": aapl_late.isoformat(), "time": "amc"},
            {"symbol": "BRK.B", "date": brkb_d.isoformat(), "time": "bmo"},  # dot -> dash
            {"symbol": "OLD", "date": old_d.isoformat(), "time": "amc"},
            {"symbol": "NOTIME", "date": notime_d.isoformat()},  # missing time -> None
        ]
        m = es.fetch_earnings_window_fmp(win_start, win_end)
        self.assertEqual(m["AAPL"], (aapl_early, "AMC"))
        self.assertEqual(m["BRK-B"], (brkb_d, "BMO"))
        self.assertNotIn("OLD", m)
        self.assertIsNone(m["NOTIME"][1])

    def test_finnhub_fallback_when_no_fmp(self):
        from datetime import date, timedelta

        from data import earnings_sources as es

        # Dates relative to today so the test doesn't rot as the clock advances
        # (earnings sources filter out already-reported dates).
        today = date.today()
        msft_d = (today + timedelta(days=6)).isoformat()
        tsla_d = (today + timedelta(days=3)).isoformat()
        win_start = (today - timedelta(days=20)).isoformat()
        win_end = (today + timedelta(days=90)).isoformat()

        os.environ["FINNHUB_API_KEY"] = "test"
        es._get_json = lambda url, timeout=20.0: {
            "earningsCalendar": [
                {"symbol": "MSFT", "date": msft_d, "hour": "amc"},
                {"symbol": "TSLA", "date": tsla_d, "hour": "bmo"},
            ]
        }
        m, src = es.fetch_earnings_window(win_start, win_end)
        self.assertEqual(src, "finnhub")
        self.assertEqual(m["MSFT"][1], "AMC")
        self.assertEqual(m["TSLA"][1], "BMO")

    def test_time_normalization(self):
        from data.earnings_sources import _norm_time

        self.assertEqual(_norm_time("amc"), "AMC")
        self.assertEqual(_norm_time("BMO"), "BMO")
        self.assertEqual(_norm_time("after market close"), "AMC")
        self.assertIsNone(_norm_time("dmh"))
        self.assertIsNone(_norm_time(None))


if __name__ == "__main__":
    unittest.main()
