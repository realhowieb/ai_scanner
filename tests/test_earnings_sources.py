import datetime as dt
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
        from data import earnings_sources as es

        os.environ["FMP_API_KEY"] = "test"
        es._get_json = lambda url, timeout=20.0: [
            {"symbol": "AAPL", "date": "2026-07-30", "time": "amc"},
            {"symbol": "AAPL", "date": "2026-10-29", "time": "amc"},  # later -> ignored
            {"symbol": "BRK.B", "date": "2026-08-02", "time": "bmo"},  # dot -> dash
            {"symbol": "OLD", "date": "2020-01-01", "time": "amc"},  # past -> ignored
            {"symbol": "NOTIME", "date": "2026-09-01"},  # missing time -> None
        ]
        m = es.fetch_earnings_window_fmp("2026-06-28", "2026-10-28")
        self.assertEqual(m["AAPL"], (dt.date(2026, 7, 30), "AMC"))
        self.assertEqual(m["BRK-B"], (dt.date(2026, 8, 2), "BMO"))
        self.assertNotIn("OLD", m)
        self.assertIsNone(m["NOTIME"][1])

    def test_finnhub_fallback_when_no_fmp(self):
        from data import earnings_sources as es

        os.environ["FINNHUB_API_KEY"] = "test"
        es._get_json = lambda url, timeout=20.0: {
            "earningsCalendar": [
                {"symbol": "MSFT", "date": "2026-07-25", "hour": "amc"},
                {"symbol": "TSLA", "date": "2026-07-20", "hour": "bmo"},
            ]
        }
        m, src = es.fetch_earnings_window("2026-06-28", "2026-10-28")
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
