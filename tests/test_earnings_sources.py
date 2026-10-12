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

    def test_slices_cover_window(self):
        from data.earnings_sources import _slices

        self.assertEqual(_slices("2026-10-01", "2026-10-15", 7),
                         [("2026-10-01", "2026-10-07"), ("2026-10-08", "2026-10-14"), ("2026-10-15", "2026-10-15")])
        self.assertEqual(_slices("2026-10-01", "2026-10-01", 7), [("2026-10-01", "2026-10-01")])

    def test_finnhub_cap_does_not_drop_next_week(self):
        """Finnhub returns at most ~1,500 rows, latest dates first. One 120-day request
        lost the coming week (DPZ, JPM, WFC...); sliced requests must keep it."""
        import urllib.parse
        from datetime import date, timedelta

        from data import earnings_sources as es

        today = date.today()
        cal = [{"symbol": "DPZ", "date": (today + timedelta(days=2)).isoformat(), "hour": "bmo"},
               {"symbol": "JPM", "date": (today + timedelta(days=2)).isoformat(), "hour": "bmo"}]
        for i in range(3000):  # a crowded calendar further out
            cal.append({"symbol": f"X{i}", "date": (today + timedelta(days=10 + i % 100)).isoformat(), "hour": ""})
        calls = []

        def fake(url, timeout=20.0):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            calls.append((q["from"], q["to"]))
            rows = [r for r in cal if q["from"] <= r["date"] <= q["to"]]
            rows.sort(key=lambda r: r["date"], reverse=True)
            return {"earningsCalendar": rows[:1500]}

        os.environ["FINNHUB_API_KEY"] = "test"
        es._get_json = fake
        m = es.fetch_earnings_window_finnhub(today.isoformat(), (today + timedelta(days=120)).isoformat())
        self.assertEqual(m["DPZ"], (today + timedelta(days=2), "BMO"))
        self.assertIn("JPM", m)
        self.assertEqual(len([k for k in m if k.startswith("X")]), 3000)
        self.assertGreater(len(calls), 1)

    def test_finnhub_busy_slice_is_split(self):
        import urllib.parse
        from datetime import date, timedelta

        from data import earnings_sources as es

        today = date.today()
        day = lambda n: (today + timedelta(days=n)).isoformat()
        cal = [{"symbol": f"B{i}", "date": day(1 + i % 7), "hour": "amc"} for i in range(2100)]

        def fake(url, timeout=20.0):
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            rows = sorted((r for r in cal if q["from"] <= r["date"] <= q["to"]), key=lambda r: r["date"], reverse=True)
            return {"earningsCalendar": rows[:1500]}

        os.environ["FINNHUB_API_KEY"] = "test"
        es._get_json = fake
        m = es.fetch_earnings_window_finnhub(day(1), day(7))
        self.assertEqual(len(m), 2100)

    def test_fmp_missing_time_keeps_finnhub_time(self):
        """FMP without a time must not blank Finnhub's BMO for the same date (JPM showed
        "After close" because the stale stored time survived)."""
        import urllib.parse
        from datetime import date, timedelta

        from data import earnings_sources as es

        today = date.today()
        d = (today + timedelta(days=1)).isoformat()
        later = (today + timedelta(days=3)).isoformat()
        seen = []

        def fake(url, timeout=20.0):
            if "finnhub" in url:
                return {"earningsCalendar": [{"symbol": "JPM", "date": d, "hour": "bmo"},
                                             {"symbol": "XYZ", "date": d, "hour": "amc"}]}
            seen.append(dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query)))
            return [{"symbol": "JPM", "date": d},              # no time
                    {"symbol": "XYZ", "date": later},          # different date: FMP wins, no time
                    {"symbol": "GS", "date": d, "time": "bmo"}]

        os.environ["FMP_API_KEY"] = "test"
        os.environ["FINNHUB_API_KEY"] = "test"
        es._get_json = fake
        m, src = es.fetch_earnings_window(today.isoformat(), (today + timedelta(days=6)).isoformat())
        self.assertEqual(src, "fmp+finnhub")
        self.assertEqual(m["JPM"], (date.fromisoformat(d), "BMO"))
        self.assertEqual(m["XYZ"], (date.fromisoformat(later), None))
        self.assertEqual(m["GS"][1], "BMO")
        self.assertTrue(all(q.get("includeReportTimes") == "true" for q in seen))

    def test_time_normalization(self):
        from data.earnings_sources import _norm_time

        self.assertEqual(_norm_time("amc"), "AMC")
        self.assertEqual(_norm_time("BMO"), "BMO")
        self.assertEqual(_norm_time("after market close"), "AMC")
        self.assertIsNone(_norm_time("dmh"))
        self.assertIsNone(_norm_time(None))


if __name__ == "__main__":
    unittest.main()
