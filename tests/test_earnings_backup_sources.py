"""Nasdaq calendar + Yahoo top-up: the backups for names Finnhub's free calendar
skips (TD, RY, NVO, RACE, ENB...)."""
import datetime as dt
import os
import unittest
import urllib.parse
from unittest import mock


def _next_weekday(n):
    d = dt.date.today() + dt.timedelta(days=n)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


class NasdaqSourceTest(unittest.TestCase):
    def setUp(self):
        for k in ("FMP_API_KEY", "FINNHUB_API_KEY", "HSF_NASDAQ_EARNINGS"):
            os.environ.pop(k, None)

    tearDown = setUp

    def _fake(self, by_day, finnhub=None, calls=None):
        def fake(url, timeout=20.0, headers=None):
            if "finnhub" in url:
                return {"earningsCalendar": finnhub or []}
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            if calls is not None:
                calls.append(q["date"])
                self.assertIn("Mozilla", (headers or {}).get("User-Agent", ""))
            return {"data": {"rows": by_day.get(q["date"], [])}}
        return fake

    def test_nasdaq_parses_rows_and_times(self):
        from data import earnings_sources as es

        d = _next_weekday(5)
        calls = []
        rows = {d.isoformat(): [{"symbol": "RY", "time": "time-pre-market"},
                                {"symbol": "TD", "time": "time-not-supplied"},
                                {"symbol": "XYZ", "time": "time-after-hours"}]}
        with mock.patch.object(es, "_get_json", self._fake(rows, calls=calls)):
            m = es.fetch_earnings_window_nasdaq(dt.date.today().isoformat(),
                                                (dt.date.today() + dt.timedelta(days=120)).isoformat())
        self.assertEqual(m["RY"], (d, "BMO"))
        self.assertEqual(m["TD"], (d, None))
        self.assertEqual(m["XYZ"], (d, "AMC"))
        # Weekdays only, and no further out than NASDAQ_MAX_DAYS.
        self.assertTrue(all(dt.date.fromisoformat(c).weekday() < 5 for c in calls))
        self.assertLessEqual(max(dt.date.fromisoformat(c) for c in calls),
                             dt.date.today() + dt.timedelta(days=es.NASDAQ_MAX_DAYS))

    def test_nasdaq_stops_after_repeated_errors(self):
        from data import earnings_sources as es

        calls = []

        def boom(url, timeout=20.0, headers=None):
            calls.append(url)
            raise OSError("blocked")
        with mock.patch.object(es, "_get_json", boom):
            m = es.fetch_earnings_window_nasdaq(dt.date.today().isoformat(),
                                                (dt.date.today() + dt.timedelta(days=60)).isoformat())
        self.assertEqual(m, {})
        self.assertEqual(len(calls), es.NASDAQ_MAX_ERRORS)

    def test_kill_switch(self):
        from data import earnings_sources as es

        os.environ["HSF_NASDAQ_EARNINGS"] = "0"
        with mock.patch.object(es, "_get_json", side_effect=AssertionError("called")):
            self.assertEqual(es.fetch_earnings_window_nasdaq("2026-11-09", "2026-12-09"), {})

    def test_nasdaq_fills_gaps_but_finnhub_keeps_its_dates(self):
        from data import earnings_sources as es

        d1, d2 = _next_weekday(3), _next_weekday(10)
        nas = {d1.isoformat(): [{"symbol": "TD", "time": "time-pre-market"},
                                {"symbol": "JPM", "time": "time-pre-market"}],
               d2.isoformat(): [{"symbol": "MSFT", "time": "time-after-hours"}]}
        fin = [{"symbol": "JPM", "date": d1.isoformat(), "hour": ""},      # no time -> Nasdaq's
               {"symbol": "MSFT", "date": d1.isoformat(), "hour": "amc"}]  # Finnhub date wins
        os.environ["FINNHUB_API_KEY"] = "test"
        with mock.patch.object(es, "_get_json", self._fake(nas, finnhub=fin)):
            m, src = es.fetch_earnings_window(dt.date.today().isoformat(),
                                              (dt.date.today() + dt.timedelta(days=30)).isoformat(),
                                              nasdaq=True)
        self.assertEqual(src, "finnhub+nasdaq")
        self.assertEqual(m["TD"], (d1, "BMO"))
        self.assertEqual(m["JPM"], (d1, "BMO"))
        self.assertEqual(m["MSFT"], (d1, "AMC"))

    def test_nasdaq_off_by_default(self):
        from data import earnings_sources as es

        os.environ["FINNHUB_API_KEY"] = "test"
        fake = mock.Mock(return_value={"earningsCalendar": []})
        with mock.patch.object(es, "_get_json", fake):
            es.fetch_earnings_window("2026-11-09", "2026-11-20")
        self.assertFalse(any("nasdaq" in str(c) for c in fake.call_args_list))

    def test_fmp_has_no_dead_v3_endpoint(self):
        from data import earnings_sources as es

        self.assertFalse(any("/api/v3/" in b for b in es.FMP_BASES))


class YahooTest(unittest.TestCase):
    def test_only_future_dates_count(self):
        import sys
        import types

        from data import earnings_sources as es

        today = dt.date.today()
        cals = {"NVO": {"Earnings Date": [today + dt.timedelta(days=90)]},
                "RACE": {"Earnings Date": [today - dt.timedelta(days=6)]},   # last report
                "RNG": {"Earnings Date": [today + dt.timedelta(days=9), today + dt.timedelta(days=14)]},
                "MODG": {}}
        fake_yf = types.SimpleNamespace(Ticker=lambda s: types.SimpleNamespace(calendar=cals[s]))
        with mock.patch.dict(sys.modules, {"yfinance": fake_yf}):
            self.assertEqual(es.fetch_next_earnings_yahoo("NVO"), today + dt.timedelta(days=90))
            self.assertIsNone(es.fetch_next_earnings_yahoo("RACE"))
            self.assertEqual(es.fetch_next_earnings_yahoo("rng"), today + dt.timedelta(days=9))
            self.assertIsNone(es.fetch_next_earnings_yahoo("MODG"))


class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.conn.sql.append((sql, params))

    def fetchall(self):
        return [(s,) for s in self.conn.dated]


class _Conn:
    def __init__(self, dated):
        self.dated, self.sql = dated, []

    def cursor(self):
        return _Cur(self)

    def commit(self):
        pass


class TopupTest(unittest.TestCase):
    def test_topup_looks_up_only_undated_and_respects_limit(self):
        from db import earnings

        d = dt.date.today() + dt.timedelta(days=20)
        conn = _Conn(dated=["AAPL"])
        asked = []

        def fetch(sym):
            asked.append(sym)
            return d if sym == "NVO" else None
        with mock.patch.object(earnings, "ensure_earnings_table", lambda c=None: None):
            found, tried = earnings.topup_earnings_from_yahoo(
                ["AAPL", "nvo", "MODG", "TD", "NVO"], limit=2, conn=conn, fetch=fetch, sleep_s=0)
        self.assertEqual(asked, ["NVO", "MODG"])
        self.assertEqual((found, tried), (1, 2))
        upserts = [p for s, p in conn.sql if "INSERT INTO earnings_calendar" in s]
        self.assertEqual(upserts, [("NVO", d, None)])


if __name__ == "__main__":
    unittest.main()
