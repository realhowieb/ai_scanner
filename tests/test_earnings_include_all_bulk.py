import unittest
from datetime import date, timedelta
from unittest import mock


class _Cur:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.rows.append(params)


class _Conn:
    def __init__(self):
        self.rows = []

    def cursor(self):
        return _Cur(self.rows)

    def commit(self):
        pass


class IncludeAllBulkTest(unittest.TestCase):
    def _run(self, **kw):
        from db import earnings

        d = date.today() + timedelta(days=3)
        bulk = {"AAPL": (d, "amc"), "ARM": (d, "amc"), "BRK-B": (d, "bmo"), "7203-T": (d, None)}
        conn = _Conn()
        with mock.patch.object(earnings, "ensure_earnings_table", lambda c=None: None), \
             mock.patch("data.earnings_sources.fetch_earnings_window", return_value=(bulk, "fmp")):
            out = earnings.populate_earnings_calendar(
                ["AAPL", "BRK-B"], conn=conn, use_yf_fallback=False, sleep_s=0, **kw)
        return out, {r[0] for r in conn.rows}

    def test_default_only_universe(self):
        out, stored = self._run()
        self.assertEqual(stored, {"AAPL", "BRKB"})  # bulk BRK-B now matches universe BRK-B
        self.assertEqual(set(out), {"AAPL", "BRKB"})

    def test_include_all_bulk_adds_us_symbols(self):
        out, stored = self._run(include_all_bulk=True)
        self.assertEqual(stored, {"AAPL", "BRKB", "ARM"})  # foreign 7203T skipped
        self.assertIn("ARM", out)


if __name__ == "__main__":
    unittest.main()
