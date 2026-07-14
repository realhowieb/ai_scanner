"""Tests for the Arrow-safe dataframe sanitizer (the segfault fix)."""
import datetime as dt
import unittest

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

requires_pandas = unittest.skipIf(pd is None, "pandas not installed")


@requires_pandas
class ArrowSafeTests(unittest.TestCase):
    def test_object_date_column_becomes_datetime64(self):
        from ui.arrow_safe import arrow_safe

        # The exact crash shape: psycopg date objects mixed with None.
        df = pd.DataFrame(
            {"symbol": ["AAPL", "MSFT"], "earnings_date": [dt.date(2026, 7, 15), None]}
        )
        self.assertEqual(df["earnings_date"].dtype, object)
        out = arrow_safe(df)
        self.assertTrue(str(out["earnings_date"].dtype).startswith("datetime64"))
        self.assertEqual(out["symbol"].dtype, object)  # strings untouched

    def test_datetime_objects_converted_too(self):
        from ui.arrow_safe import arrow_safe

        df = pd.DataFrame({"created_at": [dt.datetime(2026, 7, 1, 12), None]})
        out = arrow_safe(df)
        self.assertTrue(str(out["created_at"].dtype).startswith("datetime64"))

    def test_none_and_empty_passthrough(self):
        from ui.arrow_safe import arrow_safe

        self.assertIsNone(arrow_safe(None))
        self.assertTrue(arrow_safe(pd.DataFrame()).empty)
