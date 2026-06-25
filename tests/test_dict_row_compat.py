"""Guards against the dict_row vs tuple cursor bug class.

Production connects psycopg with row_factory=dict_row, so query results are
dicts keyed by column name. Code that used row[0] (or tuple-unpacked rows)
raised KeyError: 0 in production while passing locally on sqlite3.Row.
"""
from __future__ import annotations

import importlib.util
import unittest

_PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(_PANDAS_AVAILABLE, "db.users requires pandas")
class ScalarHelperTest(unittest.TestCase):
    def test_scalar_handles_tuple(self):
        from db.users import _scalar
        self.assertEqual(_scalar((5,)), 5)

    def test_scalar_handles_dict(self):
        from db.users import _scalar
        # psycopg dict_row for SELECT COUNT(*) -> {"count": 5}
        self.assertEqual(_scalar({"count": 5}), 5)

    def test_scalar_handles_none_and_empty(self):
        from db.users import _scalar
        self.assertIsNone(_scalar(None))
        self.assertIsNone(_scalar(()))
        self.assertIsNone(_scalar({}))


if __name__ == "__main__":
    unittest.main()
