"""Production-hardening checks for persisted HSF watchlists."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WatchlistHardeningSourceTests(unittest.TestCase):
    def test_schema_tracks_default_and_membership_metadata(self):
        source = (ROOT / "db" / "schema.py").read_text()

        self.assertIn("is_default", source)
        self.assertIn("date_added", source)
        self.assertIn("price_when_added", source)
        self.assertIn("note TEXT", source)
        self.assertIn("idx_watchlist_items_unique_symbol", source)

    def test_data_layer_exposes_lifecycle_and_bulk_operations(self):
        import db.watchlists as wl

        for name in (
            "get_default_watchlist_id",
            "set_default_watchlist",
            "rename_watchlist",
            "duplicate_watchlist",
            "add_tickers_to_watchlist",
            "remove_tickers_from_watchlist",
            "copy_tickers_between_watchlists",
            "move_tickers_between_watchlists",
            "update_watchlist_item_note",
            "get_watchlist_items",
        ):
            self.assertTrue(callable(getattr(wl, name, None)), name)

    def test_watchlist_name_normalization_rejects_blank_and_bounds_length(self):
        from db.watchlists import normalize_watchlist_name

        self.assertEqual(normalize_watchlist_name("  Day   Trades  "), "Day Trades")
        self.assertEqual(normalize_watchlist_name("   "), "")
        self.assertLessEqual(len(normalize_watchlist_name("X" * 200)), 80)

    def test_management_ui_wires_default_duplicate_notes_and_bulk_actions(self):
        source = (ROOT / "ui" / "watchlists.py").read_text()

        self.assertIn("Make this my default watchlist", source)
        self.assertIn("Set as Default", source)
        self.assertIn("Duplicate", source)
        self.assertIn("Copy Selected", source)
        self.assertIn("Move Selected", source)
        self.assertIn("Remove Selected", source)
        self.assertIn("Ticker notes", source)
        self.assertIn("Confirm delete", source)

    def test_result_watchlist_action_uses_centralized_add_path(self):
        source = (ROOT / "ui" / "result_watchlist.py").read_text()

        self.assertIn("add_tickers_to_watchlist", source)
        self.assertIn("list_watchlists", source)
        self.assertIn("get_default_watchlist_id", source)
        self.assertIn("Destination", source)


class DefaultRepairTests(unittest.TestCase):
    def test_default_repair_selects_one_deterministic_default(self):
        from db.watchlists import _repair_default_watchlist

        conn = _FakeConn(
            [
                {"id": 3, "is_default": False},
                {"id": 2, "is_default": True},
                {"id": 1, "is_default": True},
            ]
        )

        self.assertEqual(_repair_default_watchlist("user@example.com", conn), 2)
        self.assertEqual(conn.updated_default, 2)
        self.assertTrue(conn.committed)

    def test_default_repair_assigns_when_missing(self):
        from db.watchlists import _repair_default_watchlist

        conn = _FakeConn([{"id": 4, "is_default": False}, {"id": 3, "is_default": False}])

        self.assertEqual(_repair_default_watchlist("user@example.com", conn), 4)
        self.assertEqual(conn.updated_default, 4)


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.updated_default = None
        self.committed = False

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.committed = True


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._rows = []

    def execute(self, sql, params=()):
        if "SELECT id, is_default" in sql:
            self._rows = list(self.conn.rows)
        elif "UPDATE watchlists SET is_default" in sql:
            self.conn.updated_default = int(params[0])
            self._rows = []

    def fetchall(self):
        return self._rows

    def close(self):
        return None


if __name__ == "__main__":
    unittest.main()
