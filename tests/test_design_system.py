from __future__ import annotations

import unittest

from ui.design_system import (
    event_label,
    hsf_score_line,
    hsf_score_value,
    status_label,
    ticker_label,
)


class DesignSystemTests(unittest.TestCase):
    def test_ticker_status_and_event_labels_are_canonical(self):
        self.assertEqual(ticker_label(" nvda "), "NVDA")
        self.assertEqual(status_label("strong"), "STRONG")
        self.assertEqual(status_label(None), "UNRANKED")
        self.assertEqual(event_label("STATUS_UPGRADE"), "Status upgraded")
        self.assertEqual(event_label("DROPPED"), "Left ranking")

    def test_hsf_score_formatting_is_consistent(self):
        self.assertEqual(hsf_score_value(78.4), "78")
        self.assertEqual(hsf_score_value(None), "--")
        self.assertEqual(hsf_score_line(78.4, "watch"), "HSF Score 78 · WATCH")
        self.assertEqual(hsf_score_line(None, None), "HSF Score --")


if __name__ == "__main__":
    unittest.main()
