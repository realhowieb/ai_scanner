"""Tests for ui/ai_screener.py — natural-language → filter parsing."""
from __future__ import annotations

import unittest
from unittest.mock import patch


class CoerceTest(unittest.TestCase):
    def test_universe_normalization(self):
        from ui.ai_screener import _coerce
        self.assertEqual(_coerce("universe", "nasdaq"), "NASDAQ")
        self.assertEqual(_coerce("universe", "S&P500"), "SP500")
        self.assertEqual(_coerce("universe", "combo"), "Combo")
        self.assertIsNone(_coerce("universe", "russell2000"))

    def test_numeric_and_bool_coercion(self):
        from ui.ai_screener import _coerce
        self.assertEqual(_coerce("max_price", "20"), 20.0)
        self.assertEqual(_coerce("top_n", "25.0"), 25)
        self.assertTrue(_coerce("apply_gap_filter", "true"))
        self.assertFalse(_coerce("unusual_vol", "no"))

    def test_unknown_key_rejected(self):
        from ui.ai_screener import _coerce
        self.assertIsNone(_coerce("delete_everything", "yes"))


class ParseScreenRequestTest(unittest.TestCase):
    def test_empty_query(self):
        from ui.ai_screener import parse_screen_request
        filters, expl, err = parse_screen_request("   ")
        self.assertEqual(filters, {})
        self.assertIsNotNone(err)

    def test_parses_and_validates_model_json(self):
        from ui import ai_screener
        payload = (
            '{"universe":"NASDAQ","max_price":20,"apply_gap_filter":true,'
            '"min_gap":2,"unusual_vol":true,"bogus":"x",'
            '"explanation":"NASDAQ under $20 gapping up"}'
        )
        with patch("ui.ai.ask_claude", return_value=(payload, None)):
            filters, expl, err = ai_screener.parse_screen_request("nasdaq under 20 gapping up")
        self.assertIsNone(err)
        self.assertEqual(filters["universe"], "NASDAQ")
        self.assertEqual(filters["max_price"], 20.0)
        self.assertTrue(filters["apply_gap_filter"])
        self.assertNotIn("bogus", filters)  # disallowed key dropped
        self.assertIn("NASDAQ", expl)

    def test_handles_markdown_fenced_json(self):
        from ui import ai_screener
        payload = '```json\n{"max_price": 5, "explanation": "cheap"}\n```'
        with patch("ui.ai.ask_claude", return_value=(payload, None)):
            filters, expl, err = ai_screener.parse_screen_request("cheap stocks")
        self.assertIsNone(err)
        self.assertEqual(filters["max_price"], 5.0)

    def test_unparseable_response(self):
        from ui import ai_screener
        with patch("ui.ai.ask_claude", return_value=("no json here", None)):
            filters, expl, err = ai_screener.parse_screen_request("x")
        self.assertEqual(filters, {})
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
