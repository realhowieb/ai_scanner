"""BTC data helpers: 24h stats parsing for the live price header."""
from __future__ import annotations

import unittest
from unittest import mock


class Btc24hStatsTests(unittest.TestCase):
    def _resp(self, code, data):
        r = mock.MagicMock()
        r.status_code = code
        r.json.return_value = data
        return r

    def test_parses_and_computes_change(self):
        import data.crypto_btc as cb

        req = mock.MagicMock()
        req.get.return_value = self._resp(
            200, {"open": "60000", "last": "63000", "high": "63500", "low": "59500"}
        )
        with mock.patch.object(cb, "requests", req):
            s = cb.btc_24h_stats()
        self.assertEqual(s["price"], 63000.0)
        self.assertAlmostEqual(s["change_pct"], 5.0)      # (63000-60000)/60000
        self.assertEqual(s["high_24h"], 63500.0)

    def test_http_error_returns_none(self):
        import data.crypto_btc as cb

        req = mock.MagicMock()
        req.get.return_value = self._resp(500, {})
        with mock.patch.object(cb, "requests", req):
            self.assertIsNone(cb.btc_24h_stats())


if __name__ == "__main__":
    unittest.main()
