"""Run 33 — Alpaca 1-minute bar loader + no-lookahead observation builder."""
import unittest
from unittest import mock


class FetchMinuteBarsTests(unittest.TestCase):
    def _fake_requests(self, pages):
        calls = {"n": 0}
        fake = mock.MagicMock()

        def _get(url, headers=None, params=None, timeout=None):
            resp = mock.MagicMock()
            resp.status_code = 200
            resp.json.return_value = pages[min(calls["n"], len(pages) - 1)]
            resp.raise_for_status.return_value = None
            calls["n"] += 1
            return resp

        fake.get.side_effect = _get
        return fake

    def test_paginates_and_orders(self):
        from data import price_alpaca as pa
        pages = [
            {"bars": [{"t": "T1", "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 100}], "next_page_token": "p2"},
            {"bars": [{"t": "T2", "o": 1.5, "h": 2, "l": 1, "c": 1.8, "v": 200}], "next_page_token": None},
        ]
        with (
            mock.patch.object(pa, "requests", self._fake_requests(pages)),
            mock.patch.object(pa, "get_alpaca_config",
                              return_value={"data_url": "https://x", "api_key": "k", "api_secret": "s"}),
            mock.patch.object(pa, "get_alpaca_data_feed", return_value="iex"),
        ):
            bars = pa.fetch_minute_bars("NVDA", "2026-09-01")
        self.assertEqual([b["t"] for b in bars], ["T1", "T2"])
        self.assertEqual(bars[0]["c"], 1.5)

    def test_no_config_returns_empty(self):
        from data import price_alpaca as pa
        with mock.patch.object(pa, "get_alpaca_config", return_value=None):
            self.assertEqual(pa.fetch_minute_bars("NVDA", "2026-09-01"), [])

    def test_error_fails_safe(self):
        from data import price_alpaca as pa
        fake = mock.MagicMock()
        fake.get.side_effect = RuntimeError("network")
        with (
            mock.patch.object(pa, "requests", fake),
            mock.patch.object(pa, "get_alpaca_config",
                              return_value={"data_url": "https://x", "api_key": "k", "api_secret": "s"}),
            mock.patch.object(pa, "get_alpaca_data_feed", return_value="iex"),
        ):
            self.assertEqual(pa.fetch_minute_bars("NVDA", "2026-09-01"), [])


class ObservationBuilderTests(unittest.TestCase):
    def test_no_lookahead_and_features(self):
        from scripts.validate_day_trade_score import minute_bars_to_observations
        # rising session: open 100, climbing; sample every 2 bars
        bars = [{"t": f"T{i}", "o": 100, "h": 100 + i + 0.5, "l": 100 + i - 0.5,
                 "c": 100 + i, "v": 1000} for i in range(40)]
        obs = minute_bars_to_observations("NVDA", bars, sample_every=2)
        self.assertTrue(obs)
        first = obs[0]
        # intraday-derivable features present; daily indicators honestly absent
        self.assertIsNotNone(first["direction"])
        self.assertEqual(first["ticker"], "NVDA")
        # a bar late enough to have +15 future bars gets a real 15m outcome
        late = next(o for o in obs if o.get("directional_return_15m") is not None)
        self.assertIsInstance(late["directional_return_15m"], float)

    def test_too_few_bars(self):
        from scripts.validate_day_trade_score import minute_bars_to_observations
        self.assertEqual(minute_bars_to_observations("X", [{"t": "T0", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}]), [])


if __name__ == "__main__":
    unittest.main()
